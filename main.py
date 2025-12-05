# main.py
import os
import re
import json
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# ---------------------------
# Optional environment tweaks
# ---------------------------
os.environ.setdefault("FLASH_ATTENTION", "0")
os.environ.setdefault("DISABLE_FLASH_ATTENTION", "1")
os.environ.setdefault("HF_DISABLE_FLASH_ATTENTION", "1")

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "openai/gpt-oss-20b"
MAX_NEW_TOKENS = 512

# ---------------------------
# Load pipeline
# ---------------------------
print("=== Verifying MXFP4 / Triton (if available) ===")
# (any MXFP4/triton detection happens in environment; log shown by your environment)
print("=== Loading GPT-OSS-20B pipeline ===")
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(title="GPT-OSS Harmony FastAPI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Harmony prompt builder
# ---------------------------
def create_harmony_prompt(system_msg: str, developer_msg: str, user_msg: str) -> str:
    """
    Build a harmony-format prompt string. Insert variables where needed.
    """
    return f"""
<|start|>system<|message|>
{system_msg}
<|end|>

<|start|>developer<|message|>
{developer_msg}
<|end|>

<|start|>user<|message|>
{user_msg}
<|end|>

<|start|>assistant
""".strip()

# ---------------------------
# Extract final channel (safe)
# ---------------------------
def extract_final_from_generated_text(generated_text: str) -> str:
    """
    Extract only the assistant's 'final' channel content from generated_text.
    Never returns chain-of-thought ('analysis').
    """
    # 1) canonical <|channel|>final pattern
    patterns = [
        r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>",
        r"<\|channel\|>final<\|message\|>(.*?)<\|end\|>",
        r"<\|channel\|>final<\|message\|>(.*)",
        r"<\|start\|>assistant.*?<\|channel\|>final<\|message\|>(.*?)(?:<\|channel\|>|<\|end\|>|$)",
    ]
    for pat in patterns:
        m = re.search(pat, generated_text, flags=re.S)
        if m:
            return m.group(1).strip()

    # 2) Try to parse HF-like JSON blob embedded in text
    try:
        # Attempt to find a JSON-like substring containing "generated_text"
        json_match = re.search(r"(\[.*?\"generated_text\".*?\].*?|\{.*?\"generated_text\".*?\}.*)", generated_text, flags=re.S)
        if json_match:
            blob = json_match.group(0)
            # Normalize single quotes -> double quotes (best-effort)
            blob_clean = blob.replace("'", '"')
            blob_clean = re.sub(r"\bNone\b", "null", blob_clean)
            blob_clean = re.sub(r"\bTrue\b", "true", blob_clean)
            blob_clean = re.sub(r"\bFalse\b", "false", blob_clean)
            parsed = json.loads(blob_clean)
            if isinstance(parsed, list) and parsed:
                obj = parsed[0]
                if isinstance(obj, dict) and "generated_text" in obj:
                    gen = obj["generated_text"]
                    if isinstance(gen, list):
                        for entry in gen:
                            if isinstance(entry, dict) and entry.get("role") == "assistant" and "content" in entry:
                                return entry["content"].strip()
                        # fallback to first content
                        if gen and isinstance(gen[0], dict) and "content" in gen[0]:
                            return gen[0]["content"].strip()
                    elif isinstance(gen, str):
                        return extract_final_from_generated_text(gen)
    except Exception:
        pass

    # 3) Loose regex for role/content pairs
    m2 = re.search(r'"role"\s*:\s*"assistant".*?"content"\s*:\s*"(.*?)"', generated_text, flags=re.S)
    if m2:
        return bytes(m2.group(1), "utf-8").decode("unicode_escape").strip()

    # 4) Remove analysis channel and control tokens, return remainder
    sanitized = re.sub(r"<\|channel\|>analysis<\|message\|>.*?(?=<\|channel\|>|<\|return\|>|$)", "", generated_text, flags=re.S)
    sanitized = re.sub(r"<\|start\|>|<\|end\|>|<\|channel\|>.*?<\|message\|>|<\|return\|>", "", sanitized)
    sanitized = sanitized.strip()

    if not sanitized:
        return "I'm sorry — I couldn't produce a safe answer. Please try again."

    return sanitized

# ---------------------------
# Endpoint: ask_gptoss
# ---------------------------
@app.post("/ask_gptoss")
async def ask_gptoss(prompt: str = Form(...)):
    """
    Accepts form field 'prompt' and returns only the assistant's final message.
    """
    harmony_prompt = create_harmony_prompt(
        system_msg="You are an expert in Earth observation. Reasoning: medium.",
        developer_msg="Be factual, concise, follow EO best practices. Do NOT output internal chain-of-thought to end users.",
        user_msg=prompt
    )

    out = pipe(harmony_prompt, max_new_tokens=MAX_NEW_TOKENS, return_full_text=True)

    # Validate pipeline shape
    if not isinstance(out, list) or len(out) == 0 or "generated_text" not in out[0]:
        return JSONResponse({"error": "unexpected pipeline output", "raw": str(out)}, status_code=500)

    generated_text = out[0]["generated_text"]
    final_answer = extract_final_from_generated_text(generated_text)

    return JSONResponse({"response": final_answer})

# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def root():
    return {"status": "GPT-OSS Harmony server running - final channel only"}
