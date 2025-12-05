import os
import re
import json
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# ---------------------------
# Environment (optional)
# ---------------------------
os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"

# ---------------------------
# Model config
# ---------------------------
MODEL_NAME = "openai/gpt-oss-20b"
MAX_NEW_TOKENS = 512

print("=== Loading GPT-OSS-20B pipeline ===")
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ---------------------------
# FastAPI Setup
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
    Create a Harmony-format prompt with placeholders inserted.
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
# Parsers: extract final channel safely
# ---------------------------
def extract_final_from_generated_text(generated_text: str) -> str:
    """
    Extract only the assistant's final channel from the generated_text.
    This will:
      - Look for <|channel|>final<|message|> ... <|return|> or <|end|>
      - If not found, try alternate patterns (assistant nested tag)
      - If still not found, attempt to parse HF JSON-like outputs
      - As last resort, sanitize by removing known control tokens and returning trimmed text
    NEVER returns analysis channels.
    """
    # 1) canonical final channel pattern
    patterns = [
        r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>",   # final then explicit return
        r"<\|channel\|>final<\|message\|>(.*?)<\|end\|>",      # final then end
        r"<\|channel\|>final<\|message\|>(.*)",               # final to end
        r"<\|start\|>assistant.*?<\|channel\|>final<\|message\|>(.*?)(?:<\|channel\|>|<\|end\|>|$)",  # nested
    ]
    for pat in patterns:
        m = re.search(pat, generated_text, flags=re.S)
        if m:
            return m.group(1).strip()

    # 2) Try to interpret structured HF outputs embedded as repr/json, e.g.
    # [{"generated_text":[{"role":"assistant","content":"..."}]}] or similar
    try:
        # find first JSON blob in text
        json_blob_match = re.search(r"(\[.*\{.*'generated_text'|\"generated_text\".*\}.*\])", generated_text, flags=re.S)
        if not json_blob_match:
            json_blob_match = re.search(r"(\{.*'generated_text'|\"generated_text\".*\})", generated_text, flags=re.S)

        if json_blob_match:
            blob = json_blob_match.group(0)
            # sanitize single quotes -> double quotes for JSON parsing if needed
            blob_clean = blob.replace("'", '"')
            # remove trailing Python None/False/True tokens (best-effort)
            blob_clean = re.sub(r"\bNone\b", "null", blob_clean)
            blob_clean = re.sub(r"\bTrue\b", "true", blob_clean)
            blob_clean = re.sub(r"\bFalse\b", "false", blob_clean)

            parsed = json.loads(blob_clean)
            # drill-down for first assistant content
            if isinstance(parsed, list) and len(parsed) > 0:
                obj = parsed[0]
                if isinstance(obj, dict) and "generated_text" in obj:
                    gen = obj["generated_text"]
                    # gen can be a list of dicts [{"role":..,"content":..}] or a string
                    if isinstance(gen, list):
                        for entry in gen:
                            if isinstance(entry, dict) and entry.get("role") == "assistant" and "content" in entry:
                                return entry["content"].strip()
                        # fallback to first content field if exists
                        if gen and isinstance(gen[0], dict) and "content" in gen[0]:
                            return gen[0]["content"].strip()
                    elif isinstance(gen, str):
                        # parse text-looking output
                        return extract_final_from_generated_text(gen)
    except Exception:
        # Ignore JSON parsing errors; we'll fallback below
        pass

    # 3) Try to find assistant content via a loose 'role'/'content' regex
    m2 = re.search(r'"role"\s*:\s*"assistant".*?"content"\s*:\s*"(.*?)"', generated_text, flags=re.S)
    if m2:
        # decode escaped characters
        return bytes(m2.group(1), "utf-8").decode("unicode_escape").strip()

    # 4) VERY LAST RESORT: sanitize control tokens and return the text minus analysis block if possible.
    # Remove common control tokens
    sanitized = re.sub(r"<\|channel\|>analysis<\|message\|>.*?(?=<\|channel\|>|<\|return\|>|$)", "", generated_text, flags=re.S)
    sanitized = re.sub(r"<\|start\|>|<\|end\|>|<\|channel\|>.*?<\|message\|>|<\|return\|>", "", sanitized)
    sanitized = sanitized.strip()

    # if sanitized is empty, fallback to a safe default message
    if not sanitized:
        return "I'm sorry — I couldn't generate a safe answer. Please try again."

    return sanitized

# ---------------------------
# API Endpoint (returns ONLY final channel)
# ---------------------------
@app.post("/ask_gptoss")
async def ask_gptoss(prompt: str = Form(...)):
    """
    Accepts form field 'prompt' and returns only the assistant 'final' message.
    The internal 'analysis' channel will NOT be returned.
    """
    # Build harmony prompt
    harmony_prompt = create_harmony_prompt(
        system_msg="You are an expert in Earth observation. Reasoning: medium.",
        developer_msg="Be factual, concise, and follow EO best practices. Do not output private chain-of-thought to users.",
        user_msg=prompt
    )

    # Generate
    out = pipe(harmony_prompt, max_new_tokens=MAX_NEW_TOKENS, return_full_text=True)

    # Validate pipeline output
    if not isinstance(out, list) or len(out) == 0 or "generated_text" not in out[0]:
        return JSONResponse({"error": "unexpected pipeline output", "raw": str(out)}, status_code=500)

    generated_text = out[0]["generated_text"]

    # Extract final channel only (safe)
    final_answer = extract_final_from_generated_text(generated_text)

    # Return only the final channel
    return JSONResponse({"response": final_answer})


# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def root():
    return {"status": "GPT-OSS Harmony server running - final channel only"}    return JSONResponse({"response": reply})


# ------------------------------------------------------------
# Health check route
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "GPT-OSS Harmony FastAPI server is running!"}
