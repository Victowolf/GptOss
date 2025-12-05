# main.py
import os
import re
import json
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# =============================================================
#  Optional environment flags (recommended for stability)
# =============================================================
os.environ.setdefault("FLASH_ATTENTION", "0")
os.environ.setdefault("DISABLE_FLASH_ATTENTION", "1")
os.environ.setdefault("HF_DISABLE_FLASH_ATTENTION", "1")

# =============================================================
#  Model Configuration
# =============================================================
MODEL_NAME = "openai/gpt-oss-20b"
MAX_NEW_TOKENS = 512

print("=== Loading GPT-OSS-20B via Transformers Pipeline ===")
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# =============================================================
#  FastAPI Server Setup
# =============================================================
app = FastAPI(title="GPT-OSS Harmony Server (final channel only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
#  HARMONY TEMPLATE — Manual + Stable (recommended)
# =============================================================

HARMONY_PROMPT = """<|start|>system<|message|>
{system_msg}
<|end|>

<|start|>developer<|message|>
{developer_msg}
<|end|>

<|start|>user<|message|>
{user_msg}
<|end|>

<|start|>assistant
"""


def build_harmony_prompt(system_msg, developer_msg, user_msg):
    return HARMONY_PROMPT.format(
        system_msg=system_msg.strip(),
        developer_msg=developer_msg.strip(),
        user_msg=user_msg.strip(),
    )


# =============================================================
#  Extract final channel only (NO analysis leakage)
# =============================================================

FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|channel\|>|<\|end\|>|$)",
    re.S
)

ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>.*?(?=<\|channel\|>|<\|return\|>|$)",
    re.S
)


def extract_final_text(generated: str) -> str:
    """Return ONLY the <|channel|>final content. Never return analysis."""
    # 1) Remove analysis block if it exists
    generated = ANALYSIS_RE.sub("", generated)

    # 2) Extract final channel
    m = FINAL_RE.search(generated)
    if m:
        return m.group(1).strip()

    # 3) Fallback: remove Harmony control tokens and return remainder
    sanitized = re.sub(
        r"<\|start\|>|<\|end\|>|<\|channel\|>.*?<\|message\|>|<\|return\|>",
        "",
        generated
    ).strip()

    return sanitized


# =============================================================
#  Remove prompt echo (works across transformers versions)
# =============================================================

def strip_prompt_echo(generated: str, prompt: str) -> str:
    """If the model duplicated the prompt in the output, remove it."""
    norm_prompt = re.sub(r"\s+", " ", prompt).strip()
    norm_gen = re.sub(r"\s+", " ", generated).strip()

    idx = norm_gen.find(norm_prompt)
    if idx != -1:
        return generated[idx + len(prompt):].lstrip()

    return generated


# =============================================================
#  The main endpoint: returns ONLY the final answer
# =============================================================

@app.post("/ask_gptoss")
async def ask_gptoss(prompt: str = Form(...)):
    system_msg = "You are an expert in Earth observation. Reasoning: medium."
    developer_msg = (
        "Follow EO standards, be factual, concise, and DO NOT output chain-of-thought "
        "or analysis to the user."
    )

    harmony_prompt = build_harmony_prompt(system_msg, developer_msg, prompt)

    # ---- Run inference ----
    out = pipe(
        harmony_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        return_full_text=False  # prevents prompt echo in many transformers versions
    )

    if not out or "generated_text" not in out[0]:
        return JSONResponse({"error": "Unexpected model output", "raw": str(out)})

    generated = out[0]["generated_text"]

    # ---- Remove prompt echo if present ----
    if "<|start|>system" in generated:
        generated = strip_prompt_echo(generated, harmony_prompt)

    # ---- Extract ONLY the final channel ----
    final_text = extract_final_text(generated)

    return JSONResponse({"response": final_text})


# =============================================================
#  Health Check
# =============================================================
@app.get("/")
def root():
    return {"status": "GPT-OSS Harmony Server Running (final channel only)"}
