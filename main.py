# main.py
import os
import re
import torch
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import pipeline

# --------------------------------------------------------------------
# Disable FlashAttention if your hardware does not support it
# --------------------------------------------------------------------
os.environ.setdefault("FLASH_ATTENTION", "0")
os.environ.setdefault("DISABLE_FLASH_ATTENTION", "1")
os.environ.setdefault("HF_DISABLE_FLASH_ATTENTION", "1")

# --------------------------------------------------------------------
# Load GPT-OSS-20B Pipeline
# --------------------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# --------------------------------------------------------------------
# FastAPI Application
# --------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Harmony Prompt Template (FORCE FINAL CHANNEL)
# --------------------------------------------------------------------
HARMONY_TEMPLATE = """<|start|>system<|message|>
{system_msg}
<|end|>

<|start|>developer<|message|>
{developer_msg}
<|end|>

<|start|>user<|message|>
{user_msg}
<|end|>

<|start|>assistant
<|channel|>final<|message|>
"""

def build_prompt(system, developer, user):
    return HARMONY_TEMPLATE.format(
        system_msg=system.strip(),
        developer_msg=developer.strip(),
        user_msg=user.strip(),
    )

# --------------------------------------------------------------------
# Extract ONLY the final channel (ZERO analysis leakage)
# --------------------------------------------------------------------
FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*)",
    re.S
)

@app.post("/ask_gptoss")
async def ask_gptoss(prompt: str = Form(...)):

    system_msg = "You are a world-class Earth Observation analyst. Reasoning: medium."
    developer_msg = "Always respond with EO-standard terminology. NEVER output analysis, only final results."
    
    harmony_prompt = build_prompt(system_msg, developer_msg, prompt)

    # Get model output (NO prompt echo)
    out = pipe(harmony_prompt, max_new_tokens=512, return_full_text=False)

    raw = out[0]["generated_text"]

    # Extract <|channel|>final
    m = FINAL_RE.search(raw)
    if not m:
        # fallback: remove Harmony control tokens
        cleaned = re.sub(r"<\|.*?\|>", "", raw).strip()
        return JSONResponse({"response": cleaned})

    final_text = m.group(1).strip()

    # Remove any rogue analysis if model attempted to slip it in
    final_text = re.sub(r"<\|channel\|>analysis.*", "", final_text, flags=re.S).strip()

    return JSONResponse({"response": final_text})

# --------------------------------------------------------------------
# Health Endpoint
# --------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "GPT-OSS Final-Only Server Running"}
