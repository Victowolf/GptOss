import os
import re
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# ------------------------------------------------------------
# Environment configuration (optional but safe for stability)
# ------------------------------------------------------------
os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"

# ------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-20b"

print("=== Loading GPT-OSS-20B ===")

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ------------------------------------------------------------
# FastAPI Setup
# ------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow any domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Harmony Prompt Builder
# ------------------------------------------------------------
def create_harmony_prompt(system_msg, developer_msg, user_msg):
    """
    Generates a valid Harmony-format input block for GPT-OSS.
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

# ------------------------------------------------------------
# Extract the assistant "final" channel
# ------------------------------------------------------------
def extract_final_channel(output_text: str):
    """
    Extracts the <|channel|>final message from GPT-OSS output.
    Falls back gracefully if tags are missing.
    """
    # Pattern for proper final channel
    pattern = r"<\|channel\|>final<\|message\|>(.*?)(<\|return\|>|$)"
    match = re.search(pattern, output_text, flags=re.S)
    if match:
        return match.group(1).strip()

    # Fallback: simple extraction of assistant content
    fallback = re.search(r"assistant(?:.*?)(.*)", output_text, flags=re.S)
    if fallback:
        return fallback.group(1).strip()

    return output_text.strip()


# ------------------------------------------------------------
# API Endpoint
# ------------------------------------------------------------
@app.post("/ask_gptoss")
async def ask_gptoss(prompt: str = Form(...)):
    # Build Harmony prompt
    harmony_prompt = create_harmony_prompt(
        system_msg="You are an expert in Earth observation. Reasoning: medium.",
        developer_msg="Be factual and follow EO standards.",
        user_msg=prompt
    )

    # Run generation
    output = pipe(harmony_prompt, max_new_tokens=512, return_full_text=True)

    # Pipeline output always contains generated_text
    raw_text = output[0]["generated_text"]

    # Extract final answer
    reply = extract_final_channel(raw_text)

    return JSONResponse({"response": reply})


# ------------------------------------------------------------
# Health check route
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "GPT-OSS Harmony FastAPI server is running!"}
