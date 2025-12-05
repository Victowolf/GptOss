import os
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Disable FlashAttention (optional)
os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"

MODEL_NAME = "openai/gpt-oss-20b"

print("=== Loading GPT-OSS-20B with PIPELINE ===")

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask_gptoss")
async def ask(prompt: str = Form(...)):
    messages = [
        {"role": "system", "content": "You are an expert in earth observation."},
        {"role": "user", "content": prompt},
    ]

    # Generate output using Harmony template
    out = pipe(messages, max_new_tokens=512)

    # ✔ Correct extraction from pipeline output
    reply = out[0]["generated_text"][0]["content"]

    return JSONResponse({"response": reply})

@app.get("/")
def root():
    return {"status": "gpt-oss-20b is live via pipeline!"}
