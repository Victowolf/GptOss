import os
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"

MODEL_NAME = "openai/gpt-oss-20b"

print("=== Loading GPT-OSS-20B (Chat Pipeline) ===")

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask_gptoss")
async def ask_gptoss(prompt: str = Form(...)):
    messages = [
        {"role": "system", "content": "Reasoning: high"},
        {"role": "user", "content": prompt},
    ]

    out = pipe(
        messages,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7
    )

    # HuggingFace returns a list → each item has "generated_text" → list of messages
    answer = out[0]["generated_text"][-1]["content"]

    return JSONResponse({"response": answer})


@app.get("/")
def root():
    return {"status": "gpt-oss-20b chat API is live!"}
