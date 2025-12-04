import os
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware

# Disable Flash Attention for MIG
os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["FLASHATTENTION_DISABLED"] = "1"
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"

MODEL_NAME = "openai/gpt-oss-20b"

print("=== Loading GPT-OSS-20B ===")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
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
async def generate(prompt: str = Form(...)):
    messages = [
        {"role": "system", "content": "Reasoning: high"},
        {"role": "user",   "content": prompt}
    ]

    # ⭐ THIS IS THE MAGIC LINE YOU WERE MISSING
    inputs = model.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    output = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return JSONResponse({"response": decoded})


@app.get("/")
def root():
    return {"status": "gpt-oss-20b is live!"}
