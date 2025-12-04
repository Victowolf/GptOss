import os
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware

# Disable Flash Attention
os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["FLASHATTENTION_DISABLED"] = "1"
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"

MODEL_NAME = "openai/gpt-oss-20b"

print("=== Loading GPT-OSS-20B ===")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 🚨 No BitsAndBytesConfig here — the model is already MXFP4 quantized
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,       # <-- FORCE BF16
    device_map="auto",                # <-- shards across GPU and CPU
    low_cpu_mem_usage=True,           # <-- avoids full memory spike
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
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return JSONResponse({"response": result})


@app.get("/")
def root():
    return {"status": "gpt-oss-20b is live!"}
