import os
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi.middleware.cors import CORSMiddleware

os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["FLASHATTENTION_DISABLED"] = "1"
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"

MODEL_NAME = "openai/gpt-oss-20b"

print("=== Loading GPT-OSS-20B (4bit) ===")

device = "cuda" if torch.cuda.is_available() else "cpu"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate(prompt: str = Form(...)):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return JSONResponse({"response": result})


@app.get("/")
def root():
    return {"status": "gpt-oss-20b (4bit) is live!"}
