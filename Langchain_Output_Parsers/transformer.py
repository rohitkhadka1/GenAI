from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv(r"E:\GenAI\.env")
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Test with transformers library directly
pipe = pipeline("text-generation", 
                model="gpt2", 
                token=token,
                max_length=100)

result = pipe("Generate a simple pasta recipe:")
print(result)