from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import re

load_dotenv(r"E:\GenAI\.env")

# Check if API token exists
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in environment variables")
    print("Please add it to your .env file")
    exit(1)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

try:
    # Initialize the model 
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.7,
        max_new_tokens=512,
    )
    print("Model initialized successfully")
    print (os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    
except Exception as e:
    print(f"Error initializing model: {e}")
    exit(1)

# Create output parser (but don't use it in the chain)
output_parser = JsonOutputParser()

# Create a more explicit prompt template
template = PromptTemplate(
    template="""You are a helpful cooking assistant. Generate a recipe based on these instructions: {instructions}

Please respond ONLY with a valid JSON object in the following format:
{{
    "name": "Recipe Name",
    "ingredients": ["ingredient 1", "ingredient 2", "ingredient 3"],
    "instructions": ["step 1", "step 2", "step 3"],
    "prep_time": "X minutes",
    "cook_time": "X minutes"
}}

Instructions: {instructions}

JSON Response:""",
    input_variables=["instructions"]
)

# Create the chain WITHOUT the output parser
chain = template | llm

# Function to clean and extract JSON from response
def extract_json(text):
    # Remove any text before the first {
    start = text.find('{')
    if start == -1:
        return None
    
    # Find the last }
    end = text.rfind('}')
    if end == -1:
        return None
    
    json_str = text[start:end+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

# Get and parse the response
try:
    print("Generating recipe...")
    # Get raw response from the model first
    raw_response = chain.invoke({"instructions": "Make a simple and quick pasta dish."})
    print("✅ Raw response received successfully!")
    print("-" * 50)
    print(raw_response)
    print("-" * 50)
    
    # Now try to parse the response
    print("\nAttempting to parse JSON...")
    
    # First try with the standard parser
    try:
        parsed_response = output_parser.parse(raw_response)
        print("✅ Successfully parsed with JsonOutputParser:")
        print(json.dumps(parsed_response, indent=2))
        
    except Exception as parse_error:
        print(f"⚠️ JsonOutputParser failed: {parse_error}")
        print("Attempting manual extraction...")
        
        # Try manual extraction
        extracted_json = extract_json(raw_response)
        if extracted_json:
            print("✅ Successfully extracted JSON manually:")
            print(json.dumps(extracted_json, indent=2))
        else:
            print("❌ Could not extract valid JSON from response")
            print("The model response doesn't contain valid JSON format")
            print("\nTip: The model might need better prompting or a different approach")
            
except Exception as e:
    print(f"❌ Error during execution: {e}")
    print(f"Error type: {type(e).__name__}")
    

        
    if "rate limit" in str(e).lower():
        print("💡 This might be a rate limiting issue. Try again later.")
    elif "token" in str(e).lower():
        print("💡 This might be an API token issue. Check your token permissions.")
    elif "model" in str(e).lower():
        print("💡 This might be a model access issue. Ensure you have access to Llama-2.")

print("\n" + "="*60)
print("If you're still getting StopIteration errors, try this alternative:")
print("="*60)

# Alternative approach with a simpler, more accessible model
try:
    print("Trying with microsoft/DialoGPT-medium (more accessible model)...")
    
    alternative_llm = HuggingFaceEndpoint(
        repo_id="microsoft/DialoGPT-medium",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.7,
        max_new_tokens=200,
    )
    
    simple_template = PromptTemplate(
        template="Generate a simple pasta recipe with ingredients and steps: {instructions}",
        input_variables=["instructions"]
    )
    
    alt_chain = simple_template | alternative_llm
    alt_response = alt_chain.invoke({"instructions": "quick pasta dish"})
    
    print("✅ Alternative model worked!")
    print(f"Response: {alt_response}")
    
except Exception as alt_error:
    print(f"Alternative model also failed: {alt_error}")
    print("This suggests a broader API connectivity issue")