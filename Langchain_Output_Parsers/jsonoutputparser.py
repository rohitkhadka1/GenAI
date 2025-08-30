from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os 

load_dotenv(r"E:\GenAI\.env")

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),  
    # max_new_tokens=512,
    # temperature = 0.7
)

# Seems like Jsonoutputparser does not work well with parameters: max_new_tokens and temperature included

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me a detailed recipe of the following food item {food item} \n {format_instruction}',
    input_variables=['food item'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'food item':'pasta'})

print(result)
