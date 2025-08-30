from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()

# Define the model 

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b", 
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(..., description="The person's full name")
    age: int = Field(..., gt= 18, description="The person's age in years")
    email: str = Field(..., description="The person's email address")
    expertise: str = Field(..., description = "The perq son's area of expertise")


parser = PydanticOutputParser(pydantic_object= Person)

template = PromptTemplate(
        template='Generate a random person with the following details about a fictional man from {country} but the man\'s ' \
        'first name should start from the letter R: \n {format_instruction}',
        input_variables=["country"],
        partial_variables={'format_instruction': parser.get_format_instructions()}
    )

chain = template | model | parser

result = chain.invoke({'country': 'Nepal'})

print(result)