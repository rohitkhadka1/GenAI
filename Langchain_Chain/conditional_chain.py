from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field 
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)
class Review(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description = "Give the sentiment of the feedback") 

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object= Review)

sentiment = PromptTemplate(
    template = """Classify the sentiment of the following review as Positive, Negative or Neutral: \n {review}

{format_instructions}

Your response should be a JSON object matching the format above.""",
    input_variables = ['review'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = sentiment | model | parser2


movie_review = "It was awesome movie."
# result = classifier_chain.invoke({'review': movie_review}).sentiment

# print(result)
# print(result.sentiment)


prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {movie_review}',
    input_variables = ['movie_review']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {movie_review}',
    input_variables = ['movie_review']
)
branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] == 'positive', prompt2 | model | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser1),
     RunnableLambda(lambda x: print("Neutral review, no response needed"))
)

chain = classifier_chain | branch_chain

print(chain.invoke({'review': movie_review}))