from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

template3 = PromptTemplate(
    template = "Write a one-line catchy title for the following summary: \n {summary}, and strictly give a one-line title without any additional explanation.",
    input_variables = ['summary']
)

parser = StrOutputParser()

# Create separate chains
detailed_chain = template1 | model | parser
summary_chain = template2 | model | parser
title_chain = template3 | model | parser

# topic = input("Enter the topic you want to generate content on: ")
# detailed_report = detailed_chain.invoke({"topic": topic})
# summary_chain = summary_chain.invoke({"text": detailed_report})
# result = title_chain.invoke({'summary':summary_chain})

# print(result)

chain = template1 | model | parser | template2 | model | parser | template3 | model | parser 
chain.get_graph().print_ascii()