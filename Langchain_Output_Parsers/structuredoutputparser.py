from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="myth1", description="Myth 1 about the {country}"),
    ResponseSchema(name="myth2", description="Myth 2 about the {country}"),
]


parser = StructuredOutputParser.from_response_schemas(schema)   

template = PromptTemplate(
    template='List two common myths about {country} in the following format: \n {format_instruction}',
    input_variables=['country'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'country':'Nepal'})

print(result)