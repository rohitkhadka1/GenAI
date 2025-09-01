from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)


llm1 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)
model1 = ChatHuggingFace(llm=llm1)
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model | parser,
    'quiz' : prompt2 | model1 | parser
}
)

merge_chain = prompt3 | model | parser
text = """
A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape from it. 
The boundary of this region is called the event horizon. Black holes are formed when massive stars 
collapse under their own gravity at the end of their life cycles. They can also form through the merging of smaller black holes or from the remnants of supernovae.
Black holes can be classified into three main types based on their mass: stellar-mass black holes,
intermediate-mass black holes, and supermassive black holes. Stellar-mass black holes typically have masses ranging from a few to tens of times that of our Sun.
Intermediate-mass black holes have masses ranging from hundreds to thousands of solar masses, while supermassive black holes can have masses of millions to billions of times that of the Sun and are often found at the centers of galaxies.
Black holes can be detected through their interactions with nearby matter. When matter falls into a black hole, it heats up and emits X-rays and other forms of radiation, which can be observed by telescopes. Additionally, the gravitational effects of black holes on nearby stars and gas clouds can provide indirect evidence of their presence.
Black holes play a crucial role in our understanding of the universe, as they are key to studying
the nature of gravity, spacetime, and the behavior of matter under extreme conditions. They also have implications for theories of quantum mechanics and general relativity.
"""

chain = parallel_chain | merge_chain

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()