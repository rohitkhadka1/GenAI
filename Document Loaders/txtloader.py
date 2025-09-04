from langchain_community.document_loaders import TextLoader
path = r"E:\GenAI\Document Loaders\files\CFA.txt"
loader = TextLoader(path, encoding="utf8")
data = loader.load()
print(data[0].page_content)
print(data[0].metadata)
