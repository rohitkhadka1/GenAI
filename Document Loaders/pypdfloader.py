from langchain_community.document_loaders import PyPDFLoader
path = r"E:\GenAI\Document Loaders\files\Agentic AI demystified book.pdf"
path1 = r"E:\GenAI\Document Loaders\files\A Survey on activations functions.pdf"
loader = PyPDFLoader(path)
loader1 = PyPDFLoader(path1)
data = loader.load()
data1 = loader1.load()
print(data1[0].page_content)
# print(data[0].metadata)


#Simple,clean PDFs -> PyPDFLoader
#PDFs with complex layouts and scanned/images -> UnstructuredPDFLoader
#PDFs with tables/columns -> PDFPlumberLoader
#Need layout and image data -> PyMuPDFLoader