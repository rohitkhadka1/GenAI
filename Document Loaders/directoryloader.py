from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
import time
import psutil

path = r"E:\Papers"
loader = DirectoryLoader(path= path, glob = "*.pdf", 
                         loader_cls = PDFPlumberLoader)
start = time.time()
data = loader.load()
# print(data[0].page_content)
# print(data[0].metadata)
end = time.time()
print(f"Time taken to load the documents: {end - start} seconds")
print(f"Memory used: {psutil.Process().memory_info().rss / (1024 * 1024)} MB")
# print(f"Total pages: {len(data)}")
start1 = time.time()
data1 = loader.lazy_load()
end1 = time.time()
print(f"Time taken to lazy load the documents: {end1 - start1} seconds")
print(f"Memory used: {psutil.Process().memory_info().rss / (1024 * 1024)} MB")
# For a parallel document loading, and computationally efficieny, there is a concept in langchain called "Batching" and lazy loading. Please read the documentation for more details.