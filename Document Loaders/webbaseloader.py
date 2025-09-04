from langchain_community.document_loaders import WebBaseLoader
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
loader = WebBaseLoader(url)

data = loader.load()
print(data[0].page_content)
print(data[0].metadata)
print(len(data))










# Uses BeautifulSoup4 and Requests under the hood
# Does not handle Javascript heavy websites well. Use SeleniumURLLoader for that 