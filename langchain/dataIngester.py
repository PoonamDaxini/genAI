from langchain_community.document_loaders import TextLoader

loader = TextLoader("speech.txt")
documents = loader.load()
# print(documents)

# Read pdf file
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("attention.pdf")
documents = loader.load()
# print(type(documents[0]))



from langchain_community.document_loaders import WebBaseLoader
import bs4
loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
                                      ))
documents = loader.load()
# print(documents)
# Read arxiv paper
from langchain_community.document_loaders import ArxivLoader
loader = ArxivLoader(
    query="1706.03762",
    load_max_docs=2,
    # load_all_available_meta=False
)
documents = loader.load()
print(documents)

