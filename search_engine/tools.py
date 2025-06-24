# tool creation
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
print(wikipedia_tool.name)

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
print(arxiv_tool.name)

tools = [wikipedia_tool, arxiv_tool]

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = WebBaseLoader("https://www.wikipedia.org/")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
).split_documents(documents)

vectorstore = FAISS.from_documents(documents=text_splitter, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
retriver = vectorstore.as_retriever()
print(retriver)


from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever=retriver,
    name="Wikipedia Retriever",
    description="A tool to retrieve information from Wikipedia articles."
)


tools.append(retriever_tool)


