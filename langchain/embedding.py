import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# text to vectors conversion
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
text = "Hello, how are you doing today? I hope you are having a great day!"
query  = embeddings.embed_query(text)

print(query)
# Initialize Chroma vector store
