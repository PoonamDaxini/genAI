# Read pdf file
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("attention.pdf")
documents = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
# Split the documents into chunks
texts = text_splitter.split_documents(documents)
# Print the first chunk
# for i, text in enumerate(texts):
#     print(f"Chunk {i+1}:")
#     print(text.page_content)
#     print()
# # Print the metadata of the first chunk
#     print(f"Metadata of Chunk {i+1}:")
#     print(text.metadata)
#     print()
# Save the chunks to a text file


from langchain_community.document_loaders import TextLoader

loader = TextLoader("speech.txt")
docs = loader.load()


speech = ""
with open("speech.txt", "r") as f:
    speech = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

text = text_splitter.create_documents([speech])
print(text[0])
print(text[1])