from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="gemma2-9b-it",
    api_key=groq_api_key
)


# create prompt
system_template = "Transalte the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}"),
    ]
)

# parser
parser = StrOutputParser()

# chain
chain = prompt_template | model | parser

# create a FastAPI app
app = FastAPI(
    title="LangChain Groq Translation Service",
    description="A simple translation service using LangChain and Groq.",
    version="1.0.0",
)
# create  app
add_routes(
    app,
    chain,
    path="/translate",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="info")