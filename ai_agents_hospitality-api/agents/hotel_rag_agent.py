from pathlib import Path
import os
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# -------- Paths --------
BASE_PATH = Path(__file__).resolve().parents[2]

HOTELS_DIR = BASE_PATH / "bookings-db/output_files/hotels"
VECTORSTORE_DIR = BASE_PATH / "ai_agents_hospitality-api/vectorstore/chroma_db"

HOTELS_JSON = HOTELS_DIR / "hotels.json"
HOTEL_DETAILS_MD = HOTELS_DIR / "hotel_details.md"
HOTEL_ROOMS_MD = HOTELS_DIR / "hotel_rooms.md"


# -------- Load documents --------
documents = []

documents += JSONLoader(
    file_path=str(HOTELS_JSON),
    jq_schema=".",
    text_content=False,
).load()

documents += TextLoader(str(HOTEL_DETAILS_MD), encoding="utf-8").load()
documents += TextLoader(str(HOTEL_ROOMS_MD), encoding="utf-8").load()


# -------- Split --------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

docs = splitter.split_documents(documents)


# -------- Embeddings --------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)



# -------- Vector store --------
if VECTORSTORE_DIR.exists():
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )
else:
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    vectorstore.persist()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# -------- LLM --------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# -------- Prompt --------
prompt = ChatPromptTemplate.from_template(
    """
You are a hotel assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""
)


# -------- LCEL RAG pipeline --------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)


def answer_hotel_question_rag(question: str) -> str:
    """Answer hotel and room questions using RAG."""
    response = rag_chain.invoke(question)
    return response.content
