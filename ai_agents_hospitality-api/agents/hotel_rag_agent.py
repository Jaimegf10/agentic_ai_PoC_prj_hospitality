import os
from pathlib import Path

from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


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
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

docs = text_splitter.split_documents(documents)


# -------- Embeddings --------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("AI_AGENTIC_API_KEY"),
)


# -------- Vector Store --------
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=str(VECTORSTORE_DIR),
)
vectorstore.persist()


# -------- LLM --------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("AI_AGENTIC_API_KEY"),
)


# -------- RAG Chain --------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
)


def answer_hotel_question_rag(question: str) -> str:
    """Answer hotel and room questions using RAG."""
    return qa_chain.run(question)
