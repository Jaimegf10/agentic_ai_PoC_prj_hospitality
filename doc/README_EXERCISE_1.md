# 📘 README_EXERCISE_1.md
## Exercise 1 – Hotel Details with Retrieval-Augmented Generation (RAG)

---

## 🎯 Objective

The objective of **Exercise 1** is to extend the assistant created in **Exercise 0** by implementing a **Retrieval-Augmented Generation (RAG)** architecture.

Instead of relying on small in-memory datasets or hardcoded logic, this exercise introduces:

- Vector embeddings  
- A persistent vector database  
- Semantic document retrieval  
- Context-grounded response generation using an LLM  

This approach enables scalability, higher accuracy, and prepares the system for real-world data volumes.

---

## 🧠 High-Level Architecture

User Question  
↓  
Vector Retriever (ChromaDB)  
↓  
Relevant Hotel Documents  
↓  
LLM (OpenAI / Gemini)  
↓  
Final Answer  

---

## 📂 Data Sources

The RAG system uses synthetic hotel data generated with:

`python bookings-db/src/gen_synthetic_hotels.py`

The following files are consumed by the agent:

- **hotels.json** → Structured hotel metadata (name, city, category, policies)
- **hotel_details.md** → Descriptive hotel information
- **hotel_rooms.md** → Room types, capacities, prices and seasons

All files are located in:

`bookings-db/output_files/hotels/`

---

## 🧩 Phase 1 – Document Loading

All document loading logic is implemented in:

`agents/hotel_rag_agent.py`

### Loading hotels.json (JSONLoader)

```python
documents += JSONLoader(
    file_path=str(HOTELS_JSON),
    jq_schema=".",
    text_content=False,
).load()
```

This step allows the assistant to search semantically over structured hotel metadata.

---

### Loading hotel_details.md (TextLoader)

```python
documents += TextLoader(
    str(HOTEL_DETAILS_MD),
    encoding="utf-8"
).load()
```

This file contains descriptive and narrative hotel information.

---

### Loading hotel_rooms.md (TextLoader)

```python
documents += TextLoader(
    str(HOTEL_ROOMS_MD),
    encoding="utf-8"
).load()
```

This enables semantic search over room types, prices, capacities, and seasons.

---

## ✂️ Phase 2 – Text Splitting

Documents are split into overlapping chunks before embedding:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)
```

This balances semantic context with efficient retrieval.

---

## 🧬 Phase 3 – Embeddings

```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

Chosen for cost-efficiency and semantic accuracy.

---

## 🗄️ Phase 4 – Vector Store (ChromaDB)

```python
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
```

Embeddings are persisted to disk to avoid recomputation.

---

## 🤖 Phase 5 – RAG Chain

### LLM Configuration

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)
```

Deterministic output ensures consistent factual responses.

---

### Prompt Design

```python
prompt = ChatPromptTemplate.from_template(
    """You are a hotel assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{input}
"""
)
```

Prevents hallucinations and enforces grounded answers.

---

### Retrieval Chain

```python
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain,
)
```

---

## 🔌 Agent Interface

```python
def answer_hotel_question_rag(question: str) -> str:
    result = rag_chain.invoke({"input": question})
    return result["answer"]
```

---

## 🧪 Testing

Run:

```bash
python test_exercise_1.py
```

Test cases include:
- Hotel listings
- Addresses
- Meal plans
- Room comparisons
- Policy questions

---

## ⚠️ Known Limitations

- API quota limits may cause `429` or `401` errors
- These issues are external and not related to code correctness

---

## 📊 Comparison with Exercise 0

| Feature | Exercise 0 | Exercise 1 |
|------|-----------|-----------|
| Hardcoded responses | Yes | No |
| Semantic retrieval | No | Yes |
| Vector database | No | Yes |
| Scalable design | No | Yes |
| Context grounding | Partial | Full |

---

## 🏁 Conclusion

Exercise 1 implements a complete RAG architecture, enabling scalable, accurate and context-aware hotel information retrieval.  
This forms the foundation for **Exercise 2 – Booking Analytics with SQL Agent**.
