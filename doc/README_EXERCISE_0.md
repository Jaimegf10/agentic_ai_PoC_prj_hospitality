# 🏨 Exercise 0 — Simple Agentic Assistant with File Context

## 📋 Objective

The goal of **Exercise 0** is to build a simple AI agentic assistant capable of answering questions about hotels and rooms by **directly injecting hotel files into the LLM context**, without using a vector database or RAG.

This exercise serves as an introduction to agentic scaffolding and highlights the limitations of direct file-context approaches before moving to scalable RAG solutions.

---

## 📁 Data Preparation

A reduced dataset of **3 synthetic hotels** was generated to keep the context small and manageable.

### Steps performed:
- Configured the generator to produce **3 hotels**
- Generated synthetic hotel and booking data using:
  ```bash
  python bookings-db/src/gen_synthetic_hotels.py
