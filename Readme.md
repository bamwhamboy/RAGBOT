---
title: Tableau Genie
emoji: ✨
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Ask any Tableau Desktop / Web Authoring question, answered by RAG.
---

# Tableau Assistant using RAG

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot for Tableau documentation.

The chatbot answers user questions using Tableau Desktop and Tableau Web Authoring documentation.

## Architecture

PDF Documents

↓

PyMuPDF

↓

Chunking

↓

BAAI bge-small-en-v1.5 Embeddings

↓

Qdrant Vector Database

↓

Similarity Search

↓

Groq Llama 4 Scout (configurable via `GROQ_MODEL`)

↓

Answer Generation

## Tech Stack

* Python
* Streamlit
* PyMuPDF
* Sentence Transformers
* Qdrant
* Groq

## Features

* PDF ingestion
* Semantic search
* Vector database retrieval
* Source citations
* Interactive chatbot interface
* Retrieval debugging

## Sample Questions

* How do I create a calculated field?
* What is a FIXED LOD expression?
* How do parameters work?
* What is Web Authoring?

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Future Enhancements

* Hybrid Search (Vector + BM25)
* Reranking
* Conversation Memory
* Additional Tableau Documentation Sources
* Evaluation Framework
