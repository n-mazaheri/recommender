---
title: Movie Recommender
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0"
app_file: app.main
pinned: false
---
# 🎬 Movie Recommender System (FastAPI + LangGraph + FAISS)

This project is an AI-powered **movie recommender system**.  
It uses **FAISS vector search**, **local embeddings**, and **LLMs (via OpenRouter)** to recommend movies in **any language**.  

The pipeline:
1. Detects the language of the user query.
2. Translates the query into English.
3. Retrieves similar movies using embeddings + FAISS.
4. Generates natural language explanations with an LLM.
5. Translates the explanations back into the user’s language.

---

## ✨ Features
- Multilingual support (query in any language 🌍).
- Fast similarity search with **FAISS**.
- Local embeddings with [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Explanations powered by **OpenRouter LLMs**.
- Modular pipeline built with **LangGraph**.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI
- **Vector DB**: FAISS
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Orchestration**: LangChain + LangGraph
- **LLM**: OpenRouter (Meta LLaMA Scout free by default)
- **Deployment**: Docker / Hugging Face Spaces

---

## 📂 Project Structure
```
.
├── app/
│   ├── main.py           # FastAPI entry point
│   ├── recommender.py    # Core recommender logic
│   ├── graph.py          # LangGraph workflow
│   └── utils.py          # Helper functions
├── data/                 # Movies dataset
├── faiss_index/          # Prebuilt FAISS index + metadata
├── prepare_data.py       # Script to build FAISS index
├── requirements.txt
├── .env                  # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & Setup
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the project root:
```ini
OPENROUTER=your_openrouter_api_key
```

### 3. Prepare FAISS Index
If not already included:
```bash
python prepare_data.py
```

This builds:
- `faiss_index/movies_index.faiss`
- `faiss_index/movies.pkl`

### 4. Run FastAPI App
```bash
uvicorn app.main:app --reload
```

Backend will start at:  
👉 http://127.0.0.1:8000

Interactive API docs at:  
👉 http://127.0.0.1:8000/docs

---

## 📌 Example Usage

### Request
```bash
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d '{"query": "لطفا یک فیلم فانتزی هیجان انگیز شاد بهم معرفی کن", "k": 5}'
```

### Response
```json
[
  {
    "title": "The Incredibles",
    "genres": "Action|Animation|Adventure",
    "overview": "A family of superheroes...",
    "explanation": "این فیلم یک ماجراجویی شاد و هیجان‌انگیز است که با درخواست شما مطابقت دارد."
  },
  ...
]
```

---

## 🐳 Deployment with Docker
Build and run locally:
```bash
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```

For Hugging Face Spaces:  
- Only `/tmp` is writable at runtime.  
- Pre-download embeddings + FAISS index during build.  

---

## 🧩 Next Steps
- Add **user profiles** for personalized recommendations.
- Support **hybrid search** (metadata + embeddings).
- Add **Next.js frontend** for a full-stack app.
- Deploy to **Hugging Face Spaces**.

---

## 📜 License
MIT License. Free to use & modify.
