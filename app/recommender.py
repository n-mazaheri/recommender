import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langdetect import detect
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ


class Recommender:
    def __init__(self, index_dir="faiss_index"):
        # ✅ Embeddings (English only)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"cache_dir": "/tmp/hf_cache"}
        )
        self.db = FAISS.load_local(
            index_dir, self.embeddings, allow_dangerous_deserialization=True
        )

        # ✅ OpenRouter LLM (used for explanations + translation)
        self.llmExplanation = ChatOpenAI(
            openai_api_key=os.environ["OPENROUTER"],
            openai_api_base="https://openrouter.ai/api/v1",
            model="meta-llama/llama-4-scout:free",
            temperature=0,
            max_tokens=512,
        )
        self.llmTranslation = ChatOpenAI(
            openai_api_key=os.environ["OPENROUTER"],
            openai_api_base="https://openrouter.ai/api/v1",
            model="meta-llama/llama-4-scout:free",  # switch here
            temperature=0,
            max_tokens=512
        )

    # 🔹 Stage 1a: Language detection
    def detect_language(self, text: str) -> str:
        return detect(text)

    # 🔹 Stage 1b + 4: Translation (to/from English)
    def translate(self, text: str, target_lang: str = "en") -> str:
        prompt = f"Translate this text into {target_lang}: {text}"
        return self.llmTranslation.invoke(prompt).content

    # 🔹 Stage 2: Retrieval
    def search(self, query: str, k: int = 10):
        return self.db.similarity_search(query, k=k)

    # 🔹 Stage 3: Explanation (always in English)
    def explain(self, query: str, docs, user_lang="en"):
        results = []
        for d in docs:
            prompt = (
                f"User request: {query}\n"
                f"Candidate movie: {d.metadata['title']} "
                f"({d.metadata.get('genres')}).\n"
                f"Overview: {d.metadata.get('overview')}\n\n"
                "Explain in one sentence why this movie could be a good recommendation "
                "for the user’s request. Focus only on positive connections."
            )
            response = self.llmExplanation.invoke(prompt).content

            results.append({
                "title": d.metadata["title"],
                "genres": d.metadata["genres"],
                "overview": d.metadata["overview"],
                "director": d.metadata.get("director"),
                "cast": d.metadata.get("cast"),
                "release_date": d.metadata.get("release_date"),
                "vote_average": d.metadata.get("vote_average"),
                "explanation": response,  # always English at this stage
            })
        return results








