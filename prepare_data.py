'''
import pandas as pd
import numpy as np
import faiss, pickle, os
from app.utils import GemmaEmbeddings

def build_index(
    csv_path="data/movies.csv",
    out_dir="faiss_index",
    batch_size=32,
    checkpoint_size=1000
):
    df = pd.read_csv(csv_path)
    texts = df["overview"].fillna("").tolist()
    total = len(texts)

    os.makedirs(out_dir, exist_ok=True)
    embedder = GemmaEmbeddings()

    embeddings = []
    start_idx = 0

    # 🔹 Check for existing partial progress
    checkpoint_file = f"{out_dir}/progress.pkl"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            saved = pickle.load(f)
            embeddings = saved["embeddings"]
            start_idx = saved["next_idx"]
        print(f"🔄 Resuming from index {start_idx}")

    # 🔹 Process in batches
    for i in range(start_idx, total, batch_size):
        batch = texts[i:i+batch_size]
        vectors = embedder.embed(batch)
        embeddings.extend(vectors)
        print(f"✅ Processed {i+len(batch)} / {total}")

        # Save checkpoint every `checkpoint_size`
        if (i + batch_size) % (10*batch_size) == 0 or (i + batch_size) >= total:
            with open(checkpoint_file, "wb") as f:
                pickle.dump({
                    "embeddings": embeddings,
                    "next_idx": i + batch_size
                }, f)
            print(f"💾 Saved checkpoint at {i+batch_size}")

    # 🔹 Build FAISS index at the end
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, f"{out_dir}/movies_index.faiss")
    with open(f"{out_dir}/movies.pkl", "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    # Remove checkpoint after success
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print("🎉 Index built successfully!")

if __name__ == "__main__":
    build_index()
'''


import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_faiss(csv_path="data/movies.csv", out_dir="faiss_index"):
    df = pd.read_csv(csv_path).fillna("")

    texts, metadatas = [], []
    for _, row in df.iterrows():
        text = (
            f"Title: {row['title']}.\n"
            f"Overview: {row['overview']}.\n"
            f"Genres: {row['genres']}.\n"
            f"Director: {row['director']}.\n"
            f"Cast: {row['cast']}."
        )
        texts.append(text)
        metadatas.append({
            "id": row["id"],
            "title": row["title"],
            "genres": row["genres"],
            "overview": row["overview"],
            "director": row["director"],
            "cast": row["cast"],
            "release_date": row["release_date"],
            "vote_average": row["vote_average"],
            "popularity": row["popularity"]
        })

    # ✅ Use local MiniLM embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    os.makedirs(out_dir, exist_ok=True)
    db.save_local(out_dir)
    print(f"✅ Saved FAISS index with {len(df)} movies to {out_dir}")

if __name__ == "__main__":
    build_faiss("data/movies.csv")

