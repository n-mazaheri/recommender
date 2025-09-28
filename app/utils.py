import os
from transformers import AutoModel, AutoTokenizer
import torch
from dotenv import load_dotenv
from langchain.schema.embeddings import Embeddings


load_dotenv()  # ✅ make sure .env is read

class GemmaEmbeddings:
    def __init__(self, model_name="google/embeddinggemma-300m", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        hf_token = os.environ.get("HUGGINGFACETOEN")
        if not hf_token:
            raise ValueError("❌ Hugging Face token not found. Please set HF_TOKEN in .env")

        # ✅ Pass token when loading
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token).to(self.device)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        encodings = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encodings)

        embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.tolist()



class GemmaLangChainEmbeddings(Embeddings):
    def __init__(self, model_name="google/embeddinggemma-300m"):
        self.gemma = GemmaEmbeddings(model_name=model_name)

    def embed_query(self, text: str):
        return self.gemma.embed(text)[0]

    def embed_documents(self, texts: list[str]):
        return self.gemma.embed(texts)



