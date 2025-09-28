from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import RecommendRequest, RecommendResponse
from app.recommender import Recommender
from app.langgraph_flow import build_graph

app = FastAPI(title="Movie Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = Recommender()
graph = build_graph(recommender)

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    state = {"query": req.query, "k": req.k}
    result = graph.invoke(state)
    return {"recommendations": result["recommendations"]}

@app.get("/health")
async def health():
    return {"status": "ok"}
