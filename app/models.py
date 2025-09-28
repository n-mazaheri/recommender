from pydantic import BaseModel
from typing import List, Optional

class RecommendRequest(BaseModel):
    query: str
    k: Optional[int] = 5

class Recommendation(BaseModel):
    title: str
    genres: str
    overview: str
    director: Optional[str]
    cast: Optional[str]
    release_date: Optional[str]
    vote_average: Optional[float]
    explanation: str

class RecommendResponse(BaseModel):
    recommendations: List[Recommendation]
