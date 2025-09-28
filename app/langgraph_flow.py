from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain.schema import Document


from typing import TypedDict, List, Dict, Optional
from langchain.schema import Document

class State(TypedDict):
    query: str                     # user query (any language)
    user_lang: str                 # detected language (e.g., "es")
    k: int                          # ✅ add this line
    translated_query: Optional[str] # query in English
    docs: Optional[List[Document]]
    recommendations: Optional[List[Dict]]

from langgraph.graph import StateGraph, END

def build_graph(recommender):
    graph = StateGraph(State)

    # Stage 1: Detect + translate query
    def translate_in(state: State):
        user_lang = recommender.detect_language(state["query"])
        translated_query = state["query"]
        if user_lang != "en":
            translated_query = recommender.translate(state["query"], "en")
        return {"user_lang": user_lang, "translated_query": translated_query}

    # Stage 2: Retrieval
    def retrieve(state: State):
        docs = recommender.search(state["translated_query"], k=state["k"] * 2)
        return {"docs": docs}

    # Stage 3: Explanation (in English)
    def explain(state: State):
        recs = recommender.explain(state["translated_query"], state["docs"][: state["k"]], user_lang="en")
        return {"recommendations": recs}

    # Stage 4: Translate explanations back
    def translate_out(state: State):
        if state["user_lang"] != "en":
            for r in state["recommendations"]:
                r["explanation"] = recommender.translate(r["explanation"], state["user_lang"])
        return {"recommendations": state["recommendations"]}

    # Build graph
    graph.add_node("translate_in", translate_in)
    graph.add_node("retrieve", retrieve)
    graph.add_node("explain", explain)
    graph.add_node("translate_out", translate_out)

    graph.set_entry_point("translate_in")
    graph.add_edge("translate_in", "retrieve")
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", "translate_out")
    graph.add_edge("translate_out", END)

    return graph.compile()

