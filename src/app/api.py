from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from src.app.biograph_service import BioGraphService

state = {}


@asynccontextmanager
async def lifespan(app):
    state["service"] = BioGraphService()
    yield
    state.clear()


app = FastAPI(title="BioGraph AI", lifespan=lifespan)

DISCLAIMER = (
    "Research prototype. Rankings are hypotheses from the Hetionet graph "
    "for experts to check. Not medical advice."
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "service" in state}


@app.get("/drugs")
def drugs():
    return {"drugs": state["service"].drug_names()}


@app.get("/rank")
def rank(
    drug: str,
    top_n: int = Query(10, ge=1, le=50),
    include_known: bool = False,
):
    service = state["service"]
    if drug not in service.drug_index:
        raise HTTPException(status_code=404, detail=f"Unknown drug: {drug}")

    rows = service.rank_diseases(drug, top_n=top_n, include_known=include_known)
    for row in rows:
        row["gnn_score"] = round(row.pop("gnn_prob"), 3)

    return {"drug": drug, "disclaimer": DISCLAIMER, "results": rows}