import streamlit as st

from src.app.biograph_service import BioGraphService

st.set_page_config(page_title="BioGraph AI", layout="wide")


@st.cache_resource
def get_service():
    return BioGraphService()


st.title("BioGraph AI")
st.warning(
    "Research prototype. Rankings are hypotheses from the Hetionet graph "
    "for experts to check. This is not medical advice."
)

service = get_service()
names = service.drug_names()
default = names.index("Metformin") if "Metformin" in names else 0

drug = st.selectbox("Drug", names, index=default)
top_n = st.slider("Number of diseases", 5, 30, 10)
include_known = st.checkbox("Include diseases already linked to this drug in Hetionet")

rows = service.rank_diseases(drug, top_n=top_n, include_known=include_known)

st.subheader(f"Candidate diseases for {drug}")
table = [
    {
        "Rank": i + 1,
        "Disease": r["disease"],
        "Shared genes": r["shared_gene_count"],
        "GNN score": round(r["gnn_prob"], 3),
        "Known in Hetionet": r["known"],
    }
    for i, r in enumerate(rows)
]
st.dataframe(table, hide_index=True)

st.subheader("Evidence: genes shared by the drug and the disease")
for i, r in enumerate(rows):
    title = f"{i + 1}. {r['disease']} ({r['shared_gene_count']} shared genes)"
    with st.expander(title):
        st.write(", ".join(r["shared_genes"]) or "No shared genes")

st.caption(
    "Ranking: number of shared genes, then disease popularity. In our test this "
    "rule reached MRR 0.287, against 0.172 for the GNN. The GNN score is "
    "uncalibrated and is not a probability of treatment."
)