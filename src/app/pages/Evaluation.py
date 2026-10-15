import streamlit as st

st.set_page_config(page_title="BioGraph AI – Evaluation", layout="wide")

st.title("Evaluation")
st.warning(
    "Research prototype. These are the results of our own experiments on Hetionet, "
    "reported against simple baselines."
)

st.subheader("Protocol A: AUROC and AUPRC against random negatives")
st.dataframe(
    [
        {"Method": "Degree baseline (no learning)", "AUROC": "0.912 ± 0.018", "AUPRC": "0.909 ± 0.016"},
        {"Method": "GNN (HGT + DistMult)", "AUROC": "0.887 ± 0.019", "AUPRC": "0.877 ± 0.022"},
        {"Method": "GNN, disjoint-edge training", "AUROC": "0.858 ± 0.021", "AUPRC": "0.834 ± 0.030"},
    ],
    hide_index=True,
)
st.caption("Five seeds (42–46), mean ± standard deviation. The no-learning baseline beats the GNN.")

st.subheader("Protocol B: filtered ranking over 137 candidate diseases")
st.dataframe(
    [
        {"Method": "Shared genes + popularity", "MRR": 0.287, "Hits@10": 0.467},
        {"Method": "Disease popularity", "MRR": 0.205, "Hits@10": 0.413},
        {"Method": "Shared genes only", "MRR": 0.233, "Hits@10": 0.373},
        {"Method": "GNN, random features", "MRR": 0.172, "Hits@10": 0.360},
        {"Method": "GNN, learned embeddings", "MRR": 0.116, "Hits@10": 0.320},
    ],
    hide_index=True,
)
st.caption("Seed 42, 75 test positives. Random ranking would give an MRR near 0.04.")

st.subheader("Explainability")
st.write(
    "Removing a shared gene changes the GNN score by at most 0.004 on three case pairs "
    "(Irbesartan–hypertension, Ramipril–type 2 diabetes mellitus, Epirubicin–prostate cancer). "
    "The model leans on popularity, not on gene paths."
)