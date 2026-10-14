"""Streamlit dashboard: pick a drug + disease, get a prediction + explanation."""
import sys
from pathlib import Path

import streamlit as st
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.biograph_model import BioGraphModel
from src.utils.config import EXPERIMENTS_DIR, MODEL_CONFIG

st.set_page_config(page_title="BioGraph AI", layout="wide")
st.title("🧬 BioGraph AI — Drug Repurposing Explorer")

CKPT_PATH = EXPERIMENTS_DIR / "best_model.pt"


@st.cache_resource
def load_model(metadata):
    model = BioGraphModel(
        MODEL_CONFIG["hidden_channels"],
        MODEL_CONFIG["out_channels"],
        MODEL_CONFIG["num_heads"],
        MODEL_CONFIG["num_layers"],
        metadata,
    )
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    else:
        st.warning("कोई trained checkpoint नहीं मिला — पहले `python main.py --mode train` चलाएं।")
    model.eval()
    return model


st.info(
    "यह dashboard एक trained checkpoint (`experiments/best_model.pt`) की उम्मीद करता है। "
    "अपना डेटा और graph object जोड़ने के बाद drug/disease लिस्ट यहाँ dropdown में दिखेंगी।"
)

col1, col2 = st.columns(2)
drug = col1.text_input("दवा का नाम (drug id)", "Metformin")
disease = col2.text_input("बीमारी का नाम (disease id)", "PCOS")

if st.button("Predict & Explain", type="primary"):
    st.warning(
        "यह placeholder है — अपने graph_data, node_id_to_name और name_to_id मैपिंग को "
        "यहाँ जोड़कर असली prediction + explanation पाइपलाइन (main.py के explain मोड जैसी) चलाएं।"
    )
