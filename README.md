# BioGraph AI

Explainable Heterogeneous Graph Neural Networks for Biomedical Knowledge Graph
Completion and Drug–Disease Association Prediction.

The goal is **drug repurposing**: predict missing `(drug, treats, disease)` links in the
[Hetionet](https://github.com/hetio/hetionet) knowledge graph using a Heterogeneous Graph
Transformer (HGT) encoder and a DistMult decoder.

## Status

| Stage | State |
|---|---|
| Data download and ID mapping | Done |
| Heterogeneous graph (drug, gene, disease) | Done |
| Leak-free train / validation / test split | Done |
| HGT encoder + DistMult decoder | Done |
| Training loop and evaluation (AUROC, AUPRC) | In progress |
| Explainability (GNNExplainer, attention, paths) | Planned |
| Streamlit dashboard, FastAPI, Docker deployment | Planned |

## Graph

| Node type | Count | | Relation | Edges |
|---|---|---|---|---|
| drug | 1,552 | | drug treats disease (target) | 755 |
| gene | 20,945 | | drug targets gene | 11,571 |
| disease | 137 | | disease associated with gene | 12,623 |

The 755 `treats` edges are split 80 / 10 / 10 with one negative sample per positive.
Reverse copies of held-out edges are removed from the training graph to prevent data leakage.

## Setup

```bash
python -m venv venv
venv\Scripts\activate           # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Run what exists today

```bash
# 1. Download Hetionet into data/raw/
python -m src.data.download

# 2. Check the graph split (prints positive/negative counts)
python test_split.py

# 3. Check the model forward pass and initial loss
python test_model.py
```

## Project structure

```
biograph-ai/
├── src/
│   ├── data/          # download, ID mapping, graph construction, split
│   ├── models/        # HGT encoder, DistMult decoder, full model
│   └── training/      # training loop and evaluation
├── tests/             # unit tests
└── requirements.txt
```

Other folders (`explainability/`, `app/`, `utils/`) are placeholders for the planned stages.

## Notes

- Raw Hetionet files (about 90 MB) are not stored in this repository. Run the download step above.
- Node features are random for now; real biological features are a planned improvement.
- Only 3 of Hetionet's 11 node types are used so far.