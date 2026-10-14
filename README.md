# BioGraph AI

Explainable Heterogeneous Graph Neural Networks for Biomedical Knowledge Graph
Completion and Drug–Disease Association Prediction.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Download data
python -m src.data.download

# 2. Build graph + train model
python main.py --mode train

# 3. Explain a prediction
python main.py --mode explain --drug Metformin --disease PCOS

# 4. Run the dashboard
streamlit run src/app/dashboard.py

# 5. Run the API
uvicorn src.app.api:app --reload
```

## Project Structure

```
biograph-ai/
├── data/                  # raw, processed, and graph data
├── src/
│   ├── data/              # download, ID mapping, graph construction, splitting
│   ├── models/             # HGT encoder + DistMult decoder
│   ├── training/            # training loop, negative sampling, evaluation
│   ├── explainability/      # attention, GNNExplainer, path-based explanations
│   ├── utils/                # config, logging
│   └── app/                  # Streamlit dashboard + FastAPI serving
├── notebooks/                 # exploratory analysis, case studies
├── experiments/configs/       # yaml configs per run
├── tests/                     # pytest unit tests
├── main.py                    # CLI entry point (train / evaluate / explain)
├── Dockerfile
└── requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Deployment

See `Dockerfile` for containerized deployment, or push to Streamlit Community
Cloud / Hugging Face Spaces for a free hosted demo.
