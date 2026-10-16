"""Download Hetionet nodes and edges into data/raw/."""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

NODES_URL = "https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv"
EDGES_URL = "https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"

def download_hetionet_nodes():
    df = pd.read_csv(NODES_URL, sep="\t")
    out_path = RAW_DIR / "hetionet_nodes.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Downloaded {len(df)} nodes -> {out_path}")
    return df


def download_hetionet_edges():
    df = pd.read_csv(EDGES_URL, sep="\t", compression="gzip")
    out_path = RAW_DIR / "hetionet_edges.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Downloaded {len(df)} edges -> {out_path}")
    return df


if __name__ == "__main__":
    download_hetionet_nodes()
    download_hetionet_edges()
