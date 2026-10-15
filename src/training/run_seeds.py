import numpy as np

from src.training.train import main


def run_seeds(seeds=(42, 43, 44, 45, 46)):
    results = []
    for seed in seeds:
        print(f"=== seed {seed} ===")
        results.append(main(seed=seed))

    arr = np.array(results)
    names = ["best val AUROC", "TEST AUROC", "TEST AUPRC"]
    print("\n=== summary over", len(seeds), "seeds ===")
    for i, name in enumerate(names):
        print(f"{name}: {arr[:, i].mean():.4f} +/- {arr[:, i].std():.4f}")


if __name__ == "__main__":
    run_seeds()