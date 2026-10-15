from torch_geometric.transforms import RandomLinkSplit, ToUndirected

TARGET_EDGE = ("drug", "treats", "disease")
REV_TARGET_EDGE = ("disease", "rev_treats", "drug")

def split_drug_disease_edges(data, val_ratio: float = 0.1, test_ratio: float = 0.1):
    data = ToUndirected()(data)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        disjoint_train_ratio=0.0,
        edge_types=[TARGET_EDGE],
        rev_edge_types=[REV_TARGET_EDGE],
    )

    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data