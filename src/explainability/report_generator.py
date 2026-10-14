"""Turn a biological path + score into a human-readable explanation string."""


def generate_explanation_text(drug_name, disease_name, top_paths, node_id_to_name):
    explanations = []
    for path, score in top_paths:
        readable_path = " -> ".join(node_id_to_name.get(n, str(n)) for n in path)
        explanations.append(
            f"'{drug_name}' may treat '{disease_name}' via: {readable_path} "
            f"(confidence score: {score:.3f})"
        )
    return "\n".join(explanations) if explanations else "No biological path found."
