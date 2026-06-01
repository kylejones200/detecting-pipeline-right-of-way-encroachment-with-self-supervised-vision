import numpy as np

def generate_embedding_clusters(n_normal=4000, n_vegetation=500, n_structure=300, n_vehicle=200, seed=42):
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for n, lab, center, scale in [
        (n_normal, 0, 0.0, 0.25), (n_vegetation, 1, 0.6, 0.35),
        (n_structure, 2, 1.3, 0.5), (n_vehicle, 3, -0.9, 0.4),
    ]:
        e = rng.normal(0, scale, (n, 384))
        e[:, 0] += center
        parts.append(e)
        labels.extend([lab] * n)
    return np.vstack(parts).ravel(), np.array(labels, dtype=np.int32)
