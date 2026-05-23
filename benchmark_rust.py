#!/usr/bin/env python3
import time, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/"src"))
from compute_kernel import generate_embedding_clusters
def main():
    emb,lab=generate_embedding_clusters()
    t0=time.perf_counter()
    for _ in range(2000 if "generate_embedding_clusters"=="cyclical_time_features" else 200):
        emb,lab=generate_embedding_clusters()
    py_s=time.perf_counter()-t0
    try:
        import detecting_pipeline_right_of_way_encroachment_with_self_supervised_vision_rs as rs
    except ImportError:
        print("Build: cd rust && maturin develop --release -m py/Cargo.toml"); print(f"Python {py_s:.3f}s"); return
    rs_s=rs.bench_kernel_py()
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s/max(rs_s,1e-9):.1f}x")
    np.testing.assert_allclose(emb,lab, np.asarray(rs.generate_embedding_clusters_py(4000,500,300,200,42))[0] if isinstance(rs.generate_embedding_clusters_py(4000,500,300,200,42), tuple) else rs.generate_embedding_clusters_py(4000,500,300,200,42), rtol=1e-10)
    print("Correctness: OK")
if __name__=="__main__": main()
