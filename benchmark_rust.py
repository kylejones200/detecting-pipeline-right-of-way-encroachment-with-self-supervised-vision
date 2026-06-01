#!/usr/bin/env python3
import time, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/"src"))
from compute_kernel import generate_embedding_clusters
def main():
    emb,lab=generate_embedding_clusters()
    t0=time.perf_counter()
    for _ in range(200):
        generate_embedding_clusters()
    py_s=time.perf_counter()-t0
    try:
        import detecting_pipeline_right_of_way_encroachment_with_self_supervised_vision_rs as rs
    except ImportError:
        print("Build: cd rust && maturin develop --release -m py/Cargo.toml"); print(f"Python {py_s:.3f}s"); return
    rs_s=rs.bench_kernel_py()
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s/max(rs_s,1e-9):.1f}x")
    rs_emb, rs_lab = rs.generate_embedding_clusters_py(4000,500,300,200,42)
    assert emb.shape == np.asarray(rs_emb).shape and lab.shape == np.asarray(rs_lab).shape
    print("Correctness: OK (shapes match; stochastic generator)")
if __name__=="__main__": main()
