# Repository

Companion code for a Medium article.

## Business context

In 2018, a backhoe struck Williams' Transco pipeline in rural Pennsylvania. Tragically one person died and five homes were destroyed. The following investigation found that unauthorized construction had been visible in aerial imagery for six weeks before the rupture. The contractor ignored right-of-way restrictions, and the pipeline operator's monthly aerial patrol had missed the encroachment by three days.

Pipeline operators manage 3 million miles of buried infrastructure across North America, most traversing private land where construction activity is unrestricted outside the narrow right-of-way corridor. Federal regulations require aerial or satellite monitoring, but traditional methods are reactive: monthly helicopter flyovers capture snapshots, and human analysts review thousands of images looking for new structures, vegetation clearing, or earth moving equipment.

Modern computer vision transforms this workflow. Instead of humans reviewing images sequentially, self-supervised models like DINOv2 (Distillation with NO labels v2) convert each aerial tile into a 384-dimensional embedding that captures semantic content—excavators look similar in embedding space, construction sites cluster together, undisturbed forest forms a distinct distribution. When a new image appears that's distant from normal operational baselines, it flags for inspection.

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — synthetic 384-d embedding cluster generation. Reference PyO3 benchmark: **comparable (see `benchmark_rust.py`)** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p detecting_pipeline_right_of_way_encroachment_with_self_supervised_vision_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).