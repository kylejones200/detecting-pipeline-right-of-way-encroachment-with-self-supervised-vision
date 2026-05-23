use detecting_pipeline_right_of_way_encroachment_with_self_supervised_vision_core::generate_embedding_clusters;
use numpy::{PyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (n_normal, n_vegetation, n_structure, n_vehicle, seed=42))]
fn generate_embedding_clusters_py<'py>(
    py: Python<'py>, n_normal: usize, n_vegetation: usize, n_structure: usize, n_vehicle: usize, seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>)> {
    let (emb, labels) = generate_embedding_clusters(n_normal, n_vegetation, n_structure, n_vehicle, seed);
    Ok((emb.into_pyarray(py), labels.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (n_normal=4000, n_vegetation=500, n_structure=300, n_vehicle=200, seed=42, iterations=50))]
fn bench_kernel_py(n_normal: usize, n_vegetation: usize, n_structure: usize, n_vehicle: usize, seed: u64, iterations: usize) -> PyResult<f64> {
    let start = std::time::Instant::now();
    for _ in 0..iterations { let _ = generate_embedding_clusters(n_normal, n_vegetation, n_structure, n_vehicle, seed); }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn detecting_pipeline_right_of_way_encroachment_with_self_supervised_vision_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_embedding_clusters_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
