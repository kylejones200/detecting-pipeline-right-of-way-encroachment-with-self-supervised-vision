//! Synthetic 384-d Gaussian embedding clusters (DINOv2-style).

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as f64 / (1u64 << 31) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

const DIM: usize = 384;

/// Row-major embeddings `n_images * 384` and integer cluster labels.
pub fn generate_embedding_clusters(
    n_normal: usize,
    n_vegetation: usize,
    n_structure: usize,
    n_vehicle: usize,
    seed: u64,
) -> (Vec<f64>, Vec<i32>) {
    let mut rng = Lcg::new(seed);
    let total = n_normal + n_vegetation + n_structure + n_vehicle;
    let mut embeddings = vec![0.0; total * DIM];
    let mut labels = Vec::with_capacity(total);

    let mut push_cluster =
        |rng: &mut Lcg, out: &mut Vec<f64>, labels: &mut Vec<i32>, n: usize, label: i32, center: f64, scale: f64| {
            for _ in 0..n {
                labels.push(label);
                let base = out.len();
                for d in 0..DIM {
                    let c = if d == 0 { center } else { 0.0 };
                    out.push(c + rng.normal() * scale);
                }
                let _ = base;
            }
        };

    push_cluster(&mut rng, &mut embeddings, &mut labels, n_normal, 0, 0.0, 0.25);
    push_cluster(&mut rng, &mut embeddings, &mut labels, n_vegetation, 1, 0.6, 0.35);
    push_cluster(&mut rng, &mut embeddings, &mut labels, n_structure, 2, 1.3, 0.5);
    push_cluster(&mut rng, &mut embeddings, &mut labels, n_vehicle, 3, -0.9, 0.4);

    (embeddings, labels)
}
