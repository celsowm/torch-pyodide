// Pairwise distance (torch.pdist): upper-triangular vector of p-norm distances.
//
// Input  A: array<f32> of shape [n, d] (row-major).
// Output D: array<f32> of shape [m] with m = n*(n-1)/2, the condensed
//           pairwise distances for i in [0, n), j in (i, n): D[k] = ||row_i - row_j||_p.
// Params n (u32), d (u32), p (f32).

struct Params {
  n: u32,
  d: u32,
  p: f32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> D: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params.n;
  let d = params.d;
  let p = params.p;
  let m = n * (n - 1u) / 2u;
  let k = gid.x;
  if (k >= m) {
    return;
  }

  // Decode output index k -> pair (i, j) with i < j, using the inverse of
  // C(i) = i*(2n - i - 1)/2 (cumulative count of pairs before row i).
  let disc = f32((2u * n - 1u) * (2u * n - 1u) - 8u * k);
  let i = u32(floor((f32(2u * n - 1u) - sqrt(max(disc, 0.0))) / 2.0));
  let j = k - i * (2u * n - i - 1u) / 2u + i + 1u;

  let base_i = i * d;
  let base_j = j * d;

  if (abs(p - 2.0) < 1e-6) {
    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
      let diff = A[base_i + c] - A[base_j + c];
      acc = acc + diff * diff;
    }
    D[k] = sqrt(acc);
  } else {
    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
      acc = acc + pow(abs(A[base_i + c] - A[base_j + c]), p);
    }
    D[k] = pow(acc, 1.0 / p);
  }
}
