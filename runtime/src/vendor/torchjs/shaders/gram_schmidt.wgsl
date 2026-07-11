// Gram-Schmidt orthonormalization of the ROWS of a [rows, cols] matrix.
//
// Used by nn.init.orthogonal_ to produce a matrix with orthonormal rows
// (Q Qᵀ = I), computed entirely on the GPU (no CPU readback of the matrix).
//
// Input  A: array<f32> of shape [rows, cols] (row-major, read-only).
// Output Q: array<f32> of shape [rows, cols] (read_write). The running vector
//           for row i is kept in Q[i] itself as it is orthogonalized, so no
//           large private scratch buffer is needed.
// Params rows (u32), cols (u32).
//
// The algorithm is inherently sequential across rows, so a single invocation
// drives it (sufficient for weight-init matrices). The inner loops over the
// d columns still run on the GPU.

struct Params {
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Q: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  let rows = params.rows;
  let cols = params.cols;

  for (var i: u32 = 0u; i < rows; i = i + 1u) {
    let base_i = i * cols;
    // Start from a copy of row i of A.
    for (var c: u32 = 0u; c < cols; c = c + 1u) {
      Q[base_i + c] = A[base_i + c];
    }
    // Subtract projections onto every previously computed orthonormal row.
    for (var krow: u32 = 0u; krow < i; krow = krow + 1u) {
      let base_k = krow * cols;
      var dot: f32 = 0.0;
      for (var c: u32 = 0u; c < cols; c = c + 1u) {
        dot = dot + Q[base_i + c] * Q[base_k + c];
      }
      for (var c: u32 = 0u; c < cols; c = c + 1u) {
        Q[base_i + c] = Q[base_i + c] - dot * Q[base_k + c];
      }
    }
    // Normalize to unit length.
    var norm: f32 = 0.0;
    for (var c: u32 = 0u; c < cols; c = c + 1u) {
      norm = norm + Q[base_i + c] * Q[base_i + c];
    }
    norm = sqrt(norm);
    if (norm > 1e-10) {
      let inv = 1.0 / norm;
      for (var c: u32 = 0u; c < cols; c = c + 1u) {
        Q[base_i + c] = Q[base_i + c] * inv;
      }
    } else {
      for (var c: u32 = 0u; c < cols; c = c + 1u) {
        Q[base_i + c] = 0.0;
      }
    }
  }
}
