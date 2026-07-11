struct Dims {
    N: u32,
    batch: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
// Pivot permutation stored as f32 to match the runtime's f32 storage
// convention (integers are stored/read back as float values).
@group(0) @binding(1) var<storage, read_write> P: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn lu_pivot(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= dims.batch) {
        return;
    }

    // Explicit use of P to ensure it's in the layout
    if (P[0] < -0.5) { return; }

    let n = dims.N;
    let k = dims.k;
    let offset = b * n * n;
    let pOffset = b * n;

    var maxVal: f32 = 0.0;
    var pivotRow: u32 = k;
    for (var i = k; i < n; i = i + 1u) {
        let val = abs(A[offset + i * n + k]);
        if (val > maxVal) {
            maxVal = val;
            pivotRow = i;
        }
    }

    if (pivotRow != k) {
        for (var j = 0u; j < n; j = j + 1u) {
            let temp = A[offset + k * n + j];
            A[offset + k * n + j] = A[offset + pivotRow * n + j];
            A[offset + pivotRow * n + j] = temp;
        }
        let tempP = P[pOffset + k];
        P[pOffset + k] = P[pOffset + pivotRow];
        P[pOffset + pivotRow] = tempP;
    }
}

// Compute the column of multipliers L[i][k] = A[i][k] / A[k][k] for i > k.
// Runs as its own dispatch so the results are visible to lu_update without an
// (illegal / insufficient) workgroup barrier.
@compute @workgroup_size(256)
fn lu_scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = dims.N;
    let b = idx / n;
    let i = idx % n;

    if (b >= dims.batch) { return; }

    // Explicit use of P to ensure it's in the layout
    if (P[0] < -0.5) { return; }

    let k = dims.k;
    if (i <= k) { return; }

    let offset = b * n * n;
    let pivotVal = A[offset + k * n + k];
    if (abs(pivotVal) > 1e-9) {
        A[offset + i * n + k] = A[offset + i * n + k] / pivotVal;
    }
}

// Schur-complement update: A[i][j] -= L[i][k] * A[k][j] for i > k, j > k.
// 1D linear indexing over batch * n * n keeps control flow uniform.
@compute @workgroup_size(256)
fn lu_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = dims.N;
    let nn = n * n;
    let b = idx / nn;
    let rem = idx % nn;
    let i = rem / n;
    let j = rem % n;

    if (b >= dims.batch) { return; }

    // Explicit use of P to ensure it's in the layout
    if (P[0] < -0.5) { return; }

    let k = dims.k;
    if (i <= k || j <= k) { return; }

    let offset = b * nn;
    A[offset + i * n + j] = A[offset + i * n + j] - A[offset + i * n + k] * A[offset + k * n + j];
}
