struct Dims {
    N: u32,
    M: u32, // num RHS columns (for B)
    batch: u32,
    k: u32, // current step
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

// Forward substitution step (for lower triangular)
// AX = B => X_kj = (B_kj - sum_{s=0}^{k-1} A_ks * X_sj) / A_kk
@compute @workgroup_size(256)
fn forward_sub_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x; // column of B
    let b = global_id.y; // batch
    if (j >= dims.M || b >= dims.batch) {
        return;
    }
    
    let n = dims.N;
    let k = dims.k;
    let offsetA = b * n * n;
    let offsetB = b * n * dims.M;
    
    var val = B[offsetB + k * dims.M + j];
    for (var s = 0u; s < k; s++) {
        val -= A[offsetA + k * n + s] * B[offsetB + s * dims.M + j];
    }
    
    B[offsetB + k * dims.M + j] = val / A[offsetA + k * n + k];
}

// Backward substitution step (for upper triangular)
// AX = B => X_kj = (B_kj - sum_{s=k+1}^{n-1} A_ks * X_sj) / A_kk
@compute @workgroup_size(256)
fn backward_sub_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x; // column of B
    let b = global_id.y; // batch
    if (j >= dims.M || b >= dims.batch) {
        return;
    }
    
    let n = dims.N;
    let k = dims.k;
    let offsetA = b * n * n;
    let offsetB = b * n * dims.M;
    
    var val = B[offsetB + k * dims.M + j];
    for (var s = k + 1u; s < n; s++) {
        val -= A[offsetA + k * n + s] * B[offsetB + s * dims.M + j];
    }
    
    B[offsetB + k * dims.M + j] = val / A[offsetA + k * n + k];
}
