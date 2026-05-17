struct Dims {
    N: u32,
    batch: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<uniform> dims: Dims;
@group(0) @binding(2) var<storage, read_write> P: array<u32>;

@compute @workgroup_size(256)
fn lu_pivot(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= dims.batch) {
        return;
    }
    
    // Explicit use of P to ensure it's in the layout
    if (P[0] == 0xffffffffu) { return; }

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

@compute @workgroup_size(16, 16)
fn lu_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y + dims.k + 1u;
    let j = global_id.x + dims.k + 1u;
    let b = global_id.z;
    
    let n = dims.N;
    let k = dims.k;
    
    if (b >= dims.batch || i >= n) {
        return;
    }
    
    // Explicit use of P to ensure it's in the layout
    if (P[0] == 0xffffffffu) { return; }

    let offset = b * n * n;
    
    if (j == k + 1u) {
        let pivotVal = A[offset + k * n + k];
        if (abs(pivotVal) > 1e-9) {
            A[offset + i * n + k] /= pivotVal;
        }
    }
    
    workgroupBarrier();
    
    if (j < n) {
        A[offset + i * n + j] -= A[offset + i * n + k] * A[offset + k * n + j];
    }
}
