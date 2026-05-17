struct Dims {
    N: u32,
    batch: u32,
    k: u32, // current step
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<uniform> dims: Dims;

// Pass 1: Update column k
@compute @workgroup_size(256)
fn cholesky_step1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= dims.batch) {
        return;
    }
    
    let n = dims.N;
    let k = dims.k;
    let offset = b * n * n;
    
    let akk = A[offset + k * n + k];
    if (akk <= 0.0) {
        // Matrix is not positive definite
        // In PyTorch, this would throw an error or return NaN
        // For now we just set to NaN or something
        A[offset + k * n + k] = bitcast<f32>(0x7fc00000u);
        return;
    }
    
    let sqrt_akk = sqrt(akk);
    A[offset + k * n + k] = sqrt_akk;
    
    for (var i = k + 1u; i < n; i++) {
        A[offset + i * n + k] /= sqrt_akk;
    }
}

// Pass 2: Rank-1 update of the remaining submatrix
@compute @workgroup_size(16, 16)
fn cholesky_step2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y + dims.k + 1u;
    let j = global_id.x + dims.k + 1u;
    let b = global_id.z;
    
    let n = dims.N;
    let k = dims.k;
    
    if (b >= dims.batch || i >= n || j >= n || j > i) {
        return;
    }
    
    let offset = b * n * n;
    let val = A[offset + i * n + k] * A[offset + j * n + k];
    A[offset + i * n + j] -= val;
}

// Full Cholesky for small matrices in a single pass (if N is small)
@compute @workgroup_size(256)
fn cholesky_small(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= dims.batch) {
        return;
    }
    
    let n = dims.N;
    let offset = b * n * n;
    
    for (var k = 0u; k < n; k++) {
        var akk = A[offset + k * n + k];
        for (var s = 0u; s < k; s++) {
            let lks = A[offset + k * n + s];
            akk -= lks * lks;
        }
        
        if (akk <= 0.0) {
            A[offset + k * n + k] = bitcast<f32>(0x7fc00000u);
            return;
        }
        
        let sqrt_akk = sqrt(akk);
        A[offset + k * n + k] = sqrt_akk;
        
        for (var i = k + 1u; i < n; i++) {
            var aik = A[offset + i * n + k];
            for (var s = 0u; s < k; s++) {
                aik -= A[offset + i * n + s] * A[offset + k * n + s];
            }
            A[offset + i * n + k] = aik / sqrt_akk;
        }
    }
    
    // Zero out upper triangle
    for (var i = 0u; i < n; i++) {
        for (var j = i + 1u; j < n; j++) {
            A[offset + i * n + j] = 0.0;
        }
    }
}
