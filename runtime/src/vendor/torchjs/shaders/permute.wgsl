struct Dims {
    N: u32,
    batch: u32,
    cols: u32,
    reverse: u32,
}

@group(0) @binding(0) var<storage, read> B_in: array<f32>;
@group(0) @binding(1) var<storage, read> P: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;
@group(0) @binding(3) var<storage, read_write> B_out: array<f32>;

@compute @workgroup_size(256)
fn permute_rows(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let flatIdx = global_id.x;
    let b = global_id.y;

    if (b >= dims.batch) {
        return;
    }

    let n = dims.N;
    let cols = dims.cols;
    let total = n * cols;
    if (flatIdx >= total) {
        return;
    }

    let offset = b * total;
    let pOff = b * n;

    let row = flatIdx / cols;
    let col = flatIdx % cols;

    var r = row;
    if (dims.reverse == 0u) {
        for (var k = 0u; k < n; k = k + 1u) {
            let pivot = P[pOff + k];
            if (pivot == k) { continue; }
            if (r == k) { r = pivot; }
            else if (r == pivot) { r = k; }
        }
    } else {
        for (var k = n; k > 0u; k = k - 1u) {
            let kk = k - 1u;
            let pivot = P[pOff + kk];
            if (pivot == kk) { continue; }
            if (r == pivot) { r = kk; }
            else if (r == kk) { r = pivot; }
        }
    }

    let srcIdx = offset + r * cols + col;
    let dstIdx = offset + row * cols + col;
    B_out[dstIdx] = B_in[srcIdx];
}
