struct Params {
    batch_size: u32,    // outerSize: number of reduction groups
    reduce_size: u32,   // dimSize: size of the dimension being reduced
    inner_size: u32,    // innerSize: elements within each slice
    op: u32,            // operation: 0=sum, 1=mean, 2=max, 3=min, 4=prod, 5=any, 6=all
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let max_idx = params.batch_size * params.inner_size;
    if (output_idx >= max_idx) {
        return;
    }

    // Calculate which batch and inner position this work item handles
    let batch_idx = output_idx / params.inner_size;
    let inner_idx = output_idx % params.inner_size;

    var acc: f32;
    // Initialization based on op
    if (params.op == 0u || params.op == 1u || params.op == 5u) {
        acc = 0.0;
    } else if (params.op == 2u) {
        acc = -1e38;  // Approximate -FLT_MAX for max
    } else if (params.op == 3u) {
        acc = 1e38;   // Approximate FLT_MAX for min
    } else if (params.op == 4u || params.op == 6u) {
        acc = 1.0;
    }

    // Iterate over the reduce dimension, accounting for inner stride
    // global_idx = batch_idx * (reduce_size * inner_size) + r * inner_size + inner_idx
    let stride = params.reduce_size * params.inner_size;
    let batch_base = batch_idx * stride;
    for (var r = 0u; r < params.reduce_size; r++) {
        let global_idx = batch_base + r * params.inner_size + inner_idx;
        let val = input[global_idx];
        if (params.op == 0u || params.op == 1u) {
            acc += val;
        } else if (params.op == 2u) {
            acc = max(acc, val);
        } else if (params.op == 3u) {
            acc = min(acc, val);
        } else if (params.op == 4u) {
            acc *= val;
        } else if (params.op == 5u) {
            if (val != 0.0) { acc = 1.0; break; }
        } else if (params.op == 6u) {
            if (val == 0.0) { acc = 0.0; break; }
        }
    }

    if (params.op == 1u) {
        output[output_idx] = acc / f32(params.reduce_size);
    } else {
        output[output_idx] = acc;
    }
}
