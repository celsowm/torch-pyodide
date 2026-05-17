struct Params {
    length: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
       @builtin(local_invocation_id) local_id: vec3<u32>,
       @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let idx = global_id.x;
    let lid = local_id.x;
    if (idx < params.length) {
        shared_data[lid] = input[idx];
    } else {
        shared_data[lid] = 1.0;
    }
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_data[lid] *= shared_data[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        result[workgroup_id.x] = shared_data[0];
    }
}
