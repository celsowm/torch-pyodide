@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let chunk = 256u;
    let start = tid * chunk;

    var acc: f32 = 0.0;
    for (var i = 0u; i < chunk; i++) {
        let idx = start + i;
        if (idx < arrayLength(&input)) {
            acc += input[idx];
        }
    }
    result[tid] = acc;
}
