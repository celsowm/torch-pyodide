struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    pad_top: u32,
    pad_bottom: u32,
    pad_left: u32,
    pad_right: u32,
    value_bits: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.channels * params.out_h * params.out_w;
    if (idx >= total) { return; }

    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let c = (idx / (params.out_w * params.out_h)) % params.channels;
    let b = idx / (params.channels * params.out_h * params.out_w);

    let iw = i32(ow) - i32(params.pad_left);
    let ih = i32(oh) - i32(params.pad_top);

    if (iw >= 0 && iw < i32(params.in_w) && ih >= 0 && ih < i32(params.in_h)) {
        let inp_idx = ((b * params.channels + c) * params.in_h + u32(ih)) * params.in_w + u32(iw);
        output[idx] = input[inp_idx];
    } else {
        output[idx] = bitcast<f32>(params.value_bits);
    }
}
