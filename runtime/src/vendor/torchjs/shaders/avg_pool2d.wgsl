struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    count_include_pad: u32,
    _pad: u32,
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

    var sum = 0.0;
    var count = 0u;
    for (var kh = 0u; kh < params.kernel_h; kh++) {
        let ih = i32(oh * params.stride_h + kh) - i32(params.pad_h);
        for (var kw = 0u; kw < params.kernel_w; kw++) {
            let iw = i32(ow * params.stride_w + kw) - i32(params.pad_w);
            if (ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w) {
                let inp_idx = ((b * params.channels + c) * params.in_h + u32(ih)) * params.in_w + u32(iw);
                sum += input[inp_idx];
                count++;
            } else if (params.count_include_pad > 0u) {
                count++;
            }
        }
    }

    if (count > 0u) {
        output[idx] = sum / f32(count);
    } else {
        output[idx] = 0.0;
    }
}
