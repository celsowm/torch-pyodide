struct Params {
    batch: u32,
    in_channels: u32,
    out_channels: u32,
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
    groups: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn conv2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.out_channels * params.out_h * params.out_w;
    if (idx >= total) { return; }

    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let oc = (idx / (params.out_w * params.out_h)) % params.out_channels;
    let b = idx / (params.out_channels * params.out_h * params.out_w);

    let ic_per_group = params.in_channels / params.groups;
    let g = oc / (params.out_channels / params.groups);

    var sum = 0.0;
    for (var kc = 0u; kc < ic_per_group; kc++) {
        let ic = g * ic_per_group + kc;
        for (var kh = 0u; kh < params.kernel_h; kh++) {
            let ih = i32(oh * params.stride_h + kh) - i32(params.pad_h);
            for (var kw = 0u; kw < params.kernel_w; kw++) {
                let iw = i32(ow * params.stride_w + kw) - i32(params.pad_w);
                if (ih >= 0 && i32(ih) < i32(params.in_h) && iw >= 0 && i32(iw) < i32(params.in_w)) {
                    let inp_idx = ((b * params.in_channels + ic) * params.in_h + u32(ih)) * params.in_w + u32(iw);
                    let w_idx = ((oc * ic_per_group + kc) * params.kernel_h + kh) * params.kernel_w + kw;
                    sum += input[inp_idx] * weight[w_idx];
                }
            }
        }
    }

    let has_bias = params.out_channels > 0u;
    if (has_bias) {
        sum += bias[oc];
    }
    output[idx] = sum;
}

@compute @workgroup_size(256)
fn conv1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.out_channels * params.out_w;
    if (idx >= total) { return; }

    let ow = idx % params.out_w;
    let oc = (idx / params.out_w) % params.out_channels;
    let b = idx / (params.out_channels * params.out_w);

    let ic_per_group = params.in_channels / params.groups;
    let g = oc / (params.out_channels / params.groups);

    var sum = 0.0;
    for (var kc = 0u; kc < ic_per_group; kc++) {
        let ic = g * ic_per_group + kc;
        for (var kw = 0u; kw < params.kernel_w; kw++) {
            let iw = i32(ow * params.stride_w + kw) - i32(params.pad_w);
            if (iw >= 0 && i32(iw) < i32(params.in_w)) {
                let inp_idx = (b * params.in_channels + ic) * params.in_w + u32(iw);
                let w_idx = (oc * ic_per_group + kc) * params.kernel_w + kw;
                sum += input[inp_idx] * weight[w_idx];
            }
        }
    }

    let has_bias = params.out_channels > 0u;
    if (has_bias) {
        sum += bias[oc];
    }
    output[idx] = sum;
}
