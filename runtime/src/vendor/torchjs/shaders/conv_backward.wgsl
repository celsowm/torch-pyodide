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

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn conv2d_input_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.in_channels * params.in_h * params.in_w;
    if (idx >= total) { return; }

    let iw = idx % params.in_w;
    let ih = (idx / params.in_w) % params.in_h;
    let ic = (idx / (params.in_w * params.in_h)) % params.in_channels;
    let b = idx / (params.in_channels * params.in_h * params.in_w);

    let ic_per_group = params.in_channels / params.groups;
    let g = ic / ic_per_group;

    var sum = 0.0;
    for (var kh = 0u; kh < params.kernel_h; kh++) {
        for (var kw = 0u; kw < params.kernel_w; kw++) {
            let oh_base = i32(ih) + i32(params.pad_h) - i32(kh);
            let ow_base = i32(iw) + i32(params.pad_w) - i32(kw);
            
            if (oh_base >= 0 && ow_base >= 0 && 
                oh_base % i32(params.stride_h) == 0 && ow_base % i32(params.stride_w) == 0) {
                
                let oh = u32(oh_base) / params.stride_h;
                let ow = u32(ow_base) / params.stride_w;
                
                if (oh < params.out_h && ow < params.out_w) {
                    for (var oc = g * (params.out_channels / params.groups); oc < (g + 1u) * (params.out_channels / params.groups); oc++) {
                        let grad_out_idx = ((b * params.out_channels + oc) * params.out_h + oh) * params.out_w + ow;
                        let w_idx = ((oc * ic_per_group + (ic % ic_per_group)) * params.kernel_h + kh) * params.kernel_w + kw;
                        sum += grad_output[grad_out_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    grad_input[idx] = sum;
}

@group(0) @binding(4) var<storage, read_write> grad_weight: array<f32>;
@group(0) @binding(5) var<storage, read> input: array<f32>;

@compute @workgroup_size(256)
fn conv2d_weight_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.out_channels * params.in_channels * params.kernel_h * params.kernel_w;
    if (idx >= total) { return; }

    let kw = idx % params.kernel_w;
    let kh = (idx / params.kernel_w) % params.kernel_h;
    let ic = (idx / (params.kernel_w * params.kernel_h)) % params.in_channels;
    let oc = idx / (params.kernel_w * params.kernel_h * params.in_channels);

    let ic_per_group = params.in_channels / params.groups;
    let g = oc / (params.out_channels / params.groups);
    
    if (ic < g * ic_per_group || ic >= (g + 1u) * ic_per_group) {
        grad_weight[idx] = 0.0;
        return;
    }

    var sum = 0.0;
    for (var b = 0u; b < params.batch; b++) {
        for (var oh = 0u; oh < params.out_h; oh++) {
            for (var ow = 0u; ow < params.out_w; ow++) {
                let ih = i32(oh * params.stride_h + kh) - i32(params.pad_h);
                let iw = i32(ow * params.stride_w + kw) - i32(params.pad_w);
                
                if (ih >= 0 && i32(ih) < i32(params.in_h) && iw >= 0 && i32(iw) < i32(params.in_w)) {
                    let inp_idx = ((b * params.in_channels + ic) * params.in_h + u32(ih)) * params.in_w + u32(iw);
                    let grad_out_idx = ((b * params.out_channels + oc) * params.out_h + oh) * params.out_w + ow;
                    sum += grad_output[grad_out_idx] * input[inp_idx];
                }
            }
        }
    }
    grad_weight[idx] = sum;
}

@compute @workgroup_size(256)
fn conv2d_bias_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.out_channels;
    if (idx >= total) { return; }

    var sum = 0.0;
    for (var b = 0u; b < params.batch; b++) {
        for (var oh = 0u; oh < params.out_h; oh++) {
            for (var ow = 0u; ow < params.out_w; ow++) {
                let grad_out_idx = ((b * params.out_channels + idx) * params.out_h + oh) * params.out_w + ow;
                sum += grad_output[grad_out_idx];
            }
        }
    }
    grad_weight[idx] = sum;
}
