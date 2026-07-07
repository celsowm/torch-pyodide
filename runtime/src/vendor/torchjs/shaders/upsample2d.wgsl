struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    mode: u32,        // 0 = nearest, 1 = bilinear
    align_corners: u32,
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

    if (params.mode == 0u) {
        // Nearest neighbor, matching PyTorch's coordinate mapping.
        var ih: u32;
        var iw: u32;
        if (params.align_corners == 1u && params.out_h > 1u) {
            let hc = f32(oh) * f32(params.in_h - 1u) / f32(params.out_h - 1u);
            ih = u32(clamp(i32(floor(hc + 0.5)), 0, i32(params.in_h - 1u)));
        } else {
            let hc = (f32(oh) + 0.5) * f32(params.in_h) / f32(params.out_h);
            ih = u32(clamp(floor(hc), 0.0, f32(params.in_h - 1u)));
        }
        if (params.align_corners == 1u && params.out_w > 1u) {
            let wc = f32(ow) * f32(params.in_w - 1u) / f32(params.out_w - 1u);
            iw = u32(clamp(i32(floor(wc + 0.5)), 0, i32(params.in_w - 1u)));
        } else {
            let wc = (f32(ow) + 0.5) * f32(params.in_w) / f32(params.out_w);
            iw = u32(clamp(floor(wc), 0.0, f32(params.in_w - 1u)));
        }
        let inp_idx = ((b * params.channels + c) * params.in_h + ih) * params.in_w + iw;
        output[idx] = input[inp_idx];
        return;
    }

    // Bilinear interpolation, matching PyTorch's coordinate mapping.
    var hc: f32;
    var wc: f32;
    if (params.align_corners == 1u) {
        hc = select(0.0, f32(oh) * f32(params.in_h - 1u) / f32(params.out_h - 1u), params.out_h > 1u);
        wc = select(0.0, f32(ow) * f32(params.in_w - 1u) / f32(params.out_w - 1u), params.out_w > 1u);
    } else {
        hc = (f32(oh) + 0.5) * f32(params.in_h) / f32(params.out_h) - 0.5;
        wc = (f32(ow) + 0.5) * f32(params.in_w) / f32(params.out_w) - 0.5;
    }

    let ih0 = i32(floor(hc));
    let iw0 = i32(floor(wc));
    let di = hc - f32(ih0);
    let dj = wc - f32(iw0);

    let ih0c = u32(clamp(ih0, 0, i32(params.in_h - 1u)));
    let ih1c = u32(clamp(ih0 + 1, 0, i32(params.in_h - 1u)));
    let iw0c = u32(clamp(iw0, 0, i32(params.in_w - 1u)));
    let iw1c = u32(clamp(iw0 + 1, 0, i32(params.in_w - 1u)));

    let base = (b * params.channels + c) * params.in_h;
    let v00 = input[(base + ih0c) * params.in_w + iw0c];
    let v01 = input[(base + ih0c) * params.in_w + iw1c];
    let v10 = input[(base + ih1c) * params.in_w + iw0c];
    let v11 = input[(base + ih1c) * params.in_w + iw1c];

    let w00 = (1.0 - di) * (1.0 - dj);
    let w01 = (1.0 - di) * dj;
    let w10 = di * (1.0 - dj);
    let w11 = di * dj;

    output[idx] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
}
