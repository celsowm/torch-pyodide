struct Params {
  n: u32,
  p: u32,
  q: u32,
  c: f32,
  s: f32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Aout: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Vout: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// One Jacobi rotation: A' = G^T A G, V' = V G, where
//   G = I with G[p,p]=G[q,q]=c, G[p,q]=s, G[q,p]=-s.
// Thread per row; the shader writes every entry of Aout and Vout.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params.n;
  let r = gid.x;
  if (r >= n) {
    return;
  }
  let p = params.p;
  let q = params.q;
  let c = params.c;
  let s = params.s;

  // --- V (eigenvectors): right multiplication by G (column rotation) ---
  // A' = G^T A G composes V_total = G1 G2 ...; reconstruction V D V^T holds.
  let vp = V[r * n + p];
  let vq = V[r * n + q];
  Vout[r * n + p] = c * vp - s * vq;
  Vout[r * n + q] = s * vp + c * vq;
  for (var cc: u32 = 0u; cc < n; cc = cc + 1u) {
    if (cc != p && cc != q) {
      Vout[r * n + cc] = V[r * n + cc];
    }
  }

  // --- A: two-sided rotation A' = G^T A G ---
  if (r == p || r == q) {
    for (var cc: u32 = 0u; cc < n; cc = cc + 1u) {
      var Bpc: f32;
      var Bqc: f32;
      if (cc == p) {
        Bpc = c * A[p * n + p] - s * A[p * n + q];
        Bqc = c * A[q * n + p] - s * A[q * n + q];
      } else if (cc == q) {
        Bpc = s * A[p * n + p] + c * A[p * n + q];
        Bqc = s * A[q * n + p] + c * A[q * n + q];
      } else {
        Bpc = A[p * n + cc];
        Bqc = A[q * n + cc];
      }
      if (r == p) {
        Aout[p * n + cc] = c * Bpc - s * Bqc;
      } else {
        Aout[q * n + cc] = s * Bpc + c * Bqc;
      }
    }
  } else {
    let arp = A[r * n + p];
    let arq = A[r * n + q];
    Aout[r * n + p] = c * arp - s * arq;
    Aout[r * n + q] = s * arp + c * arq;
    for (var cc: u32 = 0u; cc < n; cc = cc + 1u) {
      if (cc != p && cc != q) {
        Aout[r * n + cc] = A[r * n + cc];
      }
    }
  }
}
