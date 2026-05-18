import { TensorHandle, TensorMeta, product } from "./types.js";
import {
  assertUnaryDType,
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  UNARY_SHADER,
  LEAKY_RELU_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";

const FLOAT32_ONLY_OPS = new Set([
  "relu", "sqrt", "exp", "log", "sigmoid", "tanh", "sin", "cos", "gelu", "silu",
  "sinh", "cosh", "tan", "asin", "acos", "atan",
  "asinh", "acosh", "atanh",
  "exp2", "log2", "log10", "log1p", "expm1",
  "softplus", "mish", "hardsigmoid", "hardswish", "softsign", "tanhshrink",
  "trunc", "frac", "rsqrt",
  "erf", "erfc", "lgamma", "digamma", "i0",
  "deg2rad", "rad2deg",
]);

const ENTRYPOINT_MAP: Record<string, string> = {
  abs: "abs_op",
  sqrt: "sqrt_op",
  exp: "exp_op",
  log: "log_op",
  tanh: "tanh_op",
  sin: "sin_op",
  cos: "cos_op",
  silu: "silu_op",
  gelu: "gelu",
  neg: "neg",
  floor: "floor_op",
  ceil: "ceil_op",
  round: "round_op",
  reciprocal: "reciprocal_op",
  square: "square_op",
  // Trig
  tan: "tan_op",
  asin: "asin_op",
  acos: "acos_op",
  atan: "atan_op",
  sinh: "sinh_op",
  cosh: "cosh_op",
  asinh: "asinh_op",
  acosh: "acosh_op",
  atanh: "atanh_op",
  // Exp/Log
  exp2: "exp2_op",
  log2: "log2_op",
  log10: "log10",
  log1p: "log1p",
  expm1: "expm1_op",
  // Rounding
  trunc: "trunc_op",
  frac: "frac_op",
  // Activations
  softplus: "softplus_op",
  mish: "mish_op",
  hardsigmoid: "hardsigmoid_op",
  hardswish: "hardswish_op",
  softsign: "softsign_op",
  tanhshrink: "tanhshrink_op",
  // Arithmetic
  rsqrt: "rsqrt_op",
  sign: "sign_op",
  sgn: "sgn_op",
  // Boolean
  isnan: "isnan_op",
  isinf: "isinf_op",
  isfinite: "isfinite_op",
  isposinf: "isposinf_op",
  isneginf: "isneginf_op",
  logical_not: "logical_not_op",
  // Special
  erf: "erf_op",
  erfc: "erfc_op",
  lgamma: "lgamma_op",
  digamma: "digamma_op",
  i0: "i0_op",
  // Conversion
  deg2rad: "deg2rad_op",
  rad2deg: "rad2deg_op",
};

export class UnaryOps {
  constructor(private deviceMgr: DeviceManager) {}

  async relu(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "relu");
  }

  async abs(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "abs");
  }

  async sqrt(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "sqrt");
  }

  async exp(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "exp");
  }

  async log(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "log");
  }

  async neg(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "neg");
  }

  async sigmoid(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "sigmoid");
  }

  async tanh(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "tanh");
  }

  async sin(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "sin");
  }

  async cos(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "cos");
  }

  async gelu(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "gelu");
  }

  async silu(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "silu");
  }

  async leakyRelu(tensorId: number, alpha = 0.01): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new Float32Array([alpha, 0, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(LEAKY_RELU_SHADER, "leaky_relu");
    const bindGroup = this.deviceMgr.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: meta.buffer } },
        { binding: 1, resource: { buffer: out } },
        { binding: 2, resource: { buffer: paramBuffer } },
      ],
    });
    const encoder = this.deviceMgr.device!.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / 256));
    pass.end();
    this.deviceMgr.device!.queue.submit([encoder.finish()]);
    await this.deviceMgr.syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async floor(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "floor");
  }

  async ceil(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "ceil");
  }

  async round(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "round");
  }

  async reciprocal(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "reciprocal");
  }

  async square(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "square");
  }

  // Trig
  async tan(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "tan");
  }

  async asin(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "asin");
  }

  async acos(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "acos");
  }

  async atan(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "atan");
  }

  async sinh(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "sinh");
  }

  async cosh(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "cosh");
  }

  async asinh(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "asinh");
  }

  async acosh(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "acosh");
  }

  async atanh(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "atanh");
  }

  // Exp/Log
  async exp2(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "exp2");
  }

  async log2(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "log2");
  }

  async log10(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "log10");
  }

  async log1p(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "log1p");
  }

  async expm1(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "expm1");
  }

  // Rounding
  async trunc(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "trunc");
  }

  async frac(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "frac");
  }

  // Activations
  async softplus(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "softplus");
  }

  async mish(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "mish");
  }

  async hardsigmoid(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "hardsigmoid");
  }

  async hardswish(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "hardswish");
  }

  async softsign(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "softsign");
  }

  async tanhshrink(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "tanhshrink");
  }

  // Arithmetic
  async rsqrt(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "rsqrt");
  }

  async sign(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "sign");
  }

  async sgn(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "sgn");
  }

  // Boolean
  async isnan(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "isnan");
  }

  async isinf(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "isinf");
  }

  async isfinite(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "isfinite");
  }

  async isposinf(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "isposinf");
  }

  async isneginf(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "isneginf");
  }

  async logicalNot(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "logical_not");
  }

  // Special
  async erf(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "erf");
  }

  async erfc(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "erfc");
  }

  async lgamma(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "lgamma");
  }

  async digamma(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "digamma");
  }

  async i0(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "i0");
  }

  // Conversion
  async deg2rad(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "deg2rad");
  }

  async rad2deg(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "rad2deg");
  }

  private BOOL_OPS = new Set(["isnan", "isinf", "isfinite", "isposinf", "isneginf", "logical_not"]);

  private async unary(tensorId: number, entrypoint: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    if (FLOAT32_ONLY_OPS.has(entrypoint)) {
      assertUnaryDType(meta.dtype, entrypoint);
    }
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const pipeline = getOrCreatePipeline(UNARY_SHADER, ENTRYPOINT_MAP[entrypoint] || entrypoint);
    dispatchCompute(pipeline, [meta.buffer, out], calculateWorkgroups(length));
    await syncDevice();
    const outDtype = this.BOOL_OPS.has(entrypoint) ? "bool" : meta.dtype;
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, outDtype, length);
  }

  async fill(tensorId: number, value: number): Promise<TensorHandle> {
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const dtype = meta.dtype as "float32" | "int32" | "bool";
    const data = new Float32Array(length).fill(value);
    const buffer = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    this.deviceMgr.device!.queue.writeBuffer(buffer, 0, data);
    return this.deviceMgr.registerTensorAsHandle(buffer, meta.shape, dtype, length);
  }
}
