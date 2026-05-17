import { TensorHandle, TensorMeta } from "./types.js";
import { product } from "./types.js";
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

const FLOAT32_ONLY_OPS = new Set(["relu", "sqrt", "exp", "log", "sigmoid", "tanh", "sin", "cos", "gelu", "silu"]);

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
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }
}
