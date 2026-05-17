import { TensorHandle, TensorMeta } from "./types.js";
import { cloneHandle } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  UNARY_SHADER,
  LEAKY_RELU_SHADER,
  createStorageBuffer,
  registerTensor,
  assertUnaryDType,
} from "./utils.js";
import { DeviceManager } from "./device.js";

type UnaryEntrypoint = "relu" | "abs_op" | "sqrt_op" | "exp_op" | "log_op" | "neg"
  | "sigmoid" | "tanh_op" | "sin_op" | "cos_op" | "gelu" | "silu_op"
  | "floor_op" | "ceil_op" | "round_op" | "reciprocal_op" | "square_op"
  | "leaky_relu";

const FLOAT32_ONLY_OPS: ReadonlySet<string> = new Set([
  "relu", "sqrt", "exp", "log", "sigmoid", "tanh", "gelu", "silu"
]);

export class UnaryOps {
  constructor(
    private deviceMgr: DeviceManager,
    private tensors: Map<number, TensorMeta>,
    private nextId: { current: number },
    private allocatedBytes: { current: number }
  ) {}

  async relu(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "relu");
  }

  async abs(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "abs_op");
  }

  async sqrt(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "sqrt_op");
  }

  async exp(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "exp_op");
  }

  async log(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "log_op");
  }

  async neg(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "neg");
  }

  async sigmoid(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "sigmoid");
  }

  async tanh(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "tanh_op");
  }

  async sin(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "sin_op");
  }

  async cos(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "cos_op");
  }

  async gelu(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "gelu");
  }

  async silu(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "silu_op");
  }

  async leakyRelu(tensorId: number, alpha: number = 0.01): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const device = this.deviceMgr.device!;
    const out = createStorageBuffer(device, Math.max(4, source.length * 4));
    const paramsBuf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, new Float32Array([alpha]));
    const pipeline = getOrCreatePipeline(LEAKY_RELU_SHADER, "leaky_relu");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source.buffer, offset: 0, size: source.buffer.size } },
        { binding: 1, resource: { buffer: out, offset: 0, size: out.size } },
        { binding: 2, resource: { buffer: paramsBuf, offset: 0, size: 4 } },
      ],
    });
    const wg = calculateWorkgroups(source.length);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wg[0], wg[1], wg[2]);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await syncDevice();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, source.shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  async floor(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "floor_op");
  }

  async ceil(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "ceil_op");
  }

  async round(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "round_op");
  }

  async reciprocal(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "reciprocal_op");
  }

  async square(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "square_op");
  }

  private async unary(tensorId: number, entrypoint: UnaryEntrypoint): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    if (FLOAT32_ONLY_OPS.has(entrypoint!)) {
      const opName = entrypoint!.replace("_op", "").replace("_relu", "_relu");
      assertUnaryDType(source.dtype, opName as any);
    }
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, source.length * 4));
    const pipeline = getOrCreatePipeline(UNARY_SHADER, entrypoint!);
    dispatchCompute(pipeline, [source.buffer, out], calculateWorkgroups(source.length));
    await syncDevice();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, source.shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  private getMeta(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }
}
