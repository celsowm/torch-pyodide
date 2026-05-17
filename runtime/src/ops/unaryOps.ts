import { TensorHandle, TensorMeta } from "./types.js";
import { cloneHandle } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  UNARY_SHADER,
  createStorageBuffer,
  registerTensor,
  assertUnaryDType,
} from "./utils.js";
import { DeviceManager } from "./device.js";

type UnaryEntrypoint = "relu" | "abs_op" | "sqrt_op" | "exp_op" | "log_op" | "neg"
  | "sigmoid" | "tanh" | "sin" | "cos" | "gelu" | "silu" | "leaky_relu"
  | "floor" | "ceil" | "round" | "reciprocal" | "square";

const FLOAT32_ONLY_OPS: ReadonlySet<string> = new Set([
  "relu", "sqrt", "exp", "log", "sigmoid", "tanh", "gelu", "silu", "leaky_relu"
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

  async leakyRelu(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "leaky_relu");
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
