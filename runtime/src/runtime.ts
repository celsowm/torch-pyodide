import { TensorHandle, SupportedDType } from "./ops/types.js";
import { setDeviceManager } from "./ops/utils.js";
import { DeviceManager } from "./ops/device.js";
import { CreationOps } from "./ops/creationOps.js";
import { ArithmeticOps } from "./ops/arithmeticOps.js";
import { UnaryOps } from "./ops/unaryOps.js";
import { ReductionOps } from "./ops/reductionOps.js";
import { ShapeOps } from "./ops/shapeOps.js";
import { CompareOps } from "./ops/compareOps.js";
import { MaskingOps } from "./ops/maskingOps.js";

export class TorchPyodideRuntime {
  private deviceMgr = new DeviceManager();
  private creationOps: CreationOps;
  private arithmeticOps: ArithmeticOps;
  private unaryOps: UnaryOps;
  private reductionOps: ReductionOps;
  private shapeOps: ShapeOps;
  private compareOps: CompareOps;
  private maskingOps: MaskingOps;

  constructor() {
    setDeviceManager(this.deviceMgr);
    const dm = this.deviceMgr;
    this.creationOps = new CreationOps(dm);
    this.arithmeticOps = new ArithmeticOps(dm);
    this.unaryOps = new UnaryOps(dm);
    this.reductionOps = new ReductionOps(dm);
    this.shapeOps = new ShapeOps(dm);
    this.compareOps = new CompareOps(dm);
    this.maskingOps = new MaskingOps(dm);
  }

  async init(gpuProvider?: GPU | null): Promise<void> {
    await this.deviceMgr.ensureReady(gpuProvider);
  }

  async tensorFromData(data: number[], shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.tensorFromData(data, shape, dtype);
  }

  async zeros(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.zeros(shape, dtype);
  }

  async ones(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.ones(shape, dtype);
  }

  async rand(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.rand(shape, dtype);
  }

  async randn(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.creationOps.randn(shape, dtype);
  }

  async arange(start: number, end: number, step: number, dtype: string): Promise<TensorHandle> {
    return this.creationOps.arange(start, end, step, dtype);
  }

  async full(shape: number[], fillValue: number, dtype: string): Promise<TensorHandle> {
    return this.creationOps.full(shape, fillValue, dtype);
  }

  async fullLike(tensorId: number, fillValue: number, dtype?: string): Promise<TensorHandle> {
    return this.creationOps.fullLike(tensorId, fillValue, dtype);
  }

  async zerosLike(tensorId: number, dtype?: string): Promise<TensorHandle> {
    return this.creationOps.zerosLike(tensorId, dtype);
  }

  async onesLike(tensorId: number, dtype?: string): Promise<TensorHandle> {
    return this.creationOps.onesLike(tensorId, dtype);
  }

  async emptyLike(tensorId: number, dtype?: string): Promise<TensorHandle> {
    return this.creationOps.emptyLike(tensorId, dtype);
  }

  async add(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.add(aId, bId);
  }

  async mul(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.mul(aId, bId);
  }

  async sub(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.sub(aId, bId);
  }

  async div(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.div(aId, bId);
  }

  async where(conditionId: number, xId: number, yId: number): Promise<TensorHandle> {
    return this.arithmeticOps.where(conditionId, xId, yId);
  }

  async clamp(tensorId: number, minVal: number, maxVal: number): Promise<TensorHandle> {
    return this.arithmeticOps.clamp(tensorId, minVal, maxVal);
  }

  async matmul(aId: number, bId: number): Promise<TensorHandle> {
    return this.arithmeticOps.matmul(aId, bId);
  }

  async relu(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.relu(tensorId);
  }

  async abs(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.abs(tensorId);
  }

  async sqrt(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.sqrt(tensorId);
  }

  async exp(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.exp(tensorId);
  }

  async log(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.log(tensorId);
  }

  async neg(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.neg(tensorId);
  }

  async sigmoid(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.sigmoid(tensorId);
  }

  async tanh(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.tanh(tensorId);
  }

  async sin(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.sin(tensorId);
  }

  async cos(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.cos(tensorId);
  }

  async gelu(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.gelu(tensorId);
  }

  async silu(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.silu(tensorId);
  }

  async leakyRelu(tensorId: number, alpha = 0.01): Promise<TensorHandle> {
    return this.unaryOps.leakyRelu(tensorId, alpha);
  }

  async floor(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.floor(tensorId);
  }

  async ceil(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.ceil(tensorId);
  }

  async round(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.round(tensorId);
  }

  async reciprocal(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.reciprocal(tensorId);
  }

  async square(tensorId: number): Promise<TensorHandle> {
    return this.unaryOps.square(tensorId);
  }

  async sum(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.sum(tensorId);
  }

  async mean(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.mean(tensorId);
  }

  async sumDim(tensorId: number, dim: number, keepdim: boolean): Promise<TensorHandle> {
    return this.reductionOps.sumDim(tensorId, dim, keepdim);
  }

  async meanDim(tensorId: number, dim: number, keepdim: boolean): Promise<TensorHandle> {
    return this.reductionOps.meanDim(tensorId, dim, keepdim);
  }

  async prod(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.prod(tensorId);
  }

  async min(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.min(tensorId);
  }

  async max(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.max(tensorId);
  }

  async argmax(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.argmax(tensorId);
  }

  async argmin(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.argmin(tensorId);
  }

  async eq(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.eq(aId, bId);
  }

  async ne(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.ne(aId, bId);
  }

  async lt(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.lt(aId, bId);
  }

  async le(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.le(aId, bId);
  }

  async gt(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.gt(aId, bId);
  }

  async ge(aId: number, bId: number): Promise<TensorHandle> {
    return this.compareOps.ge(aId, bId);
  }

  async maskedSelect(tensorId: number, maskId: number): Promise<TensorHandle> {
    return this.maskingOps.maskedSelect(tensorId, maskId);
  }

  async maskedFill(tensorId: number, maskId: number, value: number): Promise<TensorHandle> {
    return this.maskingOps.maskedFill(tensorId, maskId, value);
  }

  async reshape(tensorId: number, shape: number[]): Promise<TensorHandle> {
    return this.shapeOps.reshape(tensorId, shape);
  }

  async flatten(tensorId: number, startDim = 0, endDim = -1): Promise<TensorHandle> {
    return this.shapeOps.flatten(tensorId, startDim, endDim);
  }

  async squeeze(tensorId: number, dim?: number): Promise<TensorHandle> {
    return this.shapeOps.squeeze(tensorId, dim);
  }

  async unsqueeze(tensorId: number, dim: number): Promise<TensorHandle> {
    return this.shapeOps.unsqueeze(tensorId, dim);
  }

  async transpose2d(tensorId: number): Promise<TensorHandle> {
    return this.shapeOps.transpose2d(tensorId);
  }

  async transpose(tensorId: number, dim0: number, dim1: number): Promise<TensorHandle> {
    return this.shapeOps.transpose(tensorId, dim0, dim1);
  }

  async permute(tensorId: number, dims: number[]): Promise<TensorHandle> {
    return this.shapeOps.permute(tensorId, dims);
  }

  async select(tensorId: number, dim: number, index: number): Promise<TensorHandle> {
    return this.shapeOps.select(tensorId, dim, index);
  }

  async slice(tensorId: number, dim: number, start?: number, end?: number, step = 1): Promise<TensorHandle> {
    return this.shapeOps.slice(tensorId, dim, start, end, step);
  }

  async cat(tensorIds: number[], dim: number): Promise<TensorHandle> {
    return this.shapeOps.cat(tensorIds, dim);
  }

  async stack(tensorIds: number[], dim: number): Promise<TensorHandle> {
    return this.shapeOps.stack(tensorIds, dim);
  }

  async expand(tensorId: number, shape: number[]): Promise<TensorHandle> {
    return this.shapeOps.expand(tensorId, shape);
  }

  async indexSelect(tensorId: number, dim: number, indicesId: number): Promise<TensorHandle> {
    return this.shapeOps.indexSelect(tensorId, dim, indicesId);
  }

  /** Execute a batch of operations — all compute work is accumulated and submitted once. */
  async runBatch<T>(fn: () => Promise<T>): Promise<T> {
    this.deviceMgr.beginFrame();
    try {
      const result = await fn();
      await this.deviceMgr.endFrame();
      return result;
    } catch (err) {
      this.deviceMgr.cancelFrame();
      throw err;
    }
  }

  async toList(tensorId: number): Promise<number[]> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    return this.deviceMgr.readFromGPU(meta.buffer, meta.length, meta.dtype as SupportedDType);
  }

  async destroy(tensorId: number): Promise<void> {
    this.deviceMgr.destroyTensor(tensorId);
  }

  isAvailable(): boolean {
    return this.deviceMgr.isAvailable();
  }

  isInitialized(): boolean {
    return this.deviceMgr.initialized;
  }

  deviceCount(): number {
    return this.deviceMgr.deviceCount();
  }

  async currentDevice(): Promise<number> {
    return this.deviceMgr.currentDevice();
  }

  async getDeviceName(deviceIndex?: number): Promise<string> {
    return this.deviceMgr.getDeviceName(deviceIndex);
  }

  async getDeviceProperties(deviceIndex?: number): Promise<Record<string, unknown>> {
    return this.deviceMgr.getDeviceProperties(deviceIndex);
  }

  async memoryAllocated(_deviceIndex?: number): Promise<number> {
    return this.deviceMgr.memoryAllocated();
  }

  async memoryReserved(_deviceIndex?: number): Promise<number> {
    return this.deviceMgr.memoryReserved();
  }
}

export function installTorchRuntime(target: typeof globalThis = globalThis): TorchPyodideRuntime {
  const runtime = new TorchPyodideRuntime();
  (target as typeof globalThis & { __torch_pyodide_runtime__?: TorchPyodideRuntime }).__torch_pyodide_runtime__ =
    runtime;
  return runtime;
}
