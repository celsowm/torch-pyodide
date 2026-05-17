import { TensorHandle, TensorMeta, SupportedDType } from "./ops/types.js";
import { product, cloneHandle } from "./ops/types.js";
import {
  decodeValuesByDType,
  readFromGPU,
} from "./ops/utils.js";
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
  private tensors = new Map<number, TensorMeta>();
  private nextId = { current: 1 };
  private allocatedBytes = { current: 0 };
  private creationOps: CreationOps;
  private arithmeticOps: ArithmeticOps;
  private unaryOps: UnaryOps;
  private reductionOps: ReductionOps;
  private shapeOps: ShapeOps;
  private compareOps: CompareOps;
  private maskingOps: MaskingOps;

  constructor() {
    this.creationOps = new CreationOps(this.deviceMgr, this.tensors, this.nextId, this.allocatedBytes);
    this.arithmeticOps = new ArithmeticOps(this.deviceMgr, this.tensors, this.nextId, this.allocatedBytes);
    this.unaryOps = new UnaryOps(this.deviceMgr, this.tensors, this.nextId, this.allocatedBytes);
    this.reductionOps = new ReductionOps(this.deviceMgr, this.tensors, this.nextId, this.allocatedBytes);
    this.shapeOps = new ShapeOps(
      this.deviceMgr, this.tensors, this.nextId, this.allocatedBytes,
      (id) => this.toList(id)
    );
    this.compareOps = new CompareOps(this.deviceMgr, this.tensors, this.nextId, this.allocatedBytes);
    this.maskingOps = new MaskingOps(this.deviceMgr, this.tensors, this.nextId, this.allocatedBytes);
  }

  async init(gpuProvider?: GPU | null): Promise<void> {
    await this.deviceMgr.ensureReady(gpuProvider);
  }

  // Creation ops
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

  // Arithmetic ops
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

  // Unary ops
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

  async leakyRelu(tensorId: number, alpha: number = 0.01): Promise<TensorHandle> {
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

  // Reduction ops
  async sum(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.sum(tensorId);
  }

  async mean(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.mean(tensorId);
  }

  async argmax(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.argmax(tensorId);
  }

  async argmin(tensorId: number): Promise<TensorHandle> {
    return this.reductionOps.argmin(tensorId);
  }

  // Compare ops
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

  // Masking ops
  async maskedSelect(tensorId: number, maskId: number): Promise<TensorHandle> {
    return this.maskingOps.maskedSelect(tensorId, maskId);
  }

  async maskedFill(tensorId: number, maskId: number, value: number): Promise<TensorHandle> {
    return this.maskingOps.maskedFill(tensorId, maskId, value);
  }

  // Shape ops
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

  // I/O
  async toList(tensorId: number): Promise<number[]> {
    await this.deviceMgr.ensureReady();
    const meta = this.getTensor(tensorId);
    return readFromGPU(
      this.deviceMgr.device!,
      meta.buffer,
      meta.length,
      meta.dtype as SupportedDType
    );
  }

  async destroy(tensorId: number): Promise<void> {
    const meta = this.tensors.get(tensorId);
    if (!meta) return;
    meta.buffer.destroy();
    this.allocatedBytes.current = Math.max(0, this.allocatedBytes.current - meta.bytes);
    this.tensors.delete(tensorId);
  }

  // Device info
  isAvailable(): boolean {
    return this.deviceMgr.isAvailable();
  }

  isInitialized(): boolean {
    return this.deviceMgr.isInitialized();
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

  async memoryAllocated(deviceIndex?: number): Promise<number> {
    await this.deviceMgr.ensureReady();
    return this.allocatedBytes.current;
  }

  async memoryReserved(deviceIndex?: number): Promise<number> {
    await this.deviceMgr.ensureReady();
    return this.allocatedBytes.current;
  }

  private getTensor(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }
}

export function installTorchRuntime(target: typeof globalThis = globalThis): TorchPyodideRuntime {
  const runtime = new TorchPyodideRuntime();
  (target as typeof globalThis & { __torch_pyodide_runtime__?: TorchPyodideRuntime }).__torch_pyodide_runtime__ =
    runtime;
  return runtime;
}
