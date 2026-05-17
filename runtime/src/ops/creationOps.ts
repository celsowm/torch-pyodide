import { TensorHandle, TensorMeta, SupportedDType } from "./types.js";
import {
  product,
  cloneHandle,
} from "./types.js";
import {
  assertDType,
  coerceScalarByDType,
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  FILL_SHADER,
  RANDOM_SHADER,
  createStorageBuffer,
  registerTensor,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class CreationOps {
  constructor(
    private deviceMgr: DeviceManager,
    private tensors: Map<number, TensorMeta>,
    private nextId: { current: number },
    private allocatedBytes: { current: number }
  ) {}

  async tensorFromData(data: number[], shape: number[], dtype: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    const length = product(shape);
    if (length !== data.length) {
      throw new Error(`tensorFromData expected ${length} values, got ${data.length}.`);
    }
    const typed = new Float32Array(data.map((value) => coerceScalarByDType(value, dtype as SupportedDType)));
    const buffer = createStorageBuffer(this.deviceMgr.device!, typed.byteLength);
    this.deviceMgr.device!.queue.writeBuffer(buffer, 0, typed);
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, buffer, shape, dtype, length);
    return cloneHandle(meta);
  }

  async zeros(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, 0.0);
  }

  async ones(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, 1.0);
  }

  async rand(shape: number[], dtype: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    const length = product(shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const paramsData = new Uint32Array([Math.floor(Math.random() * 0xffffffff), length]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(RANDOM_SHADER, "rand");
    dispatchCompute(pipeline, [out, paramsBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, shape, dtype, length);
    return cloneHandle(meta);
  }

  async randn(shape: number[], dtype: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    const length = product(shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const paramsData = new Uint32Array([Math.floor(Math.random() * 0xffffffff), length]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(RANDOM_SHADER, "randn");
    dispatchCompute(pipeline, [out, paramsBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, shape, dtype, length);
    return cloneHandle(meta);
  }

  async arange(start: number, end: number, step: number, dtype: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    if (step === 0) {
      throw new Error("arange step must be non-zero.");
    }
    const values: number[] = [];
    if (step > 0) {
      for (let value = start; value < end; value += step) {
        values.push(coerceScalarByDType(value, dtype as SupportedDType));
      }
    } else {
      for (let value = start; value > end; value += step) {
        values.push(coerceScalarByDType(value, dtype as SupportedDType));
      }
    }
    return this.tensorFromData(values, [values.length], dtype);
  }

  async full(shape: number[], fillValue: number, dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, fillValue);
  }

  async fullLike(tensorId: number, fillValue: number, dtype?: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const outDtype = dtype ?? source.dtype;
    return this.fill(source.shape, outDtype, fillValue);
  }

  private getMeta(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }

  private async fill(shape: number[], dtype: string, value: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    const length = product(shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new ArrayBuffer(8);
    const view = new DataView(params);
    view.setFloat32(0, coerceScalarByDType(value, dtype as SupportedDType), true);
    view.setUint32(4, length, true);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: 8,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(FILL_SHADER, "fill");
    dispatchCompute(pipeline, [out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, shape, dtype, length);
    return cloneHandle(meta);
  }
}
