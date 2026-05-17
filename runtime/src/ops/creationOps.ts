import { TensorHandle, TensorMeta, SupportedDType } from "./types.js";
import { product } from "./types.js";
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
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class CreationOps {
  constructor(private deviceMgr: DeviceManager) {}

  async tensorFromData(data: number[], shape: number[], dtype: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    const length = product(shape);
    if (length !== data.length) throw new Error(`tensorFromData expected ${length} values, got ${data.length}.`);
    const typed = new Float32Array(data.map((v) => coerceScalarByDType(v, dtype as SupportedDType)));
    const buffer = createStorageBuffer(this.deviceMgr.device!, typed.byteLength);
    this.deviceMgr.writeBuffer(buffer, 0, typed);
    return this.deviceMgr.registerTensorAsHandle(buffer, shape, dtype, length);
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
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(RANDOM_SHADER, "rand");
    dispatchCompute(pipeline, [out, paramsBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramsBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, shape, dtype, length);
  }

  async randn(shape: number[], dtype: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    const length = product(shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const paramsData = new Uint32Array([Math.floor(Math.random() * 0xffffffff), length]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(RANDOM_SHADER, "randn");
    dispatchCompute(pipeline, [out, paramsBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramsBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, shape, dtype, length);
  }

  async arange(start: number, end: number, step: number, dtype: string): Promise<TensorHandle> {
    assertDType(dtype);
    if (step === 0) throw new Error("arange step must be non-zero.");
    const values: number[] = [];
    if (step > 0) {
      for (let v = start; v < end; v += step) values.push(coerceScalarByDType(v, dtype as SupportedDType));
    } else {
      for (let v = start; v > end; v += step) values.push(coerceScalarByDType(v, dtype as SupportedDType));
    }
    return this.tensorFromData(values, [values.length], dtype);
  }

  async full(shape: number[], fillValue: number, dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, fillValue);
  }

  async fullLike(tensorId: number, fillValue: number, dtype?: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.deviceMgr.getTensorMeta(tensorId);
    return this.fill(source.shape, dtype ?? source.dtype, fillValue);
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
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(FILL_SHADER, "fill");
    dispatchCompute(pipeline, [out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, shape, dtype, length);
  }
}
