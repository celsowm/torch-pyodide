import { TensorHandle, SupportedDType, product } from "./types.js";
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
  private randomSeed: number = 42;

  constructor(private deviceMgr: DeviceManager) {}

  async setSeed(seed: number): Promise<void> {
    this.randomSeed = seed;
  }

  async tensorFromData(data: number[], shape: number[], dtype: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    const length = product(shape);
    if (length !== data.length) throw new Error(`tensorFromData expected ${length} values, got ${data.length}.`);
    const coerced = data.map((v) => coerceScalarByDType(v, dtype as SupportedDType));
    // WebGPU shaders use f32. For float16/bfloat16, we store as f32 but track dtype in metadata.
    // This matches PyTorch CPU behavior: dtype is tracked, but storage is f32 for compute ops.
    const typed = new Float32Array(coerced);
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
    return this.sampleOp(shape, dtype, "rand", 0.0, 1.0);
  }

  async randn(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.sampleOp(shape, dtype, "randn", 0.0, 1.0);
  }

  async normal(shape: number[], dtype: string, mean: number, std: number): Promise<TensorHandle> {
    return this.sampleOp(shape, dtype, "normal_sample", mean, std);
  }

  async bernoulli(shape: number[], dtype: string, p: number): Promise<TensorHandle> {
    return this.sampleOp(shape, dtype, "bernoulli_sample", p, 0.0);
  }

  async exponential(shape: number[], dtype: string, rate: number): Promise<TensorHandle> {
    return this.sampleOp(shape, dtype, "exponential_sample", rate, 0.0);
  }

  async logNormal(shape: number[], dtype: string, mean: number, std: number): Promise<TensorHandle> {
    return this.sampleOp(shape, dtype, "log_normal_sample", mean, std);
  }

  private async sampleOp(
    shape: number[],
    dtype: string,
    entrypoint: string,
    param0: number,
    param1: number,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    const length = product(shape);
    const seed = this.randomSeed++;
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    // RNGParams = { seed: u32, length: u32, param0: f32, param1: f32 } = 16 bytes
    const params = new ArrayBuffer(16);
    const u32 = new Uint32Array(params);
    const f32 = new Float32Array(params);
    u32[0] = seed >>> 0;
    u32[1] = length;
    f32[2] = param0;
    f32[3] = param1;
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramsBuffer, 0, params);
    const pipeline = getOrCreatePipeline(RANDOM_SHADER, entrypoint);
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

  async zerosLike(tensorId: number, dtype?: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.deviceMgr.getTensorMeta(tensorId);
    return this.fill(source.shape, dtype ?? source.dtype, 0.0);
  }

  async onesLike(tensorId: number, dtype?: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.deviceMgr.getTensorMeta(tensorId);
    return this.fill(source.shape, dtype ?? source.dtype, 1.0);
  }

  async emptyLike(tensorId: number, dtype?: string): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.deviceMgr.getTensorMeta(tensorId);
    return this.fill(source.shape, dtype ?? source.dtype, 0.0);
  }

  async empty(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, 0.0);
  }

  private async fill(shape: number[], dtype: string, value: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    assertDType(dtype);
    const length = product(shape);
    const byteSize = length * Float32Array.BYTES_PER_ELEMENT;
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, byteSize));
    const params = new ArrayBuffer(16);
    const view = new DataView(params);
    view.setFloat32(0, coerceScalarByDType(value, dtype as SupportedDType), true);
    view.setUint32(4, length, true);
    view.setUint32(8, 0, true);
    view.setUint32(12, 0, true);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: 16,
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
