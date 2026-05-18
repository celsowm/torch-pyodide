import { TensorHandle, SupportedDType } from "./types.js";
import { product } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  LAYERNORM_SHADER,
  BATCHNORM_SHADER,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class NormalizationOps {
  constructor(private deviceMgr: DeviceManager) {}

  async batchNorm(
    inputId: number,
    weightId: number | null,
    biasId: number | null,
    runningMeanId: number | null,
    runningVarId: number | null,
    eps: number,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const input = this.deviceMgr.getTensorMeta(inputId);
    const shape = input.shape;
    // Support (N, C), (N, C, L), (N, C, H, W)
    let batch: number, channels: number, spatial: number;
    if (shape.length === 2) {
      batch = shape[0];
      channels = shape[1];
      spatial = 1;
    } else if (shape.length >= 3) {
      batch = shape[0];
      channels = shape[1];
      spatial = product(shape.slice(2));
    } else {
      throw new Error("batch_norm needs at least 2D input");
    }
    const total = batch * channels * spatial;

    // Create dummy ones/zeros if weight/bias are null
    const weightBuf = weightId !== null
      ? this.deviceMgr.getTensorMeta(weightId).buffer
      : this._makeOnesBuf(channels);
    const biasBuf = biasId !== null
      ? this.deviceMgr.getTensorMeta(biasId).buffer
      : this._makeZerosBuf(channels);

    const runningMeanShape = runningMeanId !== null
      ? this.deviceMgr.getTensorMeta(runningMeanId).buffer
      : this._makeZerosBuf(channels);
    const runningVarShape = runningVarId !== null
      ? this.deviceMgr.getTensorMeta(runningVarId).buffer
      : this._makeOnesBuf(channels);

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));

    const params = new Float32Array([batch, channels, spatial, eps, 0, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, new Uint8Array(params.buffer));

    const pipeline = getOrCreatePipeline(BATCHNORM_SHADER, "main");
    dispatchCompute(
      pipeline,
      [input.buffer, weightBuf, biasBuf, runningMeanShape, runningVarShape, out, paramBuffer],
      calculateWorkgroups(total),
    );
    await syncDevice();
    paramBuffer.destroy();
    if (weightId === null) weightBuf.destroy();
    if (biasId === null) biasBuf.destroy();
    if (runningMeanId === null) runningMeanShape.destroy();
    if (runningVarId === null) runningVarShape.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, shape, input.dtype as SupportedDType, total);
  }

  async layerNorm(
    inputId: number,
    normalizedShape: number[],
    gammaId: number | null,
    betaId: number | null,
    eps: number,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const input = this.deviceMgr.getTensorMeta(inputId);
    const shape = input.shape;
    const normalizedSize = product(normalizedShape);
    const batchSize = product(shape.slice(0, shape.length - normalizedShape.length));
    const total = product(shape);

    const gammaBuf = gammaId !== null
      ? this.deviceMgr.getTensorMeta(gammaId).buffer
      : this._makeOnesBuf(normalizedSize);
    const betaBuf = betaId !== null
      ? this.deviceMgr.getTensorMeta(betaId).buffer
      : this._makeZerosBuf(normalizedSize);

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));

    const params = new Uint32Array([batchSize, normalizedSize, 0, 0]);
    const epsView = new Float32Array(params.buffer);
    epsView[2] = eps;
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, new Uint8Array(params.buffer));

    const pipeline = getOrCreatePipeline(LAYERNORM_SHADER, "main");
    dispatchCompute(
      pipeline,
      [input.buffer, gammaBuf, betaBuf, out, paramBuffer],
      calculateWorkgroups(batchSize),
    );
    await syncDevice();
    paramBuffer.destroy();
    if (gammaId === null) gammaBuf.destroy();
    if (betaId === null) betaBuf.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, shape, input.dtype as SupportedDType, total);
  }

  private _makeOnesBuf(n: number): GPUBuffer {
    const buf = createStorageBuffer(this.deviceMgr.device!, Math.max(4, n * 4));
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = 1;
    this.deviceMgr.writeBuffer(buf, 0, data);
    return buf;
  }

  private _makeZerosBuf(n: number): GPUBuffer {
    const buf = createStorageBuffer(this.deviceMgr.device!, Math.max(4, n * 4));
    this.deviceMgr.writeBuffer(buf, 0, new Float32Array(n));
    return buf;
  }
}
