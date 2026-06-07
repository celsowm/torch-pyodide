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
  makeStorageReadLayoutEntry,
  makeStorageReadWriteLayoutEntry,
  makeUniformLayoutEntry,
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
    const DEBUG_BN = (globalThis as { __DEBUG_BN__?: number }).__DEBUG_BN__ ?? 0;
    if (DEBUG_BN > 0) {
      console.log(`[BN] call: shape=[${shape}], total=${total}, channels=${channels}, eps=${eps}`);
    }

    // Pack weight, bias, running_mean, running_var into a single buffer
    // interleaved as [w, b, m, v] per channel (length = channels * 4).
    // The WGSL parser on Chromium has a low limit on the number of `read`
    // storage buffers, so we use one buffer for all affine parameters.
    const channelsData = new Float32Array(channels * 4);
    if (weightId !== null) {
      const w = await this.deviceMgr.readFromGPU(this.deviceMgr.getTensorMeta(weightId).buffer, channels, "float32");
      for (let i = 0; i < channels; i++) channelsData[i * 4 + 0] = w[i];
    } else {
      for (let i = 0; i < channels; i++) channelsData[i * 4 + 0] = 1.0;
    }
    if (biasId !== null) {
      const b = await this.deviceMgr.readFromGPU(this.deviceMgr.getTensorMeta(biasId).buffer, channels, "float32");
      for (let i = 0; i < channels; i++) channelsData[i * 4 + 1] = b[i];
    }
    if (runningMeanId !== null) {
      const m = await this.deviceMgr.readFromGPU(this.deviceMgr.getTensorMeta(runningMeanId).buffer, channels, "float32");
      for (let i = 0; i < channels; i++) channelsData[i * 4 + 2] = m[i];
    }
    if (runningVarId !== null) {
      const v = await this.deviceMgr.readFromGPU(this.deviceMgr.getTensorMeta(runningVarId).buffer, channels, "float32");
      for (let i = 0; i < channels; i++) channelsData[i * 4 + 3] = v[i];
    } else {
      for (let i = 0; i < channels; i++) channelsData[i * 4 + 3] = 1.0;
    }
    const affineBuf = createStorageBuffer(this.deviceMgr.device!, Math.max(4, channels * 4 * 4));
    this.deviceMgr.writeBuffer(affineBuf, 0, channelsData);

    if (DEBUG_BN > 0) {
      const x0 = await this._readBuffer(input.buffer, Math.min(8, total));
      console.log(`[BN] in[0..8]=`, Array.from(x0).map((v) => v.toFixed(4)));
      console.log(`[BN] affine[0..15]=`, Array.from(channelsData.slice(0, 16)).map((v) => v.toFixed(4)));
    }

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, total * 4));

    // Match WGSL Params struct layout: batch:u32, channels:u32, spatial:u32, eps:f32, _pad:u32, _pad1:u32, _pad2:u32.
    // 28 bytes total. Use a Uint32Array for the u32 fields, then overlay the eps as f32.
    const params = new ArrayBuffer(28);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = batch >>> 0;
    paramsU32[1] = channels >>> 0;
    paramsU32[2] = spatial >>> 0;
    paramsF32[3] = eps;
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: 28,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, new Uint8Array(params));

    // Build an explicit bind group layout for BN. Bypasses WGSL auto-inference
    // because the WGSL parser on Chromium is non-deterministic with multiple
    // top-level `array<f32>` storage bindings (sometimes drops bindings 0-4).
    // 4 bindings: input (read), affine (read), output (read_write), params (uniform).
    const bnLayout = this.deviceMgr.device!.createBindGroupLayout({
      entries: [
        makeStorageReadLayoutEntry(0),
        makeStorageReadLayoutEntry(1),
        makeStorageReadWriteLayoutEntry(2),
        makeUniformLayoutEntry(3),
      ],
    });
    const pipeline = await getOrCreatePipeline(BATCHNORM_SHADER, "main", [bnLayout]);
    if (DEBUG_BN > 0) {
      console.log(`[BN] using explicit bind group layout (4 entries)`);
    }
    dispatchCompute(
      pipeline,
      [input.buffer, affineBuf, out, paramBuffer],
      calculateWorkgroups(total),
    );
    await syncDevice();
    paramBuffer.destroy();
    affineBuf.destroy();
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

    const pipeline = await getOrCreatePipeline(LAYERNORM_SHADER, "main");
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

  private async _readBuffer(buf: GPUBuffer, n: number): Promise<Float32Array> {
    const out = await this.deviceMgr.readFromGPUBuffer(buf, Math.max(4, n * 4));
    return new Float32Array(out, 0, n);
  }
}
