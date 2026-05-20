import { TensorHandle, TensorMeta } from "./types.js";
import { product } from "./types.js";
import {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  TRANSPOSE_SHADER,
  CAT_SHADER,
  STACK_SHADER,
  PERMUTE_ND_SHADER,
  SELECT_SHADER,
  SLICE_SHADER,
  SLICE_BACKWARD_SHADER,
  EXPAND_SHADER,
  EXPAND_BROADCAST_SHADER,
  INDEX_SELECT_SHADER,
  TRIL_SHADER,
  TRIU_SHADER,
  FLIP_SHADER,
  REPEAT_SHADER,
  normalizeDim,
  computeStrides,
  normalizeSliceStart,
  normalizeSliceEnd,
  padShapeTo4,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";

const UNIFORM_ALIGNMENT = 16;

function createUniformParamBuffer(
  deviceMgr: DeviceManager,
  params: ArrayBufferView,
  minSize: number,
): GPUBuffer {
  const size = Math.max(minSize, Math.ceil(params.byteLength / UNIFORM_ALIGNMENT) * UNIFORM_ALIGNMENT);
  const buffer = deviceMgr.device!.createBuffer({
    size,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
  });
  deviceMgr.writeBuffer(buffer, 0, params);
  return buffer;
}

function padShapeTo4Left(shape: number[]): [number, number, number, number] {
  if (shape.length === 0) return [1, 1, 1, 1];
  if (shape.length === 1) return [shape[0]!, 1, 1, 1];
  if (shape.length === 2) return [shape[0]!, shape[1]!, 1, 1];
  if (shape.length === 3) return [shape[0]!, shape[1]!, shape[2]!, 1];
  return shape as [number, number, number, number];
}

export class ShapeOps {
  constructor(private deviceMgr: DeviceManager) {}

  async reshape(tensorId: number, shape: number[]): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    if (product(shape) !== meta.length) throw new Error(`reshape: product ${shape} != ${meta.length}`);
    const out = createStorageBuffer(this.deviceMgr.device!, meta.bytes);
    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, out, 0, meta.bytes);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);
    return this.deviceMgr.registerTensorAsHandle(out, shape, meta.dtype, meta.length);
  }

  async flatten(tensorId: number, startDim = 0, endDim = -1): Promise<TensorHandle> {
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const rank = meta.shape.length;
    const s = startDim < 0 ? startDim + rank : startDim;
    const e = endDim < 0 ? endDim + rank : endDim;
    const flatDim = product(meta.shape.slice(s, e + 1));
    const newShape = [...meta.shape.slice(0, s), flatDim, ...meta.shape.slice(e + 1)];
    return this.reshape(tensorId, newShape);
  }

  async squeeze(tensorId: number, dim?: number): Promise<TensorHandle> {
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const newShape = dim !== undefined
      ? meta.shape.filter((_, i) => !(i === (dim < 0 ? dim + meta.shape.length : dim) && meta.shape[i] === 1))
      : meta.shape.filter(s => s !== 1);
    return newShape.length === 0 ? this.reshape(tensorId, [1]) : this.reshape(tensorId, newShape);
  }

  async unsqueeze(tensorId: number, dim: number): Promise<TensorHandle> {
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const rank = meta.shape.length;
    const d = dim < 0 ? dim + rank + 1 : dim;
    const newShape = [...meta.shape.slice(0, d), 1, ...meta.shape.slice(d)];
    return this.reshape(tensorId, newShape);
  }

  async transpose2d(tensorId: number): Promise<TensorHandle> {
    return this.transpose(tensorId, 0, 1);
  }

  async transpose(tensorId: number, dim0: number, dim1: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    if (meta.shape.length === 2) return this.transpose2dImpl(meta);
    const dims = meta.shape.map((_, i) => i);
    [dims[dim0], dims[dim1]] = [dims[dim1]!, dims[dim0]!];
    return this.permute(tensorId, dims);
  }

  async permute(tensorId: number, dims: number[]): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    if (meta.shape.length !== dims.length) throw new Error("permute: dims length mismatch");
    const outShape = dims.map(d => meta.shape[d]!);
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const perm = this.deviceMgr.device!.createBuffer({
      size: dims.length * 4,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST,
    });
    const offset = 4 - dims.length;
    this.deviceMgr.writeBuffer(perm, 0, new Uint32Array(dims.map(d => d + offset)));
    const outShapePadded = padShapeTo4(outShape);
    const srcStridesPadded = computeStrides(padShapeTo4(meta.shape));
    const outStridesPadded = computeStrides(padShapeTo4(outShape));
    const params = new Uint32Array([
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      srcStridesPadded[0], srcStridesPadded[1], srcStridesPadded[2], srcStridesPadded[3],
      outStridesPadded[0], outStridesPadded[1], outStridesPadded[2], outStridesPadded[3],
      dims.length, outLength, 0, 0,
    ]);
    const paramBuffer = createUniformParamBuffer(this.deviceMgr, params, 64);
    const pipeline = getOrCreatePipeline(PERMUTE_ND_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, perm, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    perm.destroy();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, outShape, meta.dtype, outLength);
  }

  async select(tensorId: number, dim: number, index: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const rank = meta.shape.length;
    const d = normalizeDim(dim, rank);
    const outShape = meta.shape.filter((_, i) => i !== d);
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const inShape = padShapeTo4Left(meta.shape);
    const outShapePadded = padShapeTo4Left(outShape);
    const params = new Uint32Array([
      inShape[0], inShape[1], inShape[2], inShape[3],
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      d, index, rank, outLength, 0,
    ]);
    const paramBuffer = createUniformParamBuffer(this.deviceMgr, params, 64);
    const pipeline = getOrCreatePipeline(SELECT_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, outShape, meta.dtype, outLength);
  }

  async slice(tensorId: number, dim: number, start?: number, end?: number, step = 1): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const rank = meta.shape.length;
    const d = normalizeDim(dim, rank);
    const size = meta.shape[d]!;
    const s = normalizeSliceStart(start, size, step);
    const e = normalizeSliceEnd(end, size, step);
    const sliceSize = Math.ceil(Math.abs(e - s) / Math.abs(step));
    const outShape = [...meta.shape];
    outShape[d] = sliceSize;
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const inputShape = padShapeTo4Left(meta.shape);
    const outputShape = padShapeTo4Left(outShape);
    const starts = [0, 0, 0, 0] as [number, number, number, number];
    const steps = [1, 1, 1, 1] as [number, number, number, number];
    starts[d] = s;
    steps[d] = step;
    const params = new Int32Array([
      inputShape[0], inputShape[1], inputShape[2], inputShape[3],
      outputShape[0], outputShape[1], outputShape[2], outputShape[3],
      starts[0], starts[1], starts[2], starts[3],
      steps[0], steps[1], steps[2], steps[3],
      rank, outLength, 0, 0,
    ]);
    const paramBuffer = createUniformParamBuffer(this.deviceMgr, params, 80);
    const pipeline = getOrCreatePipeline(SLICE_SHADER, "slice");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, outShape, meta.dtype, outLength);
  }

  async sliceBackward(
    gradOutputId: number,
    inputShape: number[],
    slicedShape: number[],
    dim: number,
    start: number,
    step: number,
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const gradOutput = this.deviceMgr.getTensorMeta(gradOutputId);
    const inputLength = product(inputShape);
    const gradInput = createStorageBuffer(this.deviceMgr.device!, Math.max(4, inputLength * 4));

    const inputShapePadded = padShapeTo4Left(slicedShape);
    const outputShapePadded = padShapeTo4Left(inputShape);
    const starts = [0, 0, 0, 0] as [number, number, number, number];
    const steps = [1, 1, 1, 1] as [number, number, number, number];
    const rank = inputShape.length;
    const d = dim < 0 ? dim + rank : dim;
    starts[d] = start;
    steps[d] = step;

    const params = new Int32Array([
      inputShapePadded[0], inputShapePadded[1], inputShapePadded[2], inputShapePadded[3],
      outputShapePadded[0], outputShapePadded[1], outputShapePadded[2], outputShapePadded[3],
      starts[0], starts[1], starts[2], starts[3],
      steps[0], steps[1], steps[2], steps[3],
      rank, product(slicedShape), 0, 0,
    ]);
    const paramBuffer = createUniformParamBuffer(this.deviceMgr, params, 80);

    const pipeline = getOrCreatePipeline(SLICE_BACKWARD_SHADER, "slice_backward");
    dispatchCompute(pipeline, [gradOutput.buffer, gradInput, paramBuffer], calculateWorkgroups(product(slicedShape)));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(gradInput, inputShape, gradOutput.dtype as SupportedDType, inputLength);
  }

  async cat(tensorIds: number[], dim: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    if (tensorIds.length !== 2) throw new Error("cat currently supports exactly 2 tensors.");
    const a = this.deviceMgr.getTensorMeta(tensorIds[0]!);
    const b = this.deviceMgr.getTensorMeta(tensorIds[1]!);
    const rank = a.shape.length;
    const d = normalizeDim(dim, rank);
    const outShape = [...a.shape];
    outShape[d] = a.shape[d]! + b.shape[d]!;
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const aShape = padShapeTo4Left(a.shape);
    const bShape = padShapeTo4Left(b.shape);
    const outShapePadded = padShapeTo4Left(outShape);
    const params = new Uint32Array([
      aShape[0], aShape[1], aShape[2], aShape[3],
      bShape[0], bShape[1], bShape[2], bShape[3],
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      d, rank, 0, 0,
    ]);
    const paramBuffer = createUniformParamBuffer(this.deviceMgr, params, 64);
    const pipeline = getOrCreatePipeline(CAT_SHADER, "main");
    dispatchCompute(pipeline, [a.buffer, b.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, outShape, a.dtype, outLength);
  }

  async stack(tensorIds: number[], dim: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    if (tensorIds.length !== 2) throw new Error("stack currently supports exactly 2 tensors.");
    const a = this.deviceMgr.getTensorMeta(tensorIds[0]!);
    const b = this.deviceMgr.getTensorMeta(tensorIds[1]!);
    const rank = a.shape.length;
    const d = normalizeDim(dim, rank + 1);
    const outShape = [...a.shape];
    outShape.splice(d, 0, 2);
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const inShape = padShapeTo4Left(a.shape);
    const outShapePadded = padShapeTo4Left(outShape);
    const params = new Uint32Array([
      inShape[0], inShape[1], inShape[2], inShape[3],
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      d, rank, 0, 0,
    ]);
    const paramBuffer = createUniformParamBuffer(this.deviceMgr, params, 48);
    const pipeline = getOrCreatePipeline(STACK_SHADER, "main");
    dispatchCompute(pipeline, [a.buffer, b.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, outShape, a.dtype, outLength);
  }

  async expand(tensorId: number, shape: number[]): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const outLength = product(shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));

    const rankDiff = shape.length - meta.shape.length;
    const paddedShape = [...new Array(rankDiff).fill(1), ...meta.shape];
    const strides = computeStrides(paddedShape);
    const broadcastStrides = paddedShape.map((s, i) => (s === 1 ? 0 : strides[i]!));
    const outShapePadded = padShapeTo4(shape);
    const bsPadded = padShapeTo4(broadcastStrides);

    const paramsData = new Uint32Array([
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      bsPadded[0], bsPadded[1], bsPadded[2], bsPadded[3],
      shape.length, outLength, 0, 0,
    ]);
    const paramBuffer = createUniformParamBuffer(this.deviceMgr, paramsData, 48);
    const pipeline = getOrCreatePipeline(EXPAND_BROADCAST_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, shape, meta.dtype, outLength);
  }

  async tril(tensorId: number, diagonal = 0): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));

    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, out, 0, meta.bytes);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);

    const params = new Int32Array([diagonal, length, meta.shape[meta.shape.length - 2] ?? 0, meta.shape[meta.shape.length - 1] ?? 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(TRIL_SHADER, "main");
    dispatchCompute(pipeline, [out, out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async triu(tensorId: number, diagonal = 0): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));

    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, out, 0, meta.bytes);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);

    const params = new Int32Array([diagonal, length, meta.shape[meta.shape.length - 2] ?? 0, meta.shape[meta.shape.length - 1] ?? 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(TRIU_SHADER, "main");
    dispatchCompute(pipeline, [out, out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async flip(tensorId: number, dims: number[]): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const length = product(meta.shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, length * 4));
    const params = new Int32Array([
      dims.length, length, 0, 0,
      ...dims.slice(0, 4).map(d => d < 0 ? d + meta.shape.length : d),
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: Math.max(16, Math.ceil(params.byteLength / 16) * 16),
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(FLIP_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, meta.shape, meta.dtype, length);
  }

  async repeat(tensorId: number, sizes: number[]): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const outShape = meta.shape.map((s, i) => s * (sizes[i] ?? 1));
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const outShapePadded = padShapeTo4(outShape);
    const inShapePadded = padShapeTo4(meta.shape);
    const inStrides = computeStrides(meta.shape);
    const outStrides = computeStrides(outShape);
    const repeats = new Array(4).fill(1);
    for (let i = 0; i < sizes.length; i++) {
      repeats[i] = sizes[i] ?? 1;
    }
    const params = new Uint32Array([
      inShapePadded[0], inShapePadded[1], inShapePadded[2], inShapePadded[3],
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      inStrides[0], inStrides[1], inStrides[2], inStrides[3],
      outStrides[0], outStrides[1], outStrides[2], outStrides[3],
      repeats[0], repeats[1], repeats[2], repeats[3],
      meta.shape.length, outLength, 0, 0,
    ]);
    const paramBuffer = createUniformParamBuffer(this.deviceMgr, params, 96);
    const pipeline = getOrCreatePipeline(REPEAT_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, outShape, meta.dtype, outLength);
  }

  async indexSelect(tensorId: number, dim: number, indicesId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const indices = this.deviceMgr.getTensorMeta(indicesId);
    const rank = meta.shape.length;
    if (rank > 2) throw new Error("indexSelect currently supports only 1D and 2D tensors.");
    const d = normalizeDim(dim, rank);
    const outShape = [...meta.shape];
    outShape[d] = indices.length;
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const pipeline = getOrCreatePipeline(INDEX_SELECT_SHADER, rank === 1 ? "index_select_1d" : "index_select_2d");
    if (rank === 1) {
      const params = new Uint32Array([0, 0, 0, indices.length]);
      const paramBuffer = createUniformParamBuffer(this.deviceMgr, params, 16);
      dispatchCompute(pipeline, [meta.buffer, indices.buffer, out, paramBuffer], calculateWorkgroups(outLength));
      await syncDevice();
      paramBuffer.destroy();
    } else {
      const params2d = new Uint32Array([
        d,
        meta.shape[0] ?? 1,
        meta.shape[1] ?? 1,
        indices.length,
      ]);
      const paramBuffer2d = createUniformParamBuffer(this.deviceMgr, params2d, 16);
      dispatchCompute(pipeline, [meta.buffer, indices.buffer, out, paramBuffer2d], calculateWorkgroups(outLength));
      await syncDevice();
      paramBuffer2d.destroy();
    }
    return this.deviceMgr.registerTensorAsHandle(out, outShape, meta.dtype, outLength);
  }

  private async transpose2dImpl(meta: TensorMeta): Promise<TensorHandle> {
    const [rows, cols] = meta.shape;
    const out = createStorageBuffer(this.deviceMgr.device!, meta.bytes);
    const params = new Uint32Array([rows, cols, 0, 0]);
    const paramBuffer = createUniformParamBuffer(this.deviceMgr, params, 16);
    const pipeline = getOrCreatePipeline(TRANSPOSE_SHADER, "transpose_2d");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(rows * cols));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [cols, rows], meta.dtype, rows * cols);
  }
}
