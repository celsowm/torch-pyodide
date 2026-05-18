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
  EXPAND_SHADER,
  INDEX_SELECT_SHADER,
  normalizeDim,
  computeStrides,
  normalizeSliceStart,
  normalizeSliceEnd,
  padShapeTo4,
  createStorageBuffer,
} from "./utils.js";
import { DeviceManager } from "./device.js";

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
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
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
    const sPadded = padShapeTo4(meta.shape);
    const strides = computeStrides(meta.shape);
    const params = new Uint32Array([
      sPadded[0], sPadded[1], sPadded[2], sPadded[3],
      strides[0]!, strides[1]!, strides[2]!, strides[3]!,
      d + (4 - rank), index, outLength,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
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
    const sPadded = padShapeTo4(meta.shape);
    const strides = computeStrides(meta.shape);
    const params = new Int32Array([
      sPadded[0], sPadded[1], sPadded[2], sPadded[3],
      strides[0]!, strides[1]!, strides[2]!, strides[3]!,
      d + (4 - rank), s, e, step, outLength,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(SLICE_SHADER, "slice");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, outShape, meta.dtype, outLength);
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
    const sPadded = padShapeTo4(a.shape);
    const params = new Uint32Array([sPadded[0], sPadded[1], sPadded[2], sPadded[3], d + (4 - rank), a.shape[d]!, outLength]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
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
    const sPadded = padShapeTo4(a.shape);
    const params = new Uint32Array([sPadded[0], sPadded[1], sPadded[2], sPadded[3], d + (4 - rank), outLength]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
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
    const sPadded = padShapeTo4(meta.shape);
    const strides = computeStrides(meta.shape);
    const broadcastStrides = meta.shape.map((s, i) => (s === 1 ? 0 : strides[i]!));
    const bsPadded = padShapeTo4(broadcastStrides);
    const params = new Uint32Array([
      sPadded[0], sPadded[1], sPadded[2], sPadded[3],
      bsPadded[0], bsPadded[1], bsPadded[2], bsPadded[3],
      outLength,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(EXPAND_SHADER, "main");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, shape, meta.dtype, outLength);
  }

  async indexSelect(tensorId: number, dim: number, indicesId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const meta = this.deviceMgr.getTensorMeta(tensorId);
    const indices = this.deviceMgr.getTensorMeta(indicesId);
    const data = await this.deviceMgr.readFromGPU(indices.buffer, indices.length, "int32");
    const rank = meta.shape.length;
    const d = normalizeDim(dim, rank);
    const outShape = [...meta.shape];
    outShape[d] = indices.length;
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const sPadded = padShapeTo4(meta.shape);
    const strides = computeStrides(meta.shape);
    const params = new Uint32Array([
      sPadded[0], sPadded[1], sPadded[2], sPadded[3],
      strides[0]!, strides[1]!, strides[2]!, strides[3]!,
      d + (4 - rank), indices.length, ...data,
    ]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: (12 + data.length) * 4,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(INDEX_SELECT_SHADER, "index_select_2d");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, outShape, meta.dtype, outLength);
  }

  private async transpose2dImpl(meta: TensorMeta): Promise<TensorHandle> {
    const [rows, cols] = meta.shape;
    const out = createStorageBuffer(this.deviceMgr.device!, meta.bytes);
    const params = new Uint32Array([rows, cols, 0, 0]);
    const paramBuffer = this.deviceMgr.device!.createBuffer({
      size: params.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    this.deviceMgr.writeBuffer(paramBuffer, 0, params);
    const pipeline = getOrCreatePipeline(TRANSPOSE_SHADER, "transpose_2d");
    dispatchCompute(pipeline, [meta.buffer, out, paramBuffer], calculateWorkgroups(rows * cols));
    await syncDevice();
    paramBuffer.destroy();
    return this.deviceMgr.registerTensorAsHandle(out, [cols, rows], meta.dtype, rows * cols);
  }
}
