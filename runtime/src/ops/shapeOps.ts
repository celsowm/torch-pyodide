import { TensorHandle, TensorMeta } from "./types.js";
import { cloneHandle, product } from "./types.js";
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
  createStorageBuffer,
  registerTensor,
  normalizeDim,
  computeStrides,
  normalizeSliceStart,
  normalizeSliceEnd,
  padShapeTo4,
  readFromGPU,
} from "./utils.js";
import { DeviceManager } from "./device.js";

export class ShapeOps {
  private toListFn: (tensorId: number) => Promise<number[]>;

  constructor(
    private deviceMgr: DeviceManager,
    private tensors: Map<number, TensorMeta>,
    private nextId: { current: number },
    private allocatedBytes: { current: number },
    toListFn: (tensorId: number) => Promise<number[]>
  ) {
    this.toListFn = toListFn;
  }

  async reshape(tensorId: number, shape: number[]): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const outLength = product(shape);
    if (outLength !== source.length) {
      throw new Error(`reshape mismatch: new shape has ${outLength} elements, expected ${source.length}.`);
    }
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, source.length * 4));
    const encoder = this.deviceMgr.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(source.buffer, 0, out, 0, source.length * 4);
    this.deviceMgr.device!.queue.submit([encoder.finish()]);
    await syncDevice();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  async flatten(tensorId: number, startDim = 0, endDim = -1): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const rank = source.shape.length;
    if (rank === 0) {
      return this.reshape(tensorId, [1]);
    }
    const start = normalizeDim(startDim, rank);
    const end = normalizeDim(endDim, rank);
    if (start > end) {
      throw new Error(`flatten expected start_dim <= end_dim, got ${startDim} > ${endDim}.`);
    }
    const prefix = source.shape.slice(0, start);
    const middle = product(source.shape.slice(start, end + 1));
    const suffix = source.shape.slice(end + 1);
    return this.reshape(tensorId, [...prefix, middle, ...suffix]);
  }

  async squeeze(tensorId: number, dim?: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    let outShape: number[];
    if (dim === undefined || dim === null) {
      outShape = source.shape.filter((value) => value !== 1);
    } else {
      const resolved = normalizeDim(dim, source.shape.length);
      outShape = [...source.shape];
      if (outShape[resolved] !== 1) {
        return this.reshape(tensorId, outShape);
      }
      outShape.splice(resolved, 1);
    }
    return this.reshape(tensorId, outShape);
  }

  async unsqueeze(tensorId: number, dim: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const rank = source.shape.length;
    const resolved = dim < 0 ? dim + rank + 1 : dim;
    if (resolved < 0 || resolved > rank) {
      throw new Error(`unsqueeze dim out of range for rank ${rank}: ${dim}.`);
    }
    const outShape = [...source.shape];
    outShape.splice(resolved, 0, 1);
    return this.reshape(tensorId, outShape);
  }

  async transpose2d(tensorId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    if (source.shape.length !== 2) {
      throw new Error("transpose2d currently supports only rank-2 tensors.");
    }
    const [rows, cols] = source.shape;
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, source.length * 4));
    const dimsData = new Uint32Array([rows, cols]);
    const dimsBuffer = this.deviceMgr.device!.createBuffer({
      size: dimsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(dimsBuffer, 0, dimsData);
    const pipeline = getOrCreatePipeline(TRANSPOSE_SHADER, "transpose_2d");
    dispatchCompute(pipeline, [source.buffer, out, dimsBuffer], calculateWorkgroups(source.length));
    await syncDevice();
    dimsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, [cols, rows], source.dtype, source.length);
    return cloneHandle(meta);
  }

  async transpose(tensorId: number, dim0: number, dim1: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const rank = source.shape.length;
    const d0 = normalizeDim(dim0, rank);
    const d1 = normalizeDim(dim1, rank);
    if (rank === 2 && ((d0 === 0 && d1 === 1) || (d0 === 1 && d1 === 0))) {
      return this.transpose2d(tensorId);
    }
    const perm = [...Array(rank).keys()];
    [perm[d0], perm[d1]] = [perm[d1]!, perm[d0]!];
    return this.permute(tensorId, perm);
  }

  async permute(tensorId: number, dims: number[]): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const rank = source.shape.length;
    if (dims.length !== rank) {
      throw new Error(`permute dims length ${dims.length} must match rank ${rank}.`);
    }
    const normalized = dims.map((dim) => normalizeDim(dim, rank));
    if (new Set(normalized).size !== rank) {
      throw new Error("permute dims must be a permutation without repeats.");
    }
    if (rank > 4) {
      throw new Error("permute currently supports up to 4D tensors.");
    }
    const outShape = normalized.map((axis) => source.shape[axis]!);
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));

    const inStrides = computeStrides(source.shape);
    const outStrides = computeStrides(outShape);

    const inStridesPadded = padShapeTo4(inStrides);
    const outShapePadded = padShapeTo4(outShape);
    const outStridesPadded = padShapeTo4(outStrides);

    const paramsData = new Uint32Array([
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      inStridesPadded[0], inStridesPadded[1], inStridesPadded[2], inStridesPadded[3],
      outStridesPadded[0], outStridesPadded[1], outStridesPadded[2], outStridesPadded[3],
      rank, outLength, 0, 0,
    ]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const permData = new Uint32Array(normalized);
    const permBuffer = this.deviceMgr.device!.createBuffer({
      size: permData.byteLength,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(permBuffer, 0, permData);

    const pipeline = getOrCreatePipeline(PERMUTE_ND_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, permBuffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    permBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, outShape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  async select(tensorId: number, dim: number, index: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const rank = source.shape.length;
    if (rank === 0) {
      throw new Error("select is not supported for scalar tensors.");
    }
    if (rank > 4) {
      throw new Error("select currently supports up to 4D tensors.");
    }
    const resolvedDim = normalizeDim(dim, rank);
    const axisSize = source.shape[resolvedDim]!;
    const resolvedIndex = index < 0 ? index + axisSize : index;
    if (resolvedIndex < 0 || resolvedIndex >= axisSize) {
      throw new Error(`select index out of range for dim ${dim}: ${index}.`);
    }
    const outShape = source.shape.slice(0, resolvedDim).concat(source.shape.slice(resolvedDim + 1));
    const outLength = Math.max(1, product(outShape));
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const inShapePadded = padShapeTo4(source.shape);
    const outShapePadded = padShapeTo4(outShape);
    const paramsData = new Uint32Array([
      inShapePadded[0], inShapePadded[1], inShapePadded[2], inShapePadded[3],
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      resolvedDim, resolvedIndex, rank, outLength,
    ]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(SELECT_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, outShape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  async slice(
    tensorId: number,
    dim: number,
    start?: number,
    end?: number,
    step = 1
  ): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    if (step === 0) {
      throw new Error("slice step must be non-zero.");
    }
    const rank = source.shape.length;
    if (rank > 4) {
      throw new Error("slice currently supports up to 4D tensors.");
    }
    const resolvedDim = normalizeDim(dim, rank);
    const axisSize = source.shape[resolvedDim]!;
    const normalizedStart = normalizeSliceStart(start, axisSize, step);
    const normalizedEnd = normalizeSliceEnd(end, axisSize, step);

    let axisIndicesCount: number;
    if (step > 0) {
      axisIndicesCount = Math.max(0, Math.ceil((normalizedEnd - normalizedStart) / step));
    } else {
      axisIndicesCount = Math.max(0, Math.ceil((normalizedStart - normalizedEnd) / (-step)));
    }

    const outShape = [...source.shape];
    outShape[resolvedDim] = axisIndicesCount;
    const outLength = product(outShape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));

    const inShapePadded = padShapeTo4(source.shape);
    const outShapePadded = padShapeTo4(outShape);

    const starts = new Array(4).fill(0);
    const stepsArr = new Array(4).fill(1);
    starts[resolvedDim] = normalizedStart;
    stepsArr[resolvedDim] = step;

    const paramsData = new Int32Array([
      inShapePadded[0], inShapePadded[1], inShapePadded[2], inShapePadded[3],
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      starts[0]!, starts[1]!, starts[2]!, starts[3]!,
      stepsArr[0]!, stepsArr[1]!, stepsArr[2]!, stepsArr[3]!,
      rank, outLength, 0, 0,
    ]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(SLICE_SHADER, "slice");
    dispatchCompute(pipeline, [source.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, outShape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  async cat(tensorIds: number[], dim: number): Promise<TensorHandle> {
    return this.catStack(tensorIds, dim, false);
  }

  async stack(tensorIds: number[], dim: number): Promise<TensorHandle> {
    return this.catStack(tensorIds, dim, true);
  }

  async expand(tensorId: number, shape: number[]): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    if (shape.length > 4) {
      throw new Error("expand currently supports up to 4D tensors.");
    }
    if (source.shape.length > 4) {
      throw new Error("expand currently supports up to 4D input tensors.");
    }
    const rankDiff = shape.length - source.shape.length;
    const paddedSourceShape = [...new Array(rankDiff).fill(1), ...source.shape];
    for (let i = 0; i < shape.length; i++) {
      if (paddedSourceShape[i] !== 1 && paddedSourceShape[i] !== shape[i]) {
        throw new Error(`expand: shape[${i}] mismatch: ${paddedSourceShape[i]} vs ${shape[i]}.`);
      }
    }
    const outLength = product(shape);
    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const srcStrides = computeStrides(paddedSourceShape);
    const padded = padShapeTo4(paddedSourceShape);
    const broadcastStrides = paddedSourceShape.map((s, i) => (s === 1 ? 0 : srcStrides[i]!));
    const bsPadded = padShapeTo4(broadcastStrides);
    const paramsData = new Uint32Array([
      padded[0], padded[1], padded[2], padded[3],
      bsPadded[0], bsPadded[1], bsPadded[2], bsPadded[3],
    ]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(EXPAND_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, shape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  async indexSelect(tensorId: number, dim: number, indicesId: number): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    const source = this.getMeta(tensorId);
    const indices = this.getMeta(indicesId);
    if (source.shape.length > 2) {
      throw new Error("indexSelect currently supports up to 2D input tensors.");
    }
    if (indices.shape.length !== 1) {
      throw new Error("indexSelect requires 1D indices tensor.");
    }
    const resolvedDim = normalizeDim(dim, source.shape.length);
    const numIndices = indices.length;

    let outShape: number[];
    let outLength: number;
    if (source.shape.length === 1) {
      outShape = [numIndices];
      outLength = numIndices;
    } else {
      if (resolvedDim === 0) {
        outShape = [numIndices, source.shape[1]!];
      } else {
        outShape = [source.shape[0]!, numIndices];
      }
      outLength = product(outShape);
    }

    const indexValues = await this.toListFn(indicesId);
    const indicesInt32 = new Int32Array(indexValues.map((v) => Math.trunc(v)));
    const indicesBuffer = this.deviceMgr.device!.createBuffer({
      size: indicesInt32.byteLength,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(indicesBuffer, 0, indicesInt32);

    const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
    const paramsData = new Uint32Array([
      resolvedDim,
      source.shape.length >= 1 ? source.shape[0]! : 1,
      source.shape.length >= 2 ? source.shape[1]! : 1,
      numIndices,
    ]);
    const paramsBuffer = this.deviceMgr.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const shader = source.shape.length === 1 ? "index_select_1d" : "index_select_2d";
    const pipeline = getOrCreatePipeline(INDEX_SELECT_SHADER, shader);
    dispatchCompute(pipeline, [source.buffer, indicesBuffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    indicesBuffer.destroy();
    const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, outShape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  private async catStack(tensorIds: number[], dim: number, isStack: boolean): Promise<TensorHandle> {
    await this.deviceMgr.ensureReady();
    if (tensorIds.length !== 2) {
      throw new Error(`${isStack ? "stack" : "cat"} currently supports exactly 2 tensors.`);
    }
    const a = this.getMeta(tensorIds[0]!);
    const b = this.getMeta(tensorIds[1]!);
    if (a.dtype !== b.dtype) {
      throw new Error(`${isStack ? "stack" : "cat"} requires tensors with same dtype.`);
    }
    const rank = a.shape.length;

    if (isStack) {
      const resolvedDim = normalizeDim(dim, rank + 1);
      for (let i = 0; i < rank; i++) {
        if (a.shape[i] !== b.shape[i]) {
          throw new Error(`stack requires tensors with same shape, got [${a.shape}] vs [${b.shape}].`);
        }
      }
      const outShape = [...a.shape];
      outShape.splice(resolvedDim, 0, 2);
      const outLength = product(outShape);
      const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
      const inShapePadded = padShapeTo4(a.shape);
      const outShapePadded = padShapeTo4(outShape);
      const paramsData = new Uint32Array([
        inShapePadded[0], inShapePadded[1], inShapePadded[2], inShapePadded[3],
        outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
        resolvedDim, rank, 0, 0
      ]);
      const paramsBuffer = this.deviceMgr.device!.createBuffer({
        size: paramsData.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
      });
      this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(STACK_SHADER, "main");
      dispatchCompute(pipeline, [a.buffer, b.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
      await syncDevice();
      paramsBuffer.destroy();
      const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, outShape, a.dtype, outLength);
      return cloneHandle(meta);
    } else {
      const resolvedDim = normalizeDim(dim, rank);
      if (rank === 0) {
        throw new Error("cat requires at least 1D tensors.");
      }
      for (let i = 0; i < rank; i++) {
        if (i !== resolvedDim && a.shape[i] !== b.shape[i]) {
          throw new Error(`cat requires tensors with same shape except dim ${dim}, got [${a.shape}] vs [${b.shape}].`);
        }
      }
      const outShape = [...a.shape];
      outShape[resolvedDim] = a.shape[resolvedDim] + b.shape[resolvedDim];
      const outLength = product(outShape);
      const out = createStorageBuffer(this.deviceMgr.device!, Math.max(4, outLength * 4));
      const aShapePadded = padShapeTo4(a.shape);
      const bShapePadded = padShapeTo4(b.shape);
      const outShapePadded = padShapeTo4(outShape);
      const paramsData = new Uint32Array([
        aShapePadded[0], aShapePadded[1], aShapePadded[2], aShapePadded[3],
        bShapePadded[0], bShapePadded[1], bShapePadded[2], bShapePadded[3],
        outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
        resolvedDim, rank, 0, 0
      ]);
      const paramsBuffer = this.deviceMgr.device!.createBuffer({
        size: paramsData.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
      });
      this.deviceMgr.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(CAT_SHADER, "main");
      dispatchCompute(pipeline, [a.buffer, b.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
      await syncDevice();
      paramsBuffer.destroy();
      const meta = registerTensor(this.tensors, this.nextId, this.allocatedBytes, out, outShape, a.dtype, outLength);
      return cloneHandle(meta);
    }
  }

  private getMeta(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }
}
