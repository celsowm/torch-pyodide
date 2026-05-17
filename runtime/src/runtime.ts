import {
  initWebGPU,
  getDevice,
  getAdapter,
  isInitialized as isWebGPUInitialized,
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  MapMode,
  ELEMENTWISE_SHADER,
  FILL_SHADER,
  RANDOM_SHADER,
  MATMUL_SHADER,
  REDUCE_SUM_SHADER,
  CLAMP_SHADER,
  WHERE_SHADER,
  ARGMAX_SHADER,
  ARGMIN_SHADER,
  UNARY_SHADER,
  TRANSPOSE_SHADER,
  CAT_SHADER,
  STACK_SHADER,
  PERMUTE_ND_SHADER,
  SELECT_SHADER,
  SLICE_SHADER,
  EXPAND_SHADER,
  INDEX_SELECT_SHADER
} from "./vendor/torchjs/index.js";

type TensorMeta = {
  id: number;
  buffer: GPUBuffer;
  shape: number[];
  dtype: string;
  length: number;
  bytes: number;
};

type TensorHandle = {
  id: number;
  shape: number[];
  dtype: string;
};

type SupportedDType = "float32" | "int32" | "bool";

function product(values: number[]): number {
  return values.reduce((acc, value) => acc * value, 1);
}

function cloneHandle(meta: TensorMeta): TensorHandle {
  return {
    id: meta.id,
    shape: [...meta.shape],
    dtype: meta.dtype
  };
}

export class TorchPyodideRuntime {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private initialized = false;
  private initPromise: Promise<void> | null = null;
  private initError: string | null = null;
  private nextId = 1;
  private tensors = new Map<number, TensorMeta>();
  private currentAllocatedBytes = 0;

  async init(gpuProvider?: GPU | null): Promise<void> {
    await this.ensureReady(gpuProvider);
  }

  async tensorFromData(data: number[], shape: number[], dtype: string): Promise<TensorHandle> {
    await this.ensureReady();
    this.assertDType(dtype);
    const length = product(shape);
    if (length !== data.length) {
      throw new Error(`tensorFromData expected ${length} values, got ${data.length}.`);
    }
    const typed = new Float32Array(data.map((value) => this.coerceScalarByDType(value, dtype as SupportedDType)));
    const buffer = this.createStorageBuffer(typed.byteLength);
    this.device!.queue.writeBuffer(buffer, 0, typed);
    const meta = this.registerTensor(buffer, shape, dtype, length);
    return cloneHandle(meta);
  }

  async zeros(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, 0.0);
  }

  async ones(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, 1.0);
  }

  async rand(shape: number[], dtype: string): Promise<TensorHandle> {
    await this.ensureReady();
    this.assertDType(dtype);
    const length = product(shape);
    const out = this.createStorageBuffer(Math.max(4, length * 4));
    const paramsData = new Uint32Array([Math.floor(Math.random() * 0xffffffff), length]);
    const paramsBuffer = this.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(RANDOM_SHADER, "rand");
    dispatchCompute(pipeline, [out, paramsBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = this.registerTensor(out, shape, dtype, length);
    return cloneHandle(meta);
  }

  async randn(shape: number[], dtype: string): Promise<TensorHandle> {
    await this.ensureReady();
    this.assertDType(dtype);
    const length = product(shape);
    const out = this.createStorageBuffer(Math.max(4, length * 4));
    const paramsData = new Uint32Array([Math.floor(Math.random() * 0xffffffff), length]);
    const paramsBuffer = this.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(RANDOM_SHADER, "randn");
    dispatchCompute(pipeline, [out, paramsBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = this.registerTensor(out, shape, dtype, length);
    return cloneHandle(meta);
  }

  async arange(start: number, end: number, step: number, dtype: string): Promise<TensorHandle> {
    await this.ensureReady();
    this.assertDType(dtype);
    if (step === 0) {
      throw new Error("arange step must be non-zero.");
    }
    const values: number[] = [];
    if (step > 0) {
      for (let value = start; value < end; value += step) {
        values.push(this.coerceScalarByDType(value, dtype as SupportedDType));
      }
    } else {
      for (let value = start; value > end; value += step) {
        values.push(this.coerceScalarByDType(value, dtype as SupportedDType));
      }
    }
    return this.tensorFromData(values, [values.length], dtype);
  }

  async full(shape: number[], fillValue: number, dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, fillValue);
  }

  async fullLike(tensorId: number, fillValue: number, dtype?: string): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const outDtype = dtype ?? source.dtype;
    return this.fill(source.shape, outDtype, fillValue);
  }

  async add(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "add");
  }

  async mul(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "mul");
  }

  async sub(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "sub");
  }

  async div(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "div_op");
  }

  async relu(tensorId: number): Promise<TensorHandle> {
    this.assertUnaryDType(this.getTensor(tensorId).dtype, "relu");
    return this.unary(tensorId, "relu");
  }

  async abs(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "abs_op");
  }

  async sqrt(tensorId: number): Promise<TensorHandle> {
    this.assertUnaryDType(this.getTensor(tensorId).dtype, "sqrt");
    return this.unary(tensorId, "sqrt_op");
  }

  async exp(tensorId: number): Promise<TensorHandle> {
    this.assertUnaryDType(this.getTensor(tensorId).dtype, "exp");
    return this.unary(tensorId, "exp_op");
  }

  async log(tensorId: number): Promise<TensorHandle> {
    this.assertUnaryDType(this.getTensor(tensorId).dtype, "log");
    return this.unary(tensorId, "log_op");
  }

  async neg(tensorId: number): Promise<TensorHandle> {
    return this.unary(tensorId, "neg");
  }

  private async unary(
    tensorId: number,
    entrypoint: "relu" | "abs_op" | "sqrt_op" | "exp_op" | "log_op" | "neg"
  ): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const out = this.createStorageBuffer(Math.max(4, source.length * 4));
    const pipeline = getOrCreatePipeline(UNARY_SHADER, entrypoint);
    dispatchCompute(pipeline, [source.buffer, out], calculateWorkgroups(source.length));
    await syncDevice();
    const meta = this.registerTensor(out, source.shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  async clamp(tensorId: number, minVal: number, maxVal: number): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const out = this.createStorageBuffer(Math.max(4, source.length * 4));
    const params = new ArrayBuffer(16);
    const view = new DataView(params);
    view.setFloat32(0, minVal, true);
    view.setFloat32(4, maxVal, true);
    view.setUint32(8, source.length, true);
    view.setUint32(12, 0, true);
    const paramsBuffer = this.device!.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramsBuffer, 0, params);
    const pipeline = getOrCreatePipeline(CLAMP_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, out, paramsBuffer], calculateWorkgroups(source.length));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = this.registerTensor(out, source.shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  async where(conditionId: number, xId: number, yId: number): Promise<TensorHandle> {
    await this.ensureReady();
    const condition = this.getTensor(conditionId);
    const x = this.getTensor(xId);
    const y = this.getTensor(yId);
    if (condition.length !== x.length || x.length !== y.length) {
      throw new Error("where requires condition, x and y with same number of elements.");
    }
    if (condition.shape.join(",") !== x.shape.join(",") || x.shape.join(",") !== y.shape.join(",")) {
      throw new Error("where requires condition, x and y with same shape.");
    }
    const out = this.createStorageBuffer(Math.max(4, x.length * 4));
    const pipeline = getOrCreatePipeline(WHERE_SHADER, "main");
    dispatchCompute(pipeline, [condition.buffer, x.buffer, y.buffer, out], calculateWorkgroups(x.length));
    await syncDevice();
    const meta = this.registerTensor(out, x.shape, x.dtype, x.length);
    return cloneHandle(meta);
  }

  async argmax(tensorId: number): Promise<TensorHandle> {
    return this.argReduce(tensorId, true);
  }

  async argmin(tensorId: number): Promise<TensorHandle> {
    return this.argReduce(tensorId, false);
  }

  async matmul(aId: number, bId: number): Promise<TensorHandle> {
    await this.ensureReady();
    const a = this.getTensor(aId);
    const b = this.getTensor(bId);
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("matmul currently supports only 2D tensors.");
    }
    const [m, kA] = a.shape;
    const [kB, n] = b.shape;
    if (kA !== kB) {
      throw new Error(`matmul dimension mismatch: ${kA} != ${kB}.`);
    }
    const outLength = m * n;
    const outBuffer = this.createStorageBuffer(outLength * 4);

    const dimsData = new Uint32Array([m, kA, n, 1]);
    const dimsBuffer = this.device!.createBuffer({
      size: dimsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(dimsBuffer, 0, dimsData);

    const pipeline = getOrCreatePipeline(MATMUL_SHADER, "matmul_2d");
    dispatchCompute(
      pipeline,
      [a.buffer, b.buffer, outBuffer, dimsBuffer],
      calculateWorkgroups(outLength)
    );
    await syncDevice();
    dimsBuffer.destroy();

    const out = this.registerTensor(outBuffer, [m, n], "float32", outLength);
    return cloneHandle(out);
  }

  async sum(tensorId: number): Promise<TensorHandle> {
    return this.reduce(tensorId, false);
  }

  async mean(tensorId: number): Promise<TensorHandle> {
    return this.reduce(tensorId, true);
  }

  async reshape(tensorId: number, shape: number[]): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const outLength = product(shape);
    if (outLength !== source.length) {
      throw new Error(`reshape mismatch: new shape has ${outLength} elements, expected ${source.length}.`);
    }
    const out = this.createStorageBuffer(Math.max(4, source.length * 4));
    const encoder = this.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(source.buffer, 0, out, 0, source.length * 4);
    this.device!.queue.submit([encoder.finish()]);
    await syncDevice();
    const meta = this.registerTensor(out, shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  async flatten(tensorId: number, startDim = 0, endDim = -1): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const rank = source.shape.length;
    if (rank === 0) {
      return this.reshape(tensorId, [1]);
    }
    const start = this.normalizeDim(startDim, rank);
    const end = this.normalizeDim(endDim, rank);
    if (start > end) {
      throw new Error(`flatten expected start_dim <= end_dim, got ${startDim} > ${endDim}.`);
    }
    const prefix = source.shape.slice(0, start);
    const middle = product(source.shape.slice(start, end + 1));
    const suffix = source.shape.slice(end + 1);
    return this.reshape(tensorId, [...prefix, middle, ...suffix]);
  }

  async squeeze(tensorId: number, dim?: number): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    let outShape: number[];
    if (dim === undefined || dim === null) {
      outShape = source.shape.filter((value) => value !== 1);
    } else {
      const resolved = this.normalizeDim(dim, source.shape.length);
      outShape = [...source.shape];
      if (outShape[resolved] !== 1) {
        return this.reshape(tensorId, outShape);
      }
      outShape.splice(resolved, 1);
    }
    return this.reshape(tensorId, outShape);
  }

  async unsqueeze(tensorId: number, dim: number): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
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
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    if (source.shape.length !== 2) {
      throw new Error("transpose2d currently supports only rank-2 tensors.");
    }
    const [rows, cols] = source.shape;
    const out = this.createStorageBuffer(Math.max(4, source.length * 4));
    const dimsData = new Uint32Array([rows, cols]);
    const dimsBuffer = this.device!.createBuffer({
      size: dimsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(dimsBuffer, 0, dimsData);
    const pipeline = getOrCreatePipeline(TRANSPOSE_SHADER, "transpose_2d");
    dispatchCompute(pipeline, [source.buffer, out, dimsBuffer], calculateWorkgroups(source.length));
    await syncDevice();
    dimsBuffer.destroy();
    const meta = this.registerTensor(out, [cols, rows], source.dtype, source.length);
    return cloneHandle(meta);
  }

  async transpose(tensorId: number, dim0: number, dim1: number): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const rank = source.shape.length;
    const d0 = this.normalizeDim(dim0, rank);
    const d1 = this.normalizeDim(dim1, rank);
    if (rank === 2 && ((d0 === 0 && d1 === 1) || (d0 === 1 && d1 === 0))) {
      return this.transpose2d(tensorId);
    }
    const perm = [...Array(rank).keys()];
    [perm[d0], perm[d1]] = [perm[d1]!, perm[d0]!];
    return this.permute(tensorId, perm);
  }

  async permute(tensorId: number, dims: number[]): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const rank = source.shape.length;
    if (dims.length !== rank) {
      throw new Error(`permute dims length ${dims.length} must match rank ${rank}.`);
    }
    const normalized = dims.map((dim) => this.normalizeDim(dim, rank));
    if (new Set(normalized).size !== rank) {
      throw new Error("permute dims must be a permutation without repeats.");
    }
    if (rank > 4) {
      throw new Error("permute currently supports up to 4D tensors.");
    }
    const outShape = normalized.map((axis) => source.shape[axis]!);
    const outLength = product(outShape);
    const out = this.createStorageBuffer(Math.max(4, outLength * 4));

    const inStrides = this.computeStrides(source.shape);
    const outStrides = this.computeStrides(outShape);

    const inStridesPadded = this.padShapeTo4(inStrides);
    const outShapePadded = this.padShapeTo4(outShape);
    const outStridesPadded = this.padShapeTo4(outStrides);

    const paramsData = new Uint32Array([
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      inStridesPadded[0], inStridesPadded[1], inStridesPadded[2], inStridesPadded[3],
      outStridesPadded[0], outStridesPadded[1], outStridesPadded[2], outStridesPadded[3],
      rank, outLength, 0, 0,
    ]);
    const paramsBuffer = this.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);

    // Permutation array as storage buffer (u32)
    const permData = new Uint32Array(normalized);
    const permBuffer = this.device!.createBuffer({
      size: permData.byteLength,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(permBuffer, 0, permData);

    const pipeline = getOrCreatePipeline(PERMUTE_ND_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, permBuffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    permBuffer.destroy();
    const meta = this.registerTensor(out, outShape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  async select(tensorId: number, dim: number, index: number): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const rank = source.shape.length;
    if (rank === 0) {
      throw new Error("select is not supported for scalar tensors.");
    }
    if (rank > 4) {
      throw new Error("select currently supports up to 4D tensors.");
    }
    const resolvedDim = this.normalizeDim(dim, rank);
    const axisSize = source.shape[resolvedDim]!;
    const resolvedIndex = index < 0 ? index + axisSize : index;
    if (resolvedIndex < 0 || resolvedIndex >= axisSize) {
      throw new Error(`select index out of range for dim ${dim}: ${index}.`);
    }
    const outShape = source.shape.slice(0, resolvedDim).concat(source.shape.slice(resolvedDim + 1));
    const outLength = Math.max(1, product(outShape));
    const out = this.createStorageBuffer(Math.max(4, outLength * 4));
    const inShapePadded = this.padShapeTo4(source.shape);
    const outShapePadded = this.padShapeTo4(outShape);
    const paramsData = new Uint32Array([
      inShapePadded[0], inShapePadded[1], inShapePadded[2], inShapePadded[3],
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      resolvedDim, resolvedIndex, rank, outLength,
    ]);
    const paramsBuffer = this.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(SELECT_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = this.registerTensor(out, outShape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  async slice(
    tensorId: number,
    dim: number,
    start?: number,
    end?: number,
    step = 1
  ): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    if (step === 0) {
      throw new Error("slice step must be non-zero.");
    }
    const rank = source.shape.length;
    if (rank > 4) {
      throw new Error("slice currently supports up to 4D tensors.");
    }
    const resolvedDim = this.normalizeDim(dim, rank);
    const axisSize = source.shape[resolvedDim]!;
    const normalizedStart = this.normalizeSliceStart(start, axisSize, step);
    const normalizedEnd = this.normalizeSliceEnd(end, axisSize, step);

    let axisIndicesCount: number;
    if (step > 0) {
      axisIndicesCount = Math.max(0, Math.ceil((normalizedEnd - normalizedStart) / step));
    } else {
      axisIndicesCount = Math.max(0, Math.ceil((normalizedStart - normalizedEnd) / (-step)));
    }

    const outShape = [...source.shape];
    outShape[resolvedDim] = axisIndicesCount;
    const outLength = product(outShape);
    const out = this.createStorageBuffer(Math.max(4, outLength * 4));

    const inShapePadded = this.padShapeTo4(source.shape);
    const outShapePadded = this.padShapeTo4(outShape);

    // Slice shader params: input_shape, output_shape, starts, steps, ndim, output_size
    // Only one dim varies; for other dims start=0, step=1
    const starts = new Array(4).fill(0);
    const steps = new Array(4).fill(1);
    starts[resolvedDim] = normalizedStart;
    steps[resolvedDim] = step;

    const paramsData = new Int32Array([
      inShapePadded[0], inShapePadded[1], inShapePadded[2], inShapePadded[3],
      outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
      starts[0]!, starts[1]!, starts[2]!, starts[3]!,
      steps[0]!, steps[1]!, steps[2]!, steps[3]!,
      rank, outLength, 0, 0,
    ]);
    const paramsBuffer = this.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(SLICE_SHADER, "slice");
    dispatchCompute(pipeline, [source.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = this.registerTensor(out, outShape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  async toList(tensorId: number): Promise<number[]> {
    await this.ensureReady();
    const meta = this.getTensor(tensorId);
    const readBuffer = this.device!.createBuffer({
      size: meta.length * 4,
      usage: BufferUsage.COPY_DST | BufferUsage.MAP_READ
    });
    const encoder = this.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, readBuffer, 0, meta.length * 4);
    this.device!.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(MapMode.READ);
    const copied = readBuffer.getMappedRange();
    const copiedBuffer = copied.slice(0);
    const values = this.decodeValuesByDType(copiedBuffer, meta.dtype as SupportedDType);
    readBuffer.unmap();
    readBuffer.destroy();
    return values;
  }

  async destroy(tensorId: number): Promise<void> {
    const meta = this.tensors.get(tensorId);
    if (!meta) {
      return;
    }
    meta.buffer.destroy();
    this.currentAllocatedBytes = Math.max(0, this.currentAllocatedBytes - meta.bytes);
    this.tensors.delete(tensorId);
  }

  async cat(tensorIds: number[], dim: number): Promise<TensorHandle> {
    return this.catStack(tensorIds, dim, false);
  }

  async stack(tensorIds: number[], dim: number): Promise<TensorHandle> {
    return this.catStack(tensorIds, dim, true);
  }

  private async catStack(tensorIds: number[], dim: number, isStack: boolean): Promise<TensorHandle> {
    await this.ensureReady();
    if (tensorIds.length !== 2) {
      throw new Error(`${isStack ? "stack" : "cat"} currently supports exactly 2 tensors.`);
    }
    const a = this.getTensor(tensorIds[0]!);
    const b = this.getTensor(tensorIds[1]!);
    const ndim = isStack ? a.shape.length : a.shape.length;
    if (a.dtype !== b.dtype) {
      throw new Error(`${isStack ? "stack" : "cat"} requires tensors with same dtype.`);
    }
    const rank = a.shape.length;

    if (isStack) {
      const resolvedDim = this.normalizeDim(dim, rank + 1);
      for (let i = 0; i < rank; i++) {
        if (a.shape[i] !== b.shape[i]) {
          throw new Error(`stack requires tensors with same shape, got [${a.shape}] vs [${b.shape}].`);
        }
      }
      const outShape = [...a.shape];
      outShape.splice(resolvedDim, 0, 2);
      const outLength = product(outShape);
      const out = this.createStorageBuffer(Math.max(4, outLength * 4));
      const inShapePadded = this.padShapeTo4(a.shape);
      const outShapePadded = this.padShapeTo4(outShape);
      const paramsData = new Uint32Array([
        inShapePadded[0], inShapePadded[1], inShapePadded[2], inShapePadded[3],
        outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
        resolvedDim, ndim, 0, 0
      ]);
      const paramsBuffer = this.device!.createBuffer({
        size: paramsData.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
      });
      this.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(STACK_SHADER, "main");
      dispatchCompute(pipeline, [a.buffer, b.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
      await syncDevice();
      paramsBuffer.destroy();
      const meta = this.registerTensor(out, outShape, a.dtype, outLength);
      return cloneHandle(meta);
    } else {
      const resolvedDim = this.normalizeDim(dim, rank);
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
      const out = this.createStorageBuffer(Math.max(4, outLength * 4));
      const aShapePadded = this.padShapeTo4(a.shape);
      const bShapePadded = this.padShapeTo4(b.shape);
      const outShapePadded = this.padShapeTo4(outShape);
      const paramsData = new Uint32Array([
        aShapePadded[0], aShapePadded[1], aShapePadded[2], aShapePadded[3],
        bShapePadded[0], bShapePadded[1], bShapePadded[2], bShapePadded[3],
        outShapePadded[0], outShapePadded[1], outShapePadded[2], outShapePadded[3],
        resolvedDim, ndim, 0, 0
      ]);
      const paramsBuffer = this.device!.createBuffer({
        size: paramsData.byteLength,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
      });
      this.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(CAT_SHADER, "main");
      dispatchCompute(pipeline, [a.buffer, b.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
      await syncDevice();
      paramsBuffer.destroy();
      const meta = this.registerTensor(out, outShape, a.dtype, outLength);
      return cloneHandle(meta);
    }
  }

  async expand(tensorId: number, shape: number[]): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
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
    const out = this.createStorageBuffer(Math.max(4, outLength * 4));
    const srcStrides = this.computeStrides(paddedSourceShape);
    const padded = this.padShapeTo4(paddedSourceShape);
    const outPadded = this.padShapeTo4(shape);
    const stridesPadded = this.padShapeTo4(srcStrides.map((s) => paddedSourceShape[s]!));
    // Actually strides: if a dim is 1, its stride should be 0 for broadcast
    const broadcastStrides = paddedSourceShape.map((s, i) => (s === 1 ? 0 : srcStrides[i]!));
    const bsPadded = this.padShapeTo4(broadcastStrides);
    // Use expand shader: input shape and strides
    const paramsData = new Uint32Array([
      padded[0], padded[1], padded[2], padded[3],
      bsPadded[0], bsPadded[1], bsPadded[2], bsPadded[3],
    ]);
    const paramsBuffer = this.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(EXPAND_SHADER, "main");
    dispatchCompute(pipeline, [source.buffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    const meta = this.registerTensor(out, shape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  async indexSelect(tensorId: number, dim: number, indicesId: number): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const indices = this.getTensor(indicesId);
    if (source.shape.length > 2) {
      throw new Error("indexSelect currently supports up to 2D input tensors.");
    }
    if (indices.shape.length !== 1) {
      throw new Error("indexSelect requires 1D indices tensor.");
    }
    const resolvedDim = this.normalizeDim(dim, source.shape.length);
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

    // Read indices from GPU and recreate as Int32 buffer for the shader
    const indexValues = await this.toList(indicesId);
    const indicesInt32 = new Int32Array(indexValues.map((v) => Math.trunc(v)));
    const indicesBuffer = this.device!.createBuffer({
      size: indicesInt32.byteLength,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(indicesBuffer, 0, indicesInt32);

    const out = this.createStorageBuffer(Math.max(4, outLength * 4));
    const paramsData = new Uint32Array([
      resolvedDim,
      source.shape.length >= 1 ? source.shape[0]! : 1,
      source.shape.length >= 2 ? source.shape[1]! : 1,
      numIndices,
    ]);
    const paramsBuffer = this.device!.createBuffer({
      size: paramsData.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const shader = source.shape.length === 1 ? "index_select_1d" : "index_select_2d";
    const pipeline = getOrCreatePipeline(INDEX_SELECT_SHADER, shader);
    dispatchCompute(pipeline, [source.buffer, indicesBuffer, out, paramsBuffer], calculateWorkgroups(outLength));
    await syncDevice();
    paramsBuffer.destroy();
    indicesBuffer.destroy();
    const meta = this.registerTensor(out, outShape, source.dtype, outLength);
    return cloneHandle(meta);
  }

  isAvailable(): boolean {
    return Boolean(globalThis.navigator?.gpu);
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  deviceCount(): number {
    return this.isAvailable() ? 1 : 0;
  }

  async currentDevice(): Promise<number> {
    await this.ensureReady();
    return 0;
  }

  async getDeviceName(deviceIndex?: number): Promise<string> {
    this.assertDeviceIndex(deviceIndex);
    await this.ensureReady();
    const properties = this.collectDeviceProperties();
    return properties.name as string;
  }

  async getDeviceProperties(deviceIndex?: number): Promise<Record<string, unknown>> {
    this.assertDeviceIndex(deviceIndex);
    await this.ensureReady();
    return this.collectDeviceProperties();
  }

  async memoryAllocated(deviceIndex?: number): Promise<number> {
    this.assertDeviceIndex(deviceIndex);
    await this.ensureReady();
    return this.currentAllocatedBytes;
  }

  async memoryReserved(deviceIndex?: number): Promise<number> {
    this.assertDeviceIndex(deviceIndex);
    await this.ensureReady();
    return this.currentAllocatedBytes;
  }

  private async fill(shape: number[], dtype: string, value: number): Promise<TensorHandle> {
    await this.ensureReady();
    this.assertDType(dtype);
    const length = product(shape);
    const out = this.createStorageBuffer(Math.max(4, length * 4));
    const params = new ArrayBuffer(8);
    const view = new DataView(params);
    view.setFloat32(0, this.coerceScalarByDType(value, dtype as SupportedDType), true);
    view.setUint32(4, length, true);
    const paramBuffer = this.device!.createBuffer({
      size: 8,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramBuffer, 0, params);

    const pipeline = getOrCreatePipeline(FILL_SHADER, "fill");
    dispatchCompute(pipeline, [out, paramBuffer], calculateWorkgroups(length));
    await syncDevice();
    paramBuffer.destroy();

    const meta = this.registerTensor(out, shape, dtype, length);
    return cloneHandle(meta);
  }

  private async elementwise(aId: number, bId: number, op: "add" | "mul" | "sub" | "div_op"): Promise<TensorHandle> {
    await this.ensureReady();
    const a = this.getTensor(aId);
    const b = this.getTensor(bId);
    if (a.length !== b.length) {
      throw new Error(`Shape mismatch for ${op}: ${a.length} != ${b.length}.`);
    }
    if (a.shape.join(",") !== b.shape.join(",")) {
      throw new Error(`Shape mismatch for ${op}: ${a.shape} vs ${b.shape}.`);
    }
    const out = this.createStorageBuffer(Math.max(4, a.length * 4));
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, op);
    dispatchCompute(pipeline, [a.buffer, b.buffer, out], calculateWorkgroups(a.length));
    await syncDevice();
    const meta = this.registerTensor(out, a.shape, a.dtype, a.length);
    return cloneHandle(meta);
  }

  private async reduce(tensorId: number, asMean: boolean): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    let currentBuffer = source.buffer;
    let currentLength = source.length;
    let temporaryToDestroy: GPUBuffer[] = [];

    while (currentLength > 1) {
      const groups = Math.ceil(currentLength / 256);
      const outBuffer = this.createStorageBuffer(Math.max(4, groups * 4));
      const paramData = new Uint32Array([currentLength]);
      const paramBuffer = this.device!.createBuffer({
        size: 4,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
      });
      this.device!.queue.writeBuffer(paramBuffer, 0, paramData);

      const pipeline = getOrCreatePipeline(REDUCE_SUM_SHADER, "main");
      dispatchCompute(pipeline, [currentBuffer, outBuffer, paramBuffer], [groups, 1, 1]);
      await syncDevice();
      paramBuffer.destroy();

      if (currentBuffer !== source.buffer) {
        temporaryToDestroy.push(currentBuffer);
      }
      currentBuffer = outBuffer;
      currentLength = groups;
    }

    let scalarValue = await this.readScalar(currentBuffer);
    if (asMean) {
      scalarValue /= source.length;
    }

    if (currentBuffer !== source.buffer) {
      temporaryToDestroy.push(currentBuffer);
    }
    for (const tmp of temporaryToDestroy) {
      tmp.destroy();
    }

    const outBuffer = this.createStorageBuffer(4);
    this.device!.queue.writeBuffer(outBuffer, 0, new Float32Array([scalarValue]));
    const meta = this.registerTensor(outBuffer, [], source.dtype, 1);
    return cloneHandle(meta);
  }

  private async argReduce(tensorId: number, asMax: boolean): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    if (source.shape.length === 0) {
      const outScalar = this.createStorageBuffer(4);
      this.device!.queue.writeBuffer(outScalar, 0, new Int32Array([0]));
      const meta = this.registerTensor(outScalar, [], "int32", 1);
      return cloneHandle(meta);
    }
    const lastDim = source.shape[source.shape.length - 1]!;
    if (lastDim <= 0) {
      throw new Error("argmax/argmin require last dimension > 0.");
    }
    const batchSize = source.length / lastDim;
    const outputShape = source.shape.length === 1 ? [] : source.shape.slice(0, -1);
    const out = this.createStorageBuffer(Math.max(4, batchSize * 4));
    const dims = new Uint32Array([batchSize, lastDim]);
    const dimsBuffer = this.device!.createBuffer({
      size: dims.byteLength,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(dimsBuffer, 0, dims);
    const shader = asMax ? ARGMAX_SHADER : ARGMIN_SHADER;
    const entry = asMax ? "argmax" : "argmin";
    const pipeline = getOrCreatePipeline(shader, entry);
    dispatchCompute(pipeline, [source.buffer, out, dimsBuffer], [batchSize, 1, 1]);
    await syncDevice();
    dimsBuffer.destroy();
    const meta = this.registerTensor(out, outputShape, "int32", batchSize);
    return cloneHandle(meta);
  }

  private async readScalar(buffer: GPUBuffer): Promise<number> {
    const readBuffer = this.device!.createBuffer({
      size: 4,
      usage: BufferUsage.COPY_DST | BufferUsage.MAP_READ
    });
    const encoder = this.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, 4);
    this.device!.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(MapMode.READ);
    const view = new Float32Array(readBuffer.getMappedRange().slice(0));
    const value = view[0];
    readBuffer.unmap();
    readBuffer.destroy();
    return value;
  }

  private registerTensor(buffer: GPUBuffer, shape: number[], dtype: string, length: number): TensorMeta {
    const id = this.nextId++;
    const bytes = buffer.size;
    const meta: TensorMeta = {
      id,
      buffer,
      shape: [...shape],
      dtype,
      length,
      bytes
    };
    this.tensors.set(id, meta);
    this.currentAllocatedBytes += bytes;
    return meta;
  }

  private getTensor(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) {
      throw new Error(`Unknown tensor id: ${id}.`);
    }
    return meta;
  }

  private createStorageBuffer(size: number): GPUBuffer {
    return this.device!.createBuffer({
      size,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST
    });
  }

  private async ensureReady(gpuProvider?: GPU | null): Promise<void> {
    if (this.initialized && this.device && this.adapter) {
      return;
    }
    if (this.initPromise) {
      await this.initPromise;
      return;
    }
    this.initPromise = this.initializeInternal(gpuProvider);
    try {
      await this.initPromise;
    } finally {
      this.initPromise = null;
    }
  }

  private async initializeInternal(gpuProvider?: GPU | null): Promise<void> {
    this.initError = null;
    if (gpuProvider === null) {
      this.initialized = false;
      this.initError = "WebGPU unavailable in this browser.";
      throw new Error(this.initError);
    }
    await initWebGPU(gpuProvider ?? undefined);
    this.device = getDevice() as GPUDevice;
    this.adapter = getAdapter() as GPUAdapter;
    this.initialized = isWebGPUInitialized();
  }

  private assertDeviceIndex(deviceIndex?: number): void {
    if (deviceIndex === undefined || deviceIndex === null) {
      return;
    }
    if (deviceIndex !== 0) {
      throw new Error(`Only device index 0 is supported in MVP, received: ${deviceIndex}.`);
    }
  }

  private collectDeviceProperties(): Record<string, unknown> {
    const adapterAny = this.adapter as unknown as {
      info?: {
        vendor?: string;
        architecture?: string;
        device?: string;
        description?: string;
      };
      isFallbackAdapter?: boolean;
    };
    const limits = this.adapter!.limits as unknown as Record<string, number>;
    const info = adapterAny.info ?? {};
    const name =
      info.description ||
      info.device ||
      info.architecture ||
      info.vendor ||
      "WebGPU Adapter";
    return {
      name,
      total_memory: 0,
      major: 0,
      minor: 0,
      multi_processor_count: 0,
      vendor: info.vendor ?? "",
      architecture: info.architecture ?? "",
      description: info.description ?? "",
      device: info.device ?? "",
      is_fallback_adapter: Boolean(adapterAny.isFallbackAdapter),
      subgroup_min_size: limits.minSubgroupSize ?? 0,
      subgroup_max_size: limits.maxSubgroupSize ?? 0,
      limits
    };
  }

  private assertDType(dtype: string) {
    if (dtype !== "float32" && dtype !== "int32" && dtype !== "bool") {
      throw new Error(`Unsupported dtype: ${dtype}. Supported dtypes: float32, int32, bool.`);
    }
  }

  private coerceScalarByDType(value: number, dtype: SupportedDType): number {
    if (dtype === "bool") {
      return value ? 1 : 0;
    }
    if (dtype === "int32") {
      return Math.trunc(value);
    }
    return value;
  }

  private decodeValuesByDType(buffer: ArrayBuffer, dtype: SupportedDType): number[] {
    if (dtype === "int32") {
      return Array.from(new Int32Array(buffer));
    }
    if (dtype === "bool") {
      return Array.from(new Float32Array(buffer)).map((value) => (value !== 0 ? 1 : 0));
    }
    return Array.from(new Float32Array(buffer));
  }

  private assertUnaryDType(dtype: string, op: "relu" | "sqrt" | "exp" | "log"): void {
    if (dtype !== "float32") {
      throw new Error(`${op} currently supports only float32 tensors, received: ${dtype}.`);
    }
  }

  private normalizeDim(dim: number, rank: number): number {
    if (rank === 0) {
      throw new Error("operation requires at least 1 dimension.");
    }
    const resolved = dim < 0 ? dim + rank : dim;
    if (resolved < 0 || resolved >= rank) {
      throw new Error(`dim out of range for rank ${rank}: ${dim}.`);
    }
    return resolved;
  }

  private computeStrides(shape: number[]): number[] {
    if (shape.length === 0) {
      return [];
    }
    const strides = new Array<number>(shape.length);
    let running = 1;
    for (let i = shape.length - 1; i >= 0; i -= 1) {
      strides[i] = running;
      running *= shape[i]!;
    }
    return strides;
  }

  private linearToCoords(index: number, shape: number[], strides: number[]): number[] {
    const coords = new Array<number>(shape.length);
    let remaining = index;
    for (let i = 0; i < shape.length; i += 1) {
      const stride = strides[i]!;
      coords[i] = Math.floor(remaining / stride);
      remaining %= stride;
    }
    return coords;
  }

  private coordsToLinear(coords: number[], strides: number[]): number {
    let out = 0;
    for (let i = 0; i < coords.length; i += 1) {
      out += coords[i]! * strides[i]!;
    }
    return out;
  }

  private normalizeSliceStart(start: number | undefined, size: number, step: number): number {
    if (start === undefined) {
      return step > 0 ? 0 : size - 1;
    }
    let value = start < 0 ? start + size : start;
    if (step > 0) {
      value = Math.max(0, Math.min(size, value));
    } else {
      value = Math.max(-1, Math.min(size - 1, value));
    }
    return value;
  }

  private padShapeTo4(shape: number[]): [number, number, number, number] {
    if (shape.length === 0) return [1, 1, 1, 1];
    if (shape.length === 1) return [1, 1, 1, shape[0]!] as [number, number, number, number];
    if (shape.length === 2) return [1, 1, shape[0]!, shape[1]!] as [number, number, number, number];
    if (shape.length === 3) return [1, shape[0]!, shape[1]!, shape[2]!] as [number, number, number, number];
    return shape as [number, number, number, number];
  }

  private normalizeSliceEnd(end: number | undefined, size: number, step: number): number {
    if (end === undefined) {
      return step > 0 ? size : -1;
    }
    let value = end < 0 ? end + size : end;
    if (step > 0) {
      value = Math.max(0, Math.min(size, value));
    } else {
      value = Math.max(-1, Math.min(size - 1, value));
    }
    return value;
  }

}

export function installTorchRuntime(target: typeof globalThis = globalThis): TorchPyodideRuntime {
  const runtime = new TorchPyodideRuntime();
  (target as typeof globalThis & { __torch_pyodide_runtime__?: TorchPyodideRuntime }).__torch_pyodide_runtime__ =
    runtime;
  return runtime;
}
