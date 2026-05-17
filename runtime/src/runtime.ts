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
  TRANSPOSE_SHADER
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
    const values = await this.toList(tensorId);
    const inStrides = this.computeStrides(source.shape);
    const outShape = normalized.map((axis) => source.shape[axis]!);
    const outStrides = this.computeStrides(outShape);
    const out = new Array<number>(source.length);
    for (let outIdx = 0; outIdx < source.length; outIdx += 1) {
      const outCoords = this.linearToCoords(outIdx, outShape, outStrides);
      const inCoords = new Array<number>(rank);
      for (let i = 0; i < rank; i += 1) {
        inCoords[normalized[i]!] = outCoords[i]!;
      }
      const inIdx = this.coordsToLinear(inCoords, inStrides);
      out[outIdx] = values[inIdx]!;
    }
    return this.tensorFromData(out, outShape, source.dtype);
  }

  async select(tensorId: number, dim: number, index: number): Promise<TensorHandle> {
    await this.ensureReady();
    const source = this.getTensor(tensorId);
    const rank = source.shape.length;
    if (rank === 0) {
      throw new Error("select is not supported for scalar tensors.");
    }
    const resolvedDim = this.normalizeDim(dim, rank);
    const axisSize = source.shape[resolvedDim]!;
    const resolvedIndex = index < 0 ? index + axisSize : index;
    if (resolvedIndex < 0 || resolvedIndex >= axisSize) {
      throw new Error(`select index out of range for dim ${dim}: ${index}.`);
    }
    const values = await this.toList(tensorId);
    const inShape = source.shape;
    const inStrides = this.computeStrides(inShape);
    const outShape = inShape.slice(0, resolvedDim).concat(inShape.slice(resolvedDim + 1));
    const outLength = Math.max(1, product(outShape));
    const out = new Array<number>(outLength);
    const outStrides = this.computeStrides(outShape);
    for (let outIdx = 0; outIdx < outLength; outIdx += 1) {
      const outCoords = outShape.length === 0 ? [] : this.linearToCoords(outIdx, outShape, outStrides);
      const inCoords = [...outCoords.slice(0, resolvedDim), resolvedIndex, ...outCoords.slice(resolvedDim)];
      out[outIdx] = values[this.coordsToLinear(inCoords, inStrides)]!;
    }
    return this.tensorFromData(out, outShape, source.dtype);
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
    const resolvedDim = this.normalizeDim(dim, rank);
    const axisSize = source.shape[resolvedDim]!;
    const normalizedStart = this.normalizeSliceStart(start, axisSize, step);
    const normalizedEnd = this.normalizeSliceEnd(end, axisSize, step);
    const axisIndices: number[] = [];
    if (step > 0) {
      for (let i = normalizedStart; i < normalizedEnd; i += step) axisIndices.push(i);
    } else {
      for (let i = normalizedStart; i > normalizedEnd; i += step) axisIndices.push(i);
    }
    const outShape = [...source.shape];
    outShape[resolvedDim] = axisIndices.length;
    const outLength = product(outShape);
    const out = new Array<number>(outLength);
    const values = await this.toList(tensorId);
    const inStrides = this.computeStrides(source.shape);
    const outStrides = this.computeStrides(outShape);
    for (let outIdx = 0; outIdx < outLength; outIdx += 1) {
      const outCoords = this.linearToCoords(outIdx, outShape, outStrides);
      const inCoords = [...outCoords];
      inCoords[resolvedDim] = axisIndices[outCoords[resolvedDim]!]!;
      out[outIdx] = values[this.coordsToLinear(inCoords, inStrides)]!;
    }
    return this.tensorFromData(out, outShape, source.dtype);
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
