import { TensorMeta, SupportedDType } from "./types.js";
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
  FILL_SHADER,
  RANDOM_SHADER,
  ELEMENTWISE_SHADER,
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
  INDEX_SELECT_SHADER,
  BROADCAST_SHADER,
  REDUCE_DIM_SHADER,
  COMPARE_SHADER,
  REDUCE_PROD_SHADER,
  REDUCE_MAX_SHADER,
  REDUCE_MIN_SHADER,
  CUMSUM_SHADER,
  CUMPROD_SHADER,
  MASKED_SELECT_SHADER,
  MASKED_FILL_SHADER,
  LEAKY_RELU_SHADER,
} from "../vendor/torchjs/index.js";

export {
  getOrCreatePipeline,
  dispatchCompute,
  calculateWorkgroups,
  syncDevice,
  BufferUsage,
  MapMode,
  FILL_SHADER,
  RANDOM_SHADER,
  ELEMENTWISE_SHADER,
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
  INDEX_SELECT_SHADER,
  BROADCAST_SHADER,
  REDUCE_DIM_SHADER,
  COMPARE_SHADER,
  REDUCE_PROD_SHADER,
  REDUCE_MAX_SHADER,
  REDUCE_MIN_SHADER,
  CUMSUM_SHADER,
  CUMPROD_SHADER,
  MASKED_SELECT_SHADER,
  MASKED_FILL_SHADER,
  LEAKY_RELU_SHADER,
};

export function assertDType(dtype: string) {
  if (dtype !== "float32" && dtype !== "int32" && dtype !== "bool") {
    throw new Error(`Unsupported dtype: ${dtype}. Supported dtypes: float32, int32, bool.`);
  }
}

export function coerceScalarByDType(value: number, dtype: SupportedDType): number {
  if (dtype === "bool") {
    return value ? 1 : 0;
  }
  if (dtype === "int32") {
    return Math.trunc(value);
  }
  return value;
}

export function decodeValuesByDType(buffer: ArrayBuffer, dtype: SupportedDType): number[] {
  if (dtype === "int32") {
    return Array.from(new Int32Array(buffer));
  }
  if (dtype === "bool") {
    return Array.from(new Float32Array(buffer)).map((value) => (value !== 0 ? 1 : 0));
  }
  return Array.from(new Float32Array(buffer));
}

export function assertUnaryDType(dtype: string, _op: string): void {
  if (dtype !== "float32") {
    throw new Error(`${_op} currently supports only float32 tensors, received: ${dtype}.`);
  }
}

export function normalizeDim(dim: number, rank: number): number {
  if (rank === 0) {
    throw new Error("operation requires at least 1 dimension.");
  }
  const resolved = dim < 0 ? dim + rank : dim;
  if (resolved < 0 || resolved >= rank) {
    throw new Error(`dim out of range for rank ${rank}: ${dim}.`);
  }
  return resolved;
}

export function computeStrides(shape: number[]): number[] {
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

export function linearToCoords(index: number, shape: number[], strides: number[]): number[] {
  const coords = new Array<number>(shape.length);
  let remaining = index;
  for (let i = 0; i < shape.length; i += 1) {
    const stride = strides[i]!;
    coords[i] = Math.floor(remaining / stride);
    remaining %= stride;
  }
  return coords;
}

export function coordsToLinear(coords: number[], strides: number[]): number {
  let out = 0;
  for (let i = 0; i < coords.length; i += 1) {
    out += coords[i]! * strides[i]!;
  }
  return out;
}

export function normalizeSliceStart(start: number | undefined, size: number, step: number): number {
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

export function normalizeSliceEnd(end: number | undefined, size: number, step: number): number {
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

export function padShapeTo4(shape: number[]): [number, number, number, number] {
  if (shape.length === 0) return [1, 1, 1, 1];
  if (shape.length === 1) return [1, 1, 1, shape[0]!] as [number, number, number, number];
  if (shape.length === 2) return [1, 1, shape[0]!, shape[1]!] as [number, number, number, number];
  if (shape.length === 3) return [1, shape[0]!, shape[1]!, shape[2]!] as [number, number, number, number];
  return shape as [number, number, number, number];
}

export function createStorageBuffer(device: GPUDevice, size: number): GPUBuffer {
  return device.createBuffer({
    size,
    usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST
  });
}

export function registerTensor(
  tensors: Map<number, TensorMeta>,
  nextId: { current: number },
  allocatedBytes: { current: number },
  buffer: GPUBuffer,
  shape: number[],
  dtype: string,
  length: number
): TensorMeta {
  const id = nextId.current++;
  const bytes = buffer.size;
  const meta: TensorMeta = { id, buffer, shape: [...shape], dtype, length, bytes };
  tensors.set(id, meta);
  allocatedBytes.current += bytes;
  return meta;
}

export function getTensor(tensors: Map<number, TensorMeta>, id: number): TensorMeta {
  const meta = tensors.get(id);
  if (!meta) {
    throw new Error(`Unknown tensor id: ${id}.`);
  }
  return meta;
}

export async function readScalar(device: GPUDevice, buffer: GPUBuffer): Promise<number> {
  const readBuffer = device.createBuffer({
    size: 4,
    usage: BufferUsage.COPY_DST | BufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, 4);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MapMode.READ);
  const view = new Float32Array(readBuffer.getMappedRange().slice(0));
  const value = view[0];
  readBuffer.unmap();
  readBuffer.destroy();
  return value;
}

export async function readFromGPU(device: GPUDevice, source: GPUBuffer, length: number, dtype: SupportedDType): Promise<number[]> {
  const readBuffer = device.createBuffer({
    size: length * 4,
    usage: BufferUsage.COPY_DST | BufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readBuffer, 0, length * 4);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MapMode.READ);
  const copied = readBuffer.getMappedRange();
  const copiedBuffer = copied.slice(0);
  const values = decodeValuesByDType(copiedBuffer, dtype);
  readBuffer.unmap();
  readBuffer.destroy();
  return values;
}
