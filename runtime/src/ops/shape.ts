import { TensorMeta, SupportedDType, product } from "./types.js";

export function assertDType(dtype: string) {
  if (dtype !== "float32" && dtype !== "int32" && dtype !== "bool") {
    throw new Error(`Unsupported dtype: ${dtype}. Supported dtypes: float32, int32, bool.`);
  }
}

export function coerceScalarByDType(value: number, dtype: SupportedDType): number {
  if (dtype === "bool") return value ? 1 : 0;
  if (dtype === "int32") return Math.trunc(value);
  return value;
}

export function decodeValuesByDType(buffer: ArrayBuffer, dtype: SupportedDType): number[] {
  if (dtype === "int32") return Array.from(new Int32Array(buffer));
  if (dtype === "bool") return Array.from(new Float32Array(buffer)).map((v) => (v !== 0 ? 1 : 0));
  return Array.from(new Float32Array(buffer));
}

export function assertUnaryDType(dtype: string, _op: string): void {
  if (dtype !== "float32") throw new Error(`${_op} currently supports only float32 tensors, received: ${dtype}.`);
}

export function normalizeDim(dim: number, rank: number): number {
  if (rank === 0) throw new Error("operation requires at least 1 dimension.");
  const resolved = dim < 0 ? dim + rank : dim;
  if (resolved < 0 || resolved >= rank) throw new Error(`dim out of range for rank ${rank}: ${dim}.`);
  return resolved;
}

export function computeStrides(shape: number[]): number[] {
  if (shape.length === 0) return [];
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
  if (start === undefined) return step > 0 ? 0 : size - 1;
  let value = start < 0 ? start + size : start;
  if (step > 0) value = Math.max(0, Math.min(size, value));
  else value = Math.max(-1, Math.min(size - 1, value));
  return value;
}

export function normalizeSliceEnd(end: number | undefined, size: number, step: number): number {
  if (end === undefined) return step > 0 ? size : -1;
  let value = end < 0 ? end + size : end;
  if (step > 0) value = Math.max(0, Math.min(size, value));
  else value = Math.max(-1, Math.min(size - 1, value));
  return value;
}

export function padShapeTo4(shape: number[]): [number, number, number, number] {
  if (shape.length === 0) return [1, 1, 1, 1];
  if (shape.length === 1) return [1, 1, 1, shape[0]!] as [number, number, number, number];
  if (shape.length === 2) return [1, 1, shape[0]!, shape[1]!] as [number, number, number, number];
  if (shape.length === 3) return [1, shape[0]!, shape[1]!, shape[2]!] as [number, number, number, number];
  return shape as [number, number, number, number];
}

export function broadcastShapes(a: number[], b: number[]): number[] {
  const maxRank = Math.max(a.length, b.length);
  const result: number[] = new Array(maxRank);
  for (let i = 0; i < maxRank; i++) {
    const aDim = a.length - maxRank + i >= 0 ? a[a.length - maxRank + i]! : 1;
    const bDim = b.length - maxRank + i >= 0 ? b[b.length - maxRank + i]! : 1;
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`Shapes [${a}] and [${b}] cannot be broadcast together.`);
    }
    result[i] = Math.max(aDim, bDim);
  }
  return result;
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
  if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
  return meta;
}
