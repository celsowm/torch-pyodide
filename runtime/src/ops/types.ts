export type TensorMeta = {
  id: number;
  buffer: GPUBuffer;
  shape: number[];
  dtype: string;
  length: number;
  bytes: number;
};

export type TensorHandle = {
  id: number;
  shape: number[];
  dtype: string;
  destroy?: () => void;
};

export type SupportedDType =
  | "float32"
  | "float16"
  | "bfloat16"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "bool";

/** Number of bytes per element for a given dtype */
export function dtypeBytes(dtype: SupportedDType | string): number {
  switch (dtype) {
    case "float32": return 4;
    case "float16": return 2;
    case "bfloat16": return 2;
    case "int8": return 1;
    case "uint8": return 1;
    case "int16": return 2;
    case "uint16": return 2;
    case "int32": return 4;
    case "uint32": return 4;
    case "int64": return 8;
    case "uint64": return 8;
    case "bool": return 1;
    default: return 4;
  }
}

/** Convert a Float32Array to a Uint16Array representing IEEE 754 float16 values */
export function f32ArrayToF16(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    const f32 = new Float32Array([v]);
    const u32 = new Uint32Array(f32.buffer)[0];
    // IEEE 754 float32 -> float16 conversion
    const sign = (u32 >> 31) & 0x1;
    let exp = ((u32 >> 23) & 0xFF) - 127;
    const mantissa = u32 & 0x7FFFFF;
    if (exp >= 16) {
      // Overflow -> infinity
      out[i] = (sign << 15) | 0x7C00;
    } else if (exp <= -15) {
      // Underflow -> 0
      out[i] = sign << 15;
    } else if (exp >= -14 && exp <= 15) {
      // Normal case
      const e = exp + 15;
      const m = mantissa >> 13;
      out[i] = (sign << 15) | (e << 10) | m;
    } else {
      // Subnormal
      out[i] = sign << 15;
    }
  }
  return out;
}

/** Convert a Uint16Array (IEEE 754 float16) back to Float32Array */
export function f16ToF32Array(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i];
    const sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1F;
    let mantissa = h & 0x3FF;
    let u32: number;
    if (exp === 0) {
      u32 = sign << 31;
    } else if (exp === 0x1F) {
      // Inf or NaN
      u32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
      const e = exp - 15 + 127;
      u32 = (sign << 31) | (e << 23) | (mantissa << 13);
    }
    const buf = new ArrayBuffer(4);
    new Uint32Array(buf)[0] = u32 >>> 0;
    out[i] = new Float32Array(buf)[0];
  }
  return out;
}

export function product(values: number[]): number {
  return values.reduce((acc, value) => acc * value, 1);
}

export function cloneHandle(meta: TensorMeta): TensorHandle {
  return {
    id: meta.id,
    shape: [...meta.shape],
    dtype: meta.dtype
  };
}
