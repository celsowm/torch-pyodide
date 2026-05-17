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
};

export type SupportedDType = "float32" | "int32" | "bool";

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
