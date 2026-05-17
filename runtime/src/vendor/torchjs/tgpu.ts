/**
 * TypeGPU integration for type-safe GPU operations.
 * Provides typed buffer management and shader template resolution.
 * @status implemented
 */

import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import type { Infer } from 'typegpu/data';

// Re-export typegpu/data for convenience
export { d };

// TypeGPU root instance
let root: TgpuRoot | null = null;

/**
 * Initialize TypeGPU with an existing device or create a new one.
 */
export async function initTypegpu(existingDevice?: GPUDevice): Promise<TgpuRoot> {
  if (root) return root;

  if (existingDevice) {
    root = tgpu.initFromDevice({ device: existingDevice });
  } else {
    root = await tgpu.init();
  }

  return root;
}

/**
 * Get the TypeGPU root instance.
 */
export function getRoot(): TgpuRoot {
  if (!root) {
    throw new Error('TypeGPU not initialized. Call torch.init() first.');
  }
  return root;
}

/**
 * Check if TypeGPU is initialized.
 */
export function isTypegpuInitialized(): boolean {
  return root !== null;
}

/**
 * Resolve a WGSL shader template with external type definitions.
 */
export function resolveShader(
  template: string,
  externals: Record<string, any> = {}
): string {
  return tgpu.resolve({ template, externals });
}

// Common tensor data schemas
export const TensorSchemas = {
  // Float32 array for tensor data
  f32Array: (maxSize: number) => d.arrayOf(d.f32, maxSize),

  // Int32 array for indices
  i32Array: (maxSize: number) => d.arrayOf(d.i32, maxSize),

  // Uint32 array for shapes/strides
  u32Array: (maxSize: number) => d.arrayOf(d.u32, maxSize),

  // Uniform params for elementwise ops
  ElementwiseParams: d.struct({
    size: d.u32,
    _pad1: d.u32,
    _pad2: d.u32,
    _pad3: d.u32,
  }),

  // Uniform params for scalar ops
  ScalarParams: d.struct({
    scalar: d.f32,
    size: d.u32,
    _pad1: d.u32,
    _pad2: d.u32,
  }),

  // Uniform params for matmul
  MatmulParams: d.struct({
    M: d.u32,
    K: d.u32,
    N: d.u32,
    _pad: d.u32,
  }),

  // Uniform params for reduction
  ReduceParams: d.struct({
    inputSize: d.u32,
    _pad1: d.u32,
    _pad2: d.u32,
    _pad3: d.u32,
  }),
};

/**
 * Create a typed storage buffer using TypeGPU.
 */
export function createTypedBuffer<T>(
  schema: d.AnyData,
  initialData?: T
) {
  const r = getRoot();
  const buffer = r.createBuffer(schema).$usage('storage');

  if (initialData !== undefined) {
    buffer.write(initialData as never);
  }

  return buffer;
}

/**
 * Create a bind group layout for compute operations.
 */
export function createComputeLayout(
  bindings: Record<string, { storage: d.AnyData; access: 'readonly' | 'mutable' }>
) {
  return tgpu.bindGroupLayout(bindings as any);
}

/**
 * Write struct data to a GPU buffer using a TypeGPU schema.
 * Handles alignment and padding automatically.
 * @example
 * const buffer = writeUniform(device, Schemas.FillParams, { value: 1.0, length: 100 });
 */
export function writeUniform<T extends d.AnyData>(
  schema: T,
  data: Infer<T>
): GPUBuffer {
  const r = getRoot();
  const buffer = r.createBuffer(schema as any).$usage('uniform');
  buffer.write(data as never);
  return buffer.buffer;
}

/**
 * Example schemas for common operations.
 * These can be used with resolveShader() to inject struct definitions.
 */
export const Schemas = {
  // Simple fill operation params
  FillParams: d.struct({
    value: d.f32,
    length: d.u32,
  }),

  // RNG params
  RngParams: d.struct({
    seed: d.u32,
    length: d.u32,
  }),

  // 2D dimension params (batch_size, num_classes)
  Dims2D: d.struct({
    dim0: d.u32,
    dim1: d.u32,
  }),
};

/**
 * Generate WGSL struct definition from a TypeGPU schema.
 * Useful for documentation or debugging.
 */
export function getStructWgsl(name: string, schema: d.AnyData): string {
  return tgpu.resolve({
    template: name,
    externals: { [name]: schema },
  });
}
