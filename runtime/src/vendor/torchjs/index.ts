/**
 * WebGPU backend exports.
 * @status implemented
 */

export { initWebGPU, getDevice, getAdapter, getTgpuRoot, isInitialized, ensureInitialized } from './device';
export { bufferPool, createStorageBuffer, createBufferWithData, readBuffer } from './buffer';
export { getOrCreatePipeline, dispatchCompute, calculateWorkgroups, syncDevice, bindEntry } from './dispatch';
export { detectCapabilities, getCapabilities } from './capabilities';
export type { GPUCapabilities } from './capabilities';
export { BufferUsage, MapMode } from './types';
export * from './shaders';

// TypeGPU integration
export {
  d,
  getRoot,
  resolveShader,
  TensorSchemas,
  createTypedBuffer,
  createComputeLayout,
  writeUniform,
  Schemas,
  getStructWgsl,
} from './tgpu';
