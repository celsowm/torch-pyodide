/**
 * Shader dispatch utilities for WebGPU compute operations.
 * @status implemented
 */

import { getDevice } from './device.js';
import { WebGPUComputePipeline, WebGPUBuffer } from './types.js';

// GPUShaderStage is a global in the browser; declare it for TS clarity.
declare const GPUShaderStage: { COMPUTE: number; VERTEX: number; FRAGMENT: number };

// Pipeline cache to avoid recompilation
const pipelineCache = new Map<string, WebGPUComputePipeline>();

/**
 * Get or create a compute pipeline for the given shader.
 * If `bindGroupLayouts` is provided, the pipeline uses an explicit layout
 * (bypasses WGSL auto-inference of binding layout — useful when the WGSL
 * parser is non-deterministic or drops bindings).
 */
export async function getOrCreatePipeline(
  shaderCode: string,
  entryPoint: string = 'main',
  bindGroupLayouts?: GPUBindGroupLayout[]
): Promise<WebGPUComputePipeline> {
  const cacheKey = bindGroupLayouts
    ? `${shaderCode}:${entryPoint}:explicit:${bindGroupLayouts.length}`
    : `${shaderCode}:${entryPoint}`;

  let pipeline = pipelineCache.get(cacheKey);
  if (pipeline) {
    return pipeline;
  }

  const device = getDevice();

  const target = globalThis as typeof globalThis & { __WGSL_WARN__?: boolean };
  if (target.__WGSL_WARN__) {
    console.log(`[dispatch] creating new pipeline, key length=${cacheKey.length}, shader length=${shaderCode.length}, explicit=${!!bindGroupLayouts}`);
  }

  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  // AWAIT compilation to avoid race conditions where the pipeline is
  // created before the WGSL parser finishes analyzing the bindings.
  if (typeof shaderModule.getCompilationInfo === 'function') {
    const info = await shaderModule.getCompilationInfo();
    for (const msg of info.messages) {
      const level = msg.type === 'error' ? 'error' : msg.type === 'warning' ? 'warn' : 'log';
      console[level](`[WGSL ${entryPoint}] ${msg.type}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
    }
  }

  const layout = bindGroupLayouts
    ? device.createPipelineLayout({ bindGroupLayouts })
    : 'auto';

  pipeline = device.createComputePipeline({
    layout,
    compute: {
      module: shaderModule,
      entryPoint,
    },
  });

  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

/**
 * Build a storage buffer entry in a bind group layout.
 */
export function makeStorageReadLayoutEntry(binding: number, minBindingSize?: number): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: 'read-only-storage', minBindingSize },
  };
}

export function makeStorageReadWriteLayoutEntry(binding: number, minBindingSize?: number): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: 'storage', minBindingSize },
  };
}

export function makeUniformLayoutEntry(binding: number, minBindingSize?: number): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: 'uniform', minBindingSize },
  };
}

/**
 * Dispatch a compute shader with the given buffers or explicit bind group entries.
 * If buffers are provided as an array, sequential binding indices (0, 1, 2, ...) are assigned.
 * For non-sequential binding indices, pass explicit GPUBindGroupEntry[].
 */
export function dispatchCompute(
  pipeline: WebGPUComputePipeline,
  buffersOrEntries: WebGPUBuffer[] | GPUBindGroupEntry[],
  workgroupCount: [number, number, number]
): void {
  const device = getDevice();

  const entries: GPUBindGroupEntry[] = isBufferArray(buffersOrEntries)
    ? (buffersOrEntries as WebGPUBuffer[]).map((buffer, index) => ({
        binding: index,
        resource: { buffer, offset: 0, size: buffer.size },
      }))
    : (buffersOrEntries as GPUBindGroupEntry[]);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  // Encode and submit
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();

  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroupCount);
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);
}

/**
 * Calculate workgroup count for a given number of elements.
 * Uses a default workgroup size of 256.
 */
export function calculateWorkgroups(
  numElements: number,
  workgroupSize: number = 256
): [number, number, number] {
  const numWorkgroups = Math.ceil(numElements / workgroupSize);

  // WebGPU limits workgroups per dimension to 65535
  // For very large tensors, we need to use multiple dimensions
  if (numWorkgroups <= 65535) {
    return [numWorkgroups, 1, 1];
  } else if (numWorkgroups <= 65535 * 65535) {
    const x = Math.ceil(Math.sqrt(numWorkgroups));
    const y = Math.ceil(numWorkgroups / x);
    return [x, y, 1];
  } else {
    throw new Error(`Tensor too large: ${numElements} elements exceeds WebGPU limits`);
  }
}

/**
 * Wait for all GPU operations to complete.
 * Useful for timing and synchronization.
 */
export async function syncDevice(): Promise<void> {
  // Performance mode: avoid per-op GPU stalls.
  // Callers that need strict completion already synchronize via mapAsync/readback paths.
  // Set globalThis.__TORCH_PYODIDE_SYNC_EAGER__ = true to force eager sync for debugging.
  const target = globalThis as typeof globalThis & { __TORCH_PYODIDE_SYNC_EAGER__?: boolean };
  if (!target.__TORCH_PYODIDE_SYNC_EAGER__) {
    return;
  }
  const device = getDevice();
  await device.queue.onSubmittedWorkDone();
}

/**
 * Create a bind group entry with proper size for wgpu-native compatibility.
 * IMPORTANT: wgpu-native requires explicit size in bind group entries.
 */
export function bindEntry(binding: number, buffer: WebGPUBuffer): GPUBindGroupEntry {
  return {
    binding,
    resource: { buffer, offset: 0, size: buffer.size },
  };
}

function isBufferArray(value: WebGPUBuffer[] | GPUBindGroupEntry[]): value is WebGPUBuffer[] {
  return value.length === 0 || !('binding' in value[0]);
}
