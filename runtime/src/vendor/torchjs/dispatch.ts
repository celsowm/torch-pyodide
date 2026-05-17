/**
 * Shader dispatch utilities for WebGPU compute operations.
 * @status implemented
 */

import { getDevice } from './device.js';
import { WebGPUComputePipeline, WebGPUBuffer } from './types.js';

// Pipeline cache to avoid recompilation
const pipelineCache = new Map<string, WebGPUComputePipeline>();

/**
 * Get or create a compute pipeline for the given shader.
 */
export function getOrCreatePipeline(
  shaderCode: string,
  entryPoint: string = 'main'
): WebGPUComputePipeline {
  const cacheKey = `${shaderCode}:${entryPoint}`;

  let pipeline = pipelineCache.get(cacheKey);
  if (pipeline) {
    return pipeline;
  }

  const device = getDevice();

  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  // Check for shader compilation errors (async but we log immediately)
  // Note: getCompilationInfo is not available in wgpu-native
  if (typeof shaderModule.getCompilationInfo === 'function') {
    shaderModule.getCompilationInfo().then((info: GPUCompilationInfo) => {
      for (const msg of info.messages) {
        const level = msg.type === 'error' ? 'error' : msg.type === 'warning' ? 'warn' : 'log';
        console[level](`[WGSL ${entryPoint}] ${msg.type}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
      }
    });
  }

  pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint,
    },
  });

  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

/**
 * Dispatch a compute shader with the given buffers.
 */
export function dispatchCompute(
  pipeline: WebGPUComputePipeline,
  buffers: WebGPUBuffer[],
  workgroupCount: [number, number, number]
): void {
  const device = getDevice();

  // Create bind group entries - must include size for wgpu-native compatibility
  const bindGroupEntries = buffers.map((buffer, index) => ({
    binding: index,
    resource: { buffer, offset: 0, size: buffer.size },
  }));

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindGroupEntries,
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
