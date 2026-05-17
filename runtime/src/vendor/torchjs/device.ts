/**
 * WebGPU device initialization and management.
 * Optimized for browser by default, but allows custom GPU provider (e.g. for Node.js).
 * Uses TypeGPU for type-safe buffer and shader management.
 * @status implemented
 */

import type { TgpuRoot } from 'typegpu';
import type { WebGPUDevice, WebGPUAdapter } from './types.js';
import { detectCapabilities } from './capabilities.js';
import { initTypegpu } from './tgpu.js';

let gpuDevice: WebGPUDevice | null = null;
let gpuAdapter: WebGPUAdapter | null = null;
let tgpuRoot: TgpuRoot | null = null;
let initialized = false;
let initPromise: Promise<void> | null = null;

/**
 * Initialize the WebGPU device.
 * Must be called before any tensor operations.
 * 
 * @param customGpu Optional WebGPU implementation (e.g. from @torchjsorg/wgpu-native)
 * @status implemented
 * @pytorch N/A (torch.js specific)
 */
export async function initWebGPU(customGpu?: GPU): Promise<void> {
  if (initialized) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const gpuProvider = customGpu || (typeof navigator !== 'undefined' ? navigator.gpu : null);

    if (!gpuProvider) {
      throw new Error(
        'WebGPU provider not found. In a browser, WebGPU may not be supported. ' +
        'In Node.js, you must provide a WebGPU implementation (e.g. via @torchjsorg/torch.node.js).'
      );
    }

    const adapter = await gpuProvider.requestAdapter({
      powerPreference: 'high-performance',
    });

    if (!adapter) {
      throw new Error('Failed to get WebGPU adapter');
    }

    gpuAdapter = adapter as unknown as WebGPUAdapter;
    
    // Check for optional features
    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features && adapter.features.has('timestamp-query')) {
      requiredFeatures.push('timestamp-query');
    }

    // Request device with sensible defaults/limits
    gpuDevice = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    }) as unknown as WebGPUDevice;

    // Handle device loss if supported by the provider
    if ('lost' in gpuDevice && (gpuDevice as any).lost) {
      (gpuDevice as any).lost.then((info: { message: string }) => {
        console.error('WebGPU device lost:', info.message);
        initialized = false;
        gpuDevice = null;
        initPromise = null;
      });
    }

    initialized = true;

    // Initialize TypeGPU with our device
    tgpuRoot = await initTypegpu(gpuDevice);

    // Detect GPU capabilities (e.g., workgroup shared memory support)
    await detectCapabilities();
  })();

  return initPromise;
}

/**
 * Get the initialized WebGPU device.
 * Throws if not initialized.
 */
export function getDevice(): WebGPUDevice {
  if (!gpuDevice) {
    throw new Error('WebGPU not initialized. Call torch.init() first.');
  }
  return gpuDevice;
}

/**
 * Get the WebGPU adapter.
 * Throws if not initialized.
 */
export function getAdapter(): WebGPUAdapter {
  if (!gpuAdapter) {
    throw new Error('WebGPU not initialized. Call torch.init() first.');
  }
  return gpuAdapter;
}

/**
 * Get the TypeGPU root instance.
 * Throws if not initialized.
 */
export function getTgpuRoot(): TgpuRoot {
  if (!tgpuRoot) {
    throw new Error('WebGPU not initialized. Call torch.init() first.');
  }
  return tgpuRoot;
}

/**
 * Check if WebGPU is initialized.
 */
export function isInitialized(): boolean {
  return initialized;
}

/**
 * Ensure WebGPU is initialized, initializing if necessary.
 */
export async function ensureInitialized(): Promise<void> {
  if (!initialized) {
    await initWebGPU();
  }
}
