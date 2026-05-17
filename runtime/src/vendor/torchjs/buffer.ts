/**
 * GPU buffer management with pooling for performance.
 * @status implemented
 */

import { getDevice } from './device.js';
import { DType, getDTypeBytes, TypedArray } from '../../dtype.js';
import { WebGPUBuffer, BufferUsage, MapMode } from './types.js';

export interface MemoryStats {
  activeBytes: number;
  pooledBytes: number;
  peakBytes: number;
  allocationCount: number;
}

/**
 * Buffer pool for reusing GPU buffers of the same size.
 * Reduces allocation overhead during training.
 */
class GPUBufferPool {
  private pools: Map<number, WebGPUBuffer[]> = new Map();
  private maxPoolSize = 32;
  
  // Tracking
  private activeBytes = 0;
  private pooledBytes = 0;
  private peakBytes = 0;
  private totalAllocations = 0;

  /**
   * Acquire a buffer of the given size in bytes.
   * Returns a pooled buffer if available, otherwise creates a new one.
   */
  acquire(sizeBytes: number, usage: number): WebGPUBuffer {
    const device = getDevice();

    // Align to 4 bytes (WebGPU requirement for storage buffers)
    const alignedSize = Math.max(4, Math.ceil(sizeBytes / 4) * 4);

    // Check pool for existing buffer
    const pool = this.pools.get(alignedSize);
    if (pool && pool.length > 0) {
      const buffer = pool.pop()!;
      this.pooledBytes -= alignedSize;
      this.activeBytes += alignedSize;
      this.updatePeak();
      return buffer;
    }

    // Create new buffer
    const buffer = device.createBuffer({
      size: alignedSize,
      usage: usage | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
    });
    
    this.activeBytes += alignedSize;
    this.totalAllocations++;
    this.updatePeak();
    
    return buffer;
  }

  /**
   * Release a buffer back to the pool for reuse.
   */
  release(buffer: WebGPUBuffer): void {
    const size = buffer.size;
    this.activeBytes -= size;
    
    let pool = this.pools.get(size);
    if (!pool) {
      pool = [];
      this.pools.set(size, pool);
    }

    if (pool.length < this.maxPoolSize) {
      pool.push(buffer);
      this.pooledBytes += size;
    } else {
      buffer.destroy();
    }
  }

  /**
   * Clear all pooled buffers.
   */
  clear(): void {
    for (const pool of this.pools.values()) {
      for (const buffer of pool) {
        buffer.destroy();
      }
    }
    this.pools.clear();
    this.pooledBytes = 0;
  }

  getStats(): MemoryStats {
    return {
      activeBytes: this.activeBytes,
      pooledBytes: this.pooledBytes,
      peakBytes: this.peakBytes,
      allocationCount: this.totalAllocations
    };
  }

  resetPeak(): void {
    this.peakBytes = this.activeBytes;
  }

  private updatePeak(): void {
    this.peakBytes = Math.max(this.peakBytes, this.activeBytes);
  }
}

export const bufferPool = new GPUBufferPool();

/**
 * Create a storage buffer for compute operations.
 */
export function createStorageBuffer(sizeBytes: number): WebGPUBuffer {
  return bufferPool.acquire(sizeBytes, BufferUsage.STORAGE);
}

/**
 * Create a buffer and upload data to it.
 */
export function createBufferWithData(data: TypedArray, _dtype: DType): WebGPUBuffer {
  const sizeBytes = data.byteLength;
  // WebGPU writeBuffer requires size to be a multiple of 4 bytes
  const alignedSize = Math.max(4, Math.ceil(sizeBytes / 4) * 4);
  const buffer = bufferPool.acquire(alignedSize, BufferUsage.STORAGE);

  // If data size isn't aligned, create a padded copy
  const device = getDevice();
  if (alignedSize > sizeBytes) {
    const padded = new Uint8Array(alignedSize);
    padded.set(new Uint8Array(data.buffer, data.byteOffset, sizeBytes));
    device.queue.writeBuffer(buffer, 0, padded);
  } else {
    device.queue.writeBuffer(buffer, 0, data);
  }

  return buffer;
}

/**
 * Read data from a GPU buffer back to CPU.
 */
export async function readBuffer(buffer: WebGPUBuffer, dtype: DType, numElements: number): Promise<TypedArray> {
  const device = getDevice();
  const bytesPerElement = getDTypeBytes(dtype);
  const sizeBytes = numElements * bytesPerElement;

  // Align to 4 bytes, minimum size of 4 bytes
  const alignedSize = Math.max(4, Math.ceil(sizeBytes / 4) * 4);

  // Create staging buffer for readback (not pooled, destroyed immediately)
  const stagingBuffer = device.createBuffer({
    size: alignedSize,
    usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST,
  });

  // Copy from storage to staging
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, alignedSize);
  device.queue.submit([commandEncoder.finish()]);

  // Wait for copy to complete and map
  await stagingBuffer.mapAsync(MapMode.READ);
  const mappedRange = stagingBuffer.getMappedRange();

  // Create appropriate typed array and copy data
  let result: TypedArray;
  switch (dtype) {
    case 'float32':
    case 'float16':
      result = new Float32Array(numElements);
      result.set(new Float32Array(mappedRange, 0, numElements));
      break;
    case 'int32':
      result = new Int32Array(numElements);
      result.set(new Int32Array(mappedRange, 0, numElements));
      break;
    case 'uint32':
      result = new Uint32Array(numElements);
      result.set(new Uint32Array(mappedRange, 0, numElements));
      break;
    case 'int8':
      result = new Int8Array(numElements);
      result.set(new Int8Array(mappedRange, 0, numElements));
      break;
    case 'uint8':
    case 'bool':
      result = new Uint8Array(numElements);
      result.set(new Uint8Array(mappedRange, 0, numElements));
      break;
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }

  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return result;
}