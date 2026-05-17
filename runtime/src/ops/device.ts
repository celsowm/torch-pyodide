import {
  initWebGPU,
  getDevice,
  getAdapter,
  isInitialized as isWebGPUInitialized,
  BufferUsage,
  MapMode,
} from "../vendor/torchjs/index.js";

const LOST_RECOVERY_RETRIES = 3;
const BACKOFF_BASE_MS = 200;

export async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export async function withRetry<T>(label: string, fn: () => Promise<T>, retries: number): Promise<T> {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === retries - 1) throw err;
      const delay = BACKOFF_BASE_MS * Math.pow(2, attempt);
      console.warn(`[DeviceManager] ${label} — attempt ${attempt + 1}/${retries} failed:`, (err as Error)?.message, `retrying in ${delay}ms`);
      await sleep(delay);
    }
  }
  throw new Error(`${label} — all ${retries} attempts failed`);
}

export interface ShadowBuffer {
  data: Float32Array;
  size: number;
}

export class DeviceManager {
  private _device: GPUDevice | null = null;
  private _adapter: GPUAdapter | null = null;
  private _initialized = false;
  private _initPromise: Promise<void> | null = null;
  private _initError: string | null = null;
  private _lostHandler: ((ev: GPUDeviceLostInfo) => void) | null = null;

  // Shadow copy: keeps CPU-side data for all created storage buffers
  private _shadowBuffers = new Map<number, ShadowBuffer>();
  private _nextShadowId = 1;

  // Callbacks to re-create resources after device recovery
  private _recoveryCallbacks: Array<() => Promise<void>> = [];
  private _pendingBuffers: Array<{ id: number; shadow: ShadowBuffer }> = [];

  get device(): GPUDevice | null {
    return this._device;
  }

  get adapter(): GPUAdapter | null {
    return this._adapter;
  }

  get initialized(): boolean {
    return this._initialized;
  }

  isAvailable(): boolean {
    return Boolean(globalThis.navigator?.gpu);
  }

  isInitialized(): boolean {
    return this._initialized;
  }

  deviceCount(): number {
    return this.isAvailable() ? 1 : 0;
  }

  async currentDevice(): Promise<number> {
    await this.ensureReady();
    return 0;
  }

  async getDeviceName(deviceIndex?: number): Promise<string> {
    this.assertDeviceIndex(deviceIndex);
    await this.ensureReady();
    const properties = this.collectDeviceProperties();
    return properties.name as string;
  }

  async getDeviceProperties(deviceIndex?: number): Promise<Record<string, unknown>> {
    this.assertDeviceIndex(deviceIndex);
    await this.ensureReady();
    return this.collectDeviceProperties();
  }

  async ensureReady(gpuProvider?: GPU | null): Promise<GPUDevice> {
    if (this._initialized && this._device) {
      return this._device;
    }
    if (this._initPromise) {
      await this._initPromise;
      if (this._device) return this._device;
    }
    this._initPromise = this.initializeInternal(gpuProvider);
    try {
      await this._initPromise;
    } finally {
      this._initPromise = null;
    }
    if (!this._device) {
      throw new Error("DeviceManager: failed to initialize GPU device");
    }
    return this._device;
  }

  private async initializeInternal(gpuProvider?: GPU | null): Promise<void> {
    this._initError = null;
    if (gpuProvider === null) {
      this._initialized = false;
      this._initError = "WebGPU unavailable in this browser.";
      throw new Error(this._initError);
    }

    await initWebGPU(gpuProvider ?? undefined);
    this._device = getDevice() as GPUDevice;
    this._adapter = getAdapter() as GPUAdapter;
    this._initialized = isWebGPUInitialized();

    if (!this._device) {
      throw new Error("WebGPU init returned null device");
    }

    this._lostHandler = async (_ev: GPUDeviceLostInfo) => {
      console.warn("[DeviceManager] Device lost detected — attempting recovery...");
      this._device = null;
      this._adapter = null;
      this._initialized = false;
      this._initPromise = null;

      await withRetry("device recovery", async () => {
        await initWebGPU();
        this._device = getDevice() as GPUDevice;
        this._adapter = getAdapter() as GPUAdapter;
        this._initialized = isWebGPUInitialized();

        if (!this._device) {
          throw new Error("Recovery: failed to reinitialize WebGPU");
        }

        if (this._device.lost) {
          this._device.lost.then(this._lostHandler!);
        }

        // Re-create all pending buffers from shadow copy
        const count = this._pendingBuffers.length;
        if (count > 0) {
          console.warn(`[DeviceManager] Re-creating ${count} buffers from shadow copy...`);
          for (const entry of this._pendingBuffers) {
            const buf = this._device!.createBuffer({
              size: entry.shadow.size,
              usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
            });
            this._device!.queue.writeBuffer(buf, 0, entry.shadow.data as unknown as BufferSource);
          }
        }

        // Run all recovery callbacks (pipeline re-creation, etc.)
        for (const cb of this._recoveryCallbacks) {
          await cb();
        }

        console.warn("[DeviceManager] Device recovery complete");
      }, LOST_RECOVERY_RETRIES);
    };

    if (this._device.lost) {
      this._device.lost.then(this._lostHandler);
    }
  }

  /**
   * Register a storage buffer with a shadow copy on CPU.
   */
  createStorageBuffer(size: number, initialData?: Float32Array): { buffer: GPUBuffer; shadowId: number } {
    if (!this._device) throw new Error("Device not initialized");
    const buffer = this._device.createBuffer({
      size,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
    });
    const shadow: ShadowBuffer = {
      data: initialData ? new Float32Array(initialData) : new Float32Array(size / 4),
      size,
    };
    const shadowId = this._nextShadowId++;
    this._shadowBuffers.set(shadowId, shadow);
    this._pendingBuffers.push({ id: shadowId, shadow });
    return { buffer, shadowId };
  }

  /**
   * Write data to a GPU buffer and update its shadow copy.
   */
  writeBuffer(buffer: GPUBuffer, offset: number, data: ArrayBufferView, shadowId?: number): void {
    if (!this._device) throw new Error("Device not initialized");
    this._device.queue.writeBuffer(buffer, offset, data as unknown as BufferSource);
    if (shadowId !== undefined) {
      const shadow = this._shadowBuffers.get(shadowId);
      if (shadow) {
        const src = new Float32Array(data.buffer as ArrayBuffer, data.byteOffset, data.byteLength / 4);
        shadow.data.set(src, offset / 4);
      }
    }
  }

  /**
   * Read data from a GPU buffer, with automatic device recovery on lost.
   */
  async readFromGPUBuffer(source: GPUBuffer, byteSize: number, maxRetries = 5): Promise<ArrayBuffer> {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        await this.ensureReady();
        const device = this._device!;
        const readBuffer = device.createBuffer({
          size: byteSize,
          usage: BufferUsage.COPY_DST | BufferUsage.MAP_READ,
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, readBuffer, 0, byteSize);
        device.queue.submit([encoder.finish()]);
        await readBuffer.mapAsync(MapMode.READ);
        const copied = readBuffer.getMappedRange().slice(0);
        readBuffer.unmap();
        readBuffer.destroy();
        return copied;
      } catch (err) {
        if (attempt < maxRetries - 1 && err instanceof Error) {
          const msg = err.message || "";
          if (msg.includes("lost") || msg.includes("Device") || msg.includes("Invalid") || msg.includes("destroyed")) {
            console.warn(`[DeviceManager] readFromGPUBuffer attempt ${attempt + 1}/${maxRetries} failed: ${msg}`);
            this._initialized = false;
            this._device = null;
            await sleep(BACKOFF_BASE_MS * Math.pow(2, attempt));
            continue;
          }
        }
        throw err;
      }
    }
    throw new Error("readFromGPUBuffer: all attempts failed");
  }

  /**
   * Re-create a buffer from its shadow copy on a new device.
   */
  recreateBuffer(shadowId: number): GPUBuffer {
    if (!this._device) throw new Error("Device not initialized");
    const shadow = this._shadowBuffers.get(shadowId);
    if (!shadow) throw new Error(`recreateBuffer: unknown shadowId ${shadowId}`);
    const buffer = this._device.createBuffer({
      size: shadow.size,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
    });
    this._device.queue.writeBuffer(buffer, 0, shadow.data as unknown as BufferSource);
    return buffer;
  }

  /**
   * Register a callback that runs after device recovery.
   */
  onRecovery(cb: () => Promise<void>): void {
    this._recoveryCallbacks.push(cb);
  }

  /**
   * Discard a shadow copy when its buffer is no longer needed.
   */
  discardShadow(shadowId: number): void {
    this._shadowBuffers.delete(shadowId);
    this._pendingBuffers = this._pendingBuffers.filter(e => e.id !== shadowId);
  }

  private assertDeviceIndex(deviceIndex?: number): void {
    if (deviceIndex === undefined || deviceIndex === null) return;
    if (deviceIndex !== 0) {
      throw new Error(`Only device index 0 is supported in MVP, received: ${deviceIndex}.`);
    }
  }

  private collectDeviceProperties(): Record<string, unknown> {
    const adapterAny = this._adapter as unknown as {
      info?: { vendor?: string; architecture?: string; device?: string; description?: string };
      isFallbackAdapter?: boolean;
      limits?: { minSubgroupSize?: number; maxSubgroupSize?: number };
    };
    const limits = this._adapter!.limits as unknown as Record<string, number>;
    const info = adapterAny.info ?? {};
    const name = info.description || info.device || info.architecture || info.vendor || "WebGPU Adapter";
    return {
      name,
      total_memory: 0,
      major: 0,
      minor: 0,
      multi_processor_count: 0,
      vendor: info.vendor ?? "",
      architecture: info.architecture ?? "",
      description: info.description ?? "",
      device: info.device ?? "",
      is_fallback_adapter: Boolean(adapterAny.isFallbackAdapter),
      limits,
    };
  }
}
