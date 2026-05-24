import {
  initWebGPU,
  getDevice,
  getAdapter,
  isInitialized as isWebGPUInitialized,
} from "../vendor/torchjs/index.js";

import { TensorMeta, TensorHandle, SupportedDType, cloneHandle, product } from "./types.js";
import { TensorRegistry } from "./tensorRegistry.js";
import { PipelineCache } from "./pipelineCache.js";

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
      console.warn(`[DeviceManager] ${label} attempt ${attempt + 1}/${retries} failed:`, (err as Error)?.message, `retry in ${delay}ms`);
      await sleep(delay);
    }
  }
  throw new Error(`${label} — all ${retries} attempts failed`);
}

interface ShadowBuffer {
  data: Float32Array;
  size: number;
}

function decodeStorageValues(values: Float32Array, length: number, dtype: SupportedDType): number[] {
  const raw = Array.from(values.slice(0, length));
  if (dtype === "bool") return raw.map((v) => (v !== 0 ? 1 : 0));
  if (
    dtype === "int8" ||
    dtype === "int16" ||
    dtype === "int32" ||
    dtype === "int64" ||
    dtype === "uint8" ||
    dtype === "uint16" ||
    dtype === "uint32" ||
    dtype === "uint64"
  ) {
    return raw.map((v) => Math.trunc(v));
  }
  return raw;
}

export class DeviceManager {
  private _device: GPUDevice | null = null;
  private _adapter: GPUAdapter | null = null;
  private _initialized = false;
  private _initPromise: Promise<void> | null = null;
  private _initError: string | null = null;
  private _lostHandler: ((ev: GPUDeviceLostInfo) => void) | null = null;

  // Pipeline cache — keyed by shader+entrypoint
  private _pipelineCache = new PipelineCache();

  // Shadow copies for device recovery
  private _shadowBuffers = new Map<number, ShadowBuffer>();
  private _nextShadowId = 1;
  private _pendingBuffers: Array<{ id: number; shadow: ShadowBuffer }> = [];
  private _recoveryCallbacks: Array<() => Promise<void>> = [];

  // Batch frame — accumulate compute passes, submit once
  private _frameEncoder: GPUCommandEncoder | null = null;
  private _frameDepth = 0;

  // Tensor registry
  private _tensorRegistry = new TensorRegistry();

  // Generation counter — increments on each recovery so readers know device changed
  private _deviceGeneration = 0;

  // Mapping GPUBuffer -> shadowId for fast lookup
  private _bufferToShadow = new Map<GPUBuffer, number>();
  // Track wrapped buffers so destroy() cleanup is attached only once
  private _wrappedDestroy = new WeakSet<GPUBuffer>();

  get device(): GPUDevice | null { return this._device; }
  get adapter(): GPUAdapter | null { return this._adapter; }
  get initialized(): boolean { return this._initialized; }
  get tensors(): Map<number, TensorMeta> { return this._tensorRegistry.all(); }

  isAvailable(): boolean {
    return Boolean(globalThis.navigator?.gpu);
  }

  deviceCount(): number {
    return this.isAvailable() ? 1 : 0;
  }

  async currentDevice(): Promise<number> {
    await this.ensureReady();
    return 0;
  }

  async getDeviceName(_deviceIndex?: number): Promise<string> {
    await this.ensureReady();
    return this.collectProperties().name as string;
  }

  async getDeviceProperties(_deviceIndex?: number): Promise<Record<string, unknown>> {
    await this.ensureReady();
    return this.collectProperties();
  }

  async memoryAllocated(): Promise<number> {
    await this.ensureReady();
    return this._tensorRegistry.memoryAllocated();
  }

  async memoryReserved(): Promise<number> {
    await this.ensureReady();
    return this._tensorRegistry.memoryAllocated();
  }

  async ensureReady(gpuProvider?: GPU | null): Promise<GPUDevice> {
    if (this._initialized && this._device) return this._device;
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
    if (!this._device) throw new Error("DeviceManager: failed to initialize GPU device");
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
    if (!this._device) throw new Error("WebGPU init returned null device");

    this._lostHandler = async (_ev: GPUDeviceLostInfo) => {
      console.warn("[DeviceManager] Device lost — recovering...");
      this._deviceGeneration++;
      this._device = null;
      this._adapter = null;
      this._initialized = false;
      this._initPromise = null;
      this._pipelineCache.clear();

      await withRetry("device recovery", async () => {
        await initWebGPU();
        this._device = getDevice() as GPUDevice;
        this._adapter = getAdapter() as GPUAdapter;
        this._initialized = isWebGPUInitialized();
        if (!this._device) throw new Error("Recovery: failed to reinitialize WebGPU");
        if (this._device.lost) this._device.lost.then(this._lostHandler!);

        const count = this._pendingBuffers.length;
        if (count > 0) {
          console.warn(`[DeviceManager] Re-creating ${count} buffers from shadow...`);
          for (const entry of this._pendingBuffers) {
            const buf = this._device!.createBuffer({
              size: entry.shadow.size,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            this._device!.queue.writeBuffer(buf, 0, entry.shadow.data as GPUAllowSharedBufferSource);
          }
        }
        for (const cb of this._recoveryCallbacks) await cb();
        console.warn("[DeviceManager] Recovery complete");
      }, LOST_RECOVERY_RETRIES);
    };
    if (this._device.lost) this._device.lost.then(this._lostHandler);
  }

  // ── Pipeline cache ──────────────────────────────────────────

  getOrCreatePipeline(shaderCode: string, entryPoint: string): GPUComputePipeline {
    if (!this._device) throw new Error("Device not initialized");
    return this._pipelineCache.getOrCreate(this._device, shaderCode, entryPoint);
  }

  // ── Batch frame ──────────────────────────────────────────────

  /** Begin a batch frame. Nested calls are safe — only the outermost call creates the encoder. */
  beginFrame(): void {
    if (this._frameDepth === 0) {
      if (!this._device) throw new Error("Device not initialized");
      this._frameEncoder = this._device.createCommandEncoder();
    }
    this._frameDepth++;
  }

  /** End a batch frame. Only the outermost call submits. Returns a promise that resolves when work completes. */
  async endFrame(): Promise<void> {
    if (this._frameDepth === 0) throw new Error("endFrame() without beginFrame()");
    this._frameDepth--;
    if (this._frameDepth > 0) return;
    const encoder = this._frameEncoder!;
    this._frameEncoder = null;
    this._device!.queue.submit([encoder.finish()]);
    // Performance mode: do not force a device-wide stall on every frame end.
    // Command ordering on a single queue is preserved; readback paths still synchronize explicitly.
    const target = globalThis as typeof globalThis & { __TORCH_PYODIDE_FRAME_SYNC_EAGER__?: boolean };
    if (target.__TORCH_PYODIDE_FRAME_SYNC_EAGER__) {
      await this._device!.queue.onSubmittedWorkDone();
    }
  }

  /** Cancel and discard a batch frame without submitting. */
  cancelFrame(): void {
    this._frameEncoder = null;
    this._frameDepth = 0;
  }

  dispatchCompute(pipeline: GPUComputePipeline, buffers: GPUBuffer[], workgroupCount: number[]): void {
    if (!this._device) throw new Error("Device not initialized");
    const bindGroup = this._device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({
        binding: i,
        resource: { buffer },
      })),
    });
    const encoder = this._frameEncoder ?? this._device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    if (workgroupCount.length === 2) {
      pass.dispatchWorkgroups(workgroupCount[0]!, workgroupCount[1]!);
    } else {
      pass.dispatchWorkgroups(workgroupCount[0]!);
    }
    pass.end();
    if (!this._frameEncoder) {
      this._device.queue.submit([encoder.finish()]);
    }
  }

  calculateWorkgroups(numElements: number, workgroupSize = 256): number[] {
    const groups = Math.ceil(numElements / workgroupSize);
    if (groups > 65535) {
      const cols = Math.ceil(Math.sqrt(groups));
      const rows = Math.ceil(groups / cols);
      return [cols, rows];
    }
    return [groups];
  }

  syncDevice(): Promise<void> {
    // If we're inside a batch frame, individual sync is deferred to endFrame
    if (this._frameDepth > 0) return Promise.resolve();
    return this._device!.queue.onSubmittedWorkDone();
  }

  // ── Buffer & shadow copy ────────────────────────────────────

  createStorageBuffer(size: number, initialData?: Float32Array): GPUBuffer {
    if (!this._device) throw new Error("Device not initialized");
    const buffer = this._device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const shadow: ShadowBuffer = {
      data: initialData ? new Float32Array(initialData) : new Float32Array(size / 4),
      size,
    };
    const shadowId = this._nextShadowId++;
    this._shadowBuffers.set(shadowId, shadow);
    this._pendingBuffers.push({ id: shadowId, shadow });
    this._bufferToShadow.set(buffer, shadowId);
    this.attachDestroyCleanup(buffer);
    return buffer;
  }

  private attachDestroyCleanup(buffer: GPUBuffer): void {
    if (this._wrappedDestroy.has(buffer)) return;
    this._wrappedDestroy.add(buffer);
    const originalDestroy = buffer.destroy.bind(buffer);
    const self = this;
    (buffer as GPUBuffer & { __torchDestroyWrapped__?: boolean }).destroy = function () {
      const shadowId = self._bufferToShadow.get(buffer);
      if (shadowId !== undefined) self.discardShadow(shadowId);
      self.forgetBuffer(buffer);
      try {
        originalDestroy();
      } catch {
        // Ignore destroy errors (device may already be lost/destroyed).
      }
    };
  }

  writeBuffer(buffer: GPUBuffer, offset: number, data: GPUAllowSharedBufferSource): void {
    if (!this._device) throw new Error("Device not initialized");
    this._device.queue.writeBuffer(buffer, offset, data);
    const shadowId = this._bufferToShadow.get(buffer);
    if (shadowId !== undefined) {
      const shadow = this._shadowBuffers.get(shadowId);
      if (shadow && data instanceof Float32Array) {
        shadow.data.set(data, offset / 4);
      }
    }
  }

  async readFromGPUBuffer(source: GPUBuffer, byteSize: number, maxRetries = 5): Promise<ArrayBuffer> {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        await this.ensureReady();
        const device = this._device!;
        const readBuffer = device.createBuffer({
          size: byteSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, readBuffer, 0, byteSize);
        device.queue.submit([encoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);
        const copied = readBuffer.getMappedRange().slice(0);
        readBuffer.unmap();
        readBuffer.destroy();
        return copied;
      } catch (err) {
        if (attempt < maxRetries - 1 && err instanceof Error) {
          const msg = err.message || "";
          if (msg.includes("lost") || msg.includes("Device") || msg.includes("Invalid") || msg.includes("destroyed")) {
            console.warn(`[DeviceManager] readFromGPU attempt ${attempt + 1}/${maxRetries}: ${msg}`);
            this._initialized = false;
            this._device = null;
            this._initPromise = null;
            this._pipelineCache.clear();
            if (this._adapter) {
              await initWebGPU();
              this._device = getDevice() as GPUDevice;
              this._initialized = isWebGPUInitialized();
              if (this._device?.lost) this._device.lost.then(this._lostHandler!);
            }
            await sleep(BACKOFF_BASE_MS * Math.pow(2, attempt));
            continue;
          }
        }
        throw err;
      }
    }
    throw new Error("readFromGPUBuffer: all attempts failed");
  }

  async readFromGPU(source: GPUBuffer, length: number, dtype: SupportedDType): Promise<number[]> {
    // Shaders use f32 storage for every logical dtype. Metadata tracks dtype
    // compatibility, but GPU readback must use the physical storage width.
    const byteSize = length * Float32Array.BYTES_PER_ELEMENT;
    const shadowId = this._bufferToShadow.get(source);
    const shadow = shadowId !== undefined ? this._shadowBuffers.get(shadowId) : undefined;
    // If device was recovered, shadow is more reliable than GPU read
    if (shadow && this._deviceGeneration > 0) {
      console.warn("[DeviceManager] Using shadow copy (device recovered)");
      return decodeStorageValues(shadow.data, length, dtype);
    }
    try {
      const data = await this.readFromGPUBuffer(source, byteSize);
      return decodeStorageValues(new Float32Array(data), length, dtype);
    } catch (err) {
      if (shadow) {
        console.warn("[DeviceManager] Falling back to shadow copy");
        return decodeStorageValues(shadow.data, length, dtype);
      }
      throw err;
    }
  }

  async readScalar(buffer: GPUBuffer): Promise<number> {
    const data = await this.readFromGPUBuffer(buffer, 4);
    return new Float32Array(data)[0];
  }

  recreateBuffer(shadowId: number): GPUBuffer {
    if (!this._device) throw new Error("Device not initialized");
    const shadow = this._shadowBuffers.get(shadowId);
    if (!shadow) throw new Error(`recreateBuffer: unknown shadowId ${shadowId}`);
    const buffer = this._device.createBuffer({
      size: shadow.size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this._device.queue.writeBuffer(buffer, 0, shadow.data as GPUAllowSharedBufferSource);
    this._bufferToShadow.set(buffer, shadowId);
    this.attachDestroyCleanup(buffer);
    return buffer;
  }

  /** Force device recovery (used for testing). Destroys current device and reinitializes. */
  async forceDeviceRecovery(): Promise<void> {
    if (this._device) {
      try { this._device.destroy(); } catch { /* ignore */ }
    }
    this._deviceGeneration++;
    this._device = null;
    this._adapter = null;
    this._initialized = false;
    this._initPromise = null;
    this._pipelineCache.clear();
    await withRetry("forced recovery", async () => {
      await initWebGPU();
      this._device = getDevice() as GPUDevice;
      this._adapter = getAdapter() as GPUAdapter;
      this._initialized = isWebGPUInitialized();
      if (!this._device) throw new Error("Forced recovery: failed to reinitialize WebGPU");
      if (this._device.lost) this._device.lost.then(this._lostHandler!);
      for (const cb of this._recoveryCallbacks) await cb();
      console.warn("[DeviceManager] Forced recovery complete");
    }, LOST_RECOVERY_RETRIES);
  }

  onRecovery(cb: () => Promise<void>): void {
    this._recoveryCallbacks.push(cb);
  }

  discardShadow(shadowId: number): void {
    this._shadowBuffers.delete(shadowId);
    this._pendingBuffers = this._pendingBuffers.filter(e => e.id !== shadowId);
  }

  forgetBuffer(buffer: GPUBuffer): void {
    this._bufferToShadow.delete(buffer);
  }

  // ── Tensor registry ─────────────────────────────────────────

  nextTensorId(): number {
    return this._tensorRegistry.nextTensorId();
  }

  allocateBytes(bytes: number): void {
    this._tensorRegistry.allocateBytes(bytes);
  }

  deallocateBytes(bytes: number): void {
    this._tensorRegistry.deallocateBytes(bytes);
  }

  getTensorMeta(id: number): TensorMeta {
    return this._tensorRegistry.getTensorMeta(id);
  }

  registerTensor(buffer: GPUBuffer, shape: number[], dtype: string, length: number): number {
    return this._tensorRegistry.registerTensor(buffer, shape, dtype, length);
  }

  /** Register a tensor buffer and return a TensorHandle (for public API consumption). */
  registerTensorAsHandle(buffer: GPUBuffer, shape: number[], dtype: string, length: number): TensorHandle {
    return this._tensorRegistry.registerTensorAsHandle(buffer, shape, dtype, length);
  }

  destroyTensor(id: number): void {
    const meta = this.tensors.get(id);
    if (!meta) return;
    const shadowId = this._bufferToShadow.get(meta.buffer);
    if (shadowId !== undefined) this.discardShadow(shadowId);
    this.forgetBuffer(meta.buffer);
    try { meta.buffer.destroy(); } catch { /* device may be gone */ }
    this._tensorRegistry.deleteTensor(id);
  }

  tensorHandle(meta: TensorMeta): { id: number; shape: number[]; dtype: string } {
    return cloneHandle(meta);
  }

  // ── Device properties ───────────────────────────────────────

  private collectProperties(): Record<string, unknown> {
    const adapterAny = this._adapter as unknown as {
      info?: Record<string, string>;
      isFallbackAdapter?: boolean;
    };
    const limits = this._adapter!.limits as unknown as Record<string, number>;
    const info = adapterAny.info ?? {};
    const name = info.description || info.device || info.architecture || info.vendor || "WebGPU Adapter";
    return { name, total_memory: 0, major: 0, minor: 0, multi_processor_count: 0, ...info, limits };
  }
}
