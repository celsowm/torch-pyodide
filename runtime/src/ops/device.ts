import {
  initWebGPU,
  getDevice,
  getAdapter,
  isInitialized as isWebGPUInitialized,
} from "../vendor/torchjs/index.js";

import { TensorMeta, SupportedDType, cloneHandle, product } from "./types.js";
import { decodeValuesByDType } from "./shape.js";

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

interface PipelineCacheEntry {
  pipeline: GPUComputePipeline;
  shader: string;
  entryPoint: string;
}

export class DeviceManager {
  private _device: GPUDevice | null = null;
  private _adapter: GPUAdapter | null = null;
  private _initialized = false;
  private _initPromise: Promise<void> | null = null;
  private _initError: string | null = null;
  private _lostHandler: ((ev: GPUDeviceLostInfo) => void) | null = null;

  // Pipeline cache — keyed by shader+entrypoint
  private _pipelines = new Map<string, PipelineCacheEntry>();
  private _shaderModules = new Map<string, GPUShaderModule>();

  // Shadow copies for device recovery
  private _shadowBuffers = new Map<number, ShadowBuffer>();
  private _nextShadowId = 1;
  private _pendingBuffers: Array<{ id: number; shadow: ShadowBuffer }> = [];
  private _recoveryCallbacks: Array<() => Promise<void>> = [];

  // Batch frame — accumulate compute passes, submit once
  private _frameEncoder: GPUCommandEncoder | null = null;
  private _frameDepth = 0;

  // Tensor registry
  private _tensors = new Map<number, TensorMeta>();
  private _nextTensorId = 1;
  private _allocatedBytes = 0;

  // Mapping GPUBuffer -> shadowId for fast lookup
  private _bufferToShadow = new Map<GPUBuffer, number>();

  get device(): GPUDevice | null { return this._device; }
  get adapter(): GPUAdapter | null { return this._adapter; }
  get initialized(): boolean { return this._initialized; }
  get tensors(): Map<number, TensorMeta> { return this._tensors; }

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
    return this._allocatedBytes;
  }

  async memoryReserved(): Promise<number> {
    await this.ensureReady();
    return this._allocatedBytes;
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
      this._device = null;
      this._adapter = null;
      this._initialized = false;
      this._initPromise = null;
      this._pipelines.clear();
      this._shaderModules.clear();

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
            this._device!.queue.writeBuffer(buf, 0, entry.shadow.data as AllowSharedBufferSource);
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
    const key = `${shaderCode.length}:${entryPoint}`;
    const cached = this._pipelines.get(key);
    if (cached) return cached.pipeline;

    let module = this._shaderModules.get(shaderCode);
    if (!module) {
      module = this._device.createShaderModule({ code: shaderCode });
      this._shaderModules.set(shaderCode, module);
    }
    const pipeline = this._device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint },
    });
    this._pipelines.set(key, { pipeline, shader: shaderCode, entryPoint });
    return pipeline;
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
    await this._device!.queue.onSubmittedWorkDone();
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
    return buffer;
  }

  writeBuffer(buffer: GPUBuffer, offset: number, data: AllowSharedBufferSource): void {
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
    const byteSize = length * 4;
    try {
      const data = await this.readFromGPUBuffer(source, byteSize);
      return decodeValuesByDType(data, dtype);
    } catch (err) {
      const shadowId = this._bufferToShadow.get(source);
      if (shadowId !== undefined) {
        const shadow = this._shadowBuffers.get(shadowId);
        if (shadow) {
          console.warn("[DeviceManager] Falling back to shadow copy");
          return Array.from(shadow.data).slice(0, length) as number[];
        }
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
    this._device.queue.writeBuffer(buffer, 0, shadow.data as AllowSharedBufferSource);
    this._bufferToShadow.set(buffer, shadowId);
    return buffer;
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
    return this._nextTensorId++;
  }

  allocateBytes(bytes: number): void {
    this._allocatedBytes += bytes;
  }

  deallocateBytes(bytes: number): void {
    this._allocatedBytes = Math.max(0, this._allocatedBytes - bytes);
  }

  getTensorMeta(id: number): TensorMeta {
    const meta = this._tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }

  registerTensor(buffer: GPUBuffer, shape: number[], dtype: string, length: number): number {
    const id = this.nextTensorId();
    const bytes = buffer.size;
    const meta: TensorMeta = { id, buffer, shape: [...shape], dtype, length, bytes };
    this._tensors.set(id, meta);
    this._allocatedBytes += bytes;
    return id;
  }

  destroyTensor(id: number): void {
    const meta = this._tensors.get(id);
    if (!meta) return;
    const shadowId = this._bufferToShadow.get(meta.buffer);
    if (shadowId !== undefined) this.discardShadow(shadowId);
    this.forgetBuffer(meta.buffer);
    try { meta.buffer.destroy(); } catch { /* device may be gone */ }
    this.deallocateBytes(meta.bytes);
    this._tensors.delete(id);
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
