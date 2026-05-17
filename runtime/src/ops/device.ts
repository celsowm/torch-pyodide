import {
  initWebGPU,
  getDevice,
  getAdapter,
  isInitialized as isWebGPUInitialized,
} from "../vendor/torchjs/index.js";

export class DeviceManager {
  private _device: GPUDevice | null = null;
  private _adapter: GPUAdapter | null = null;
  private _initialized = false;
  private _initPromise: Promise<void> | null = null;
  private _initError: string | null = null;
  private _lostHandler: ((ev: GPUDeviceLostInfo) => void) | null = null;

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

  async ensureReady(gpuProvider?: GPU | null): Promise<void> {
    if (this._initialized && this._device && this._adapter) {
      return;
    }
    if (this._initPromise) {
      await this._initPromise;
      return;
    }
    this._initPromise = this.initializeInternal(gpuProvider);
    try {
      await this._initPromise;
    } finally {
      this._initPromise = null;
    }
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
    this._lostHandler = (_ev: GPUDeviceLostInfo) => {
      this._device = null;
      this._adapter = null;
      this._initialized = false;
      this._initPromise = null;
    };
    if (this._device?.lost) {
      this._device.lost.then(this._lostHandler);
    }
  }

  private assertDeviceIndex(deviceIndex?: number): void {
    if (deviceIndex === undefined || deviceIndex === null) {
      return;
    }
    if (deviceIndex !== 0) {
      throw new Error(`Only device index 0 is supported in MVP, received: ${deviceIndex}.`);
    }
  }

  private collectDeviceProperties(): Record<string, unknown> {
    const adapterAny = this._adapter as unknown as {
      info?: { vendor?: string; architecture?: string; device?: string; description?: string };
      isFallbackAdapter?: boolean;
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
      subgroup_min_size: limits.minSubgroupSize ?? 0,
      subgroup_max_size: limits.maxSubgroupSize ?? 0,
      limits
    };
  }
}
