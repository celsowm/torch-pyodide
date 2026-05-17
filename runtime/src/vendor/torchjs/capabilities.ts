import { getAdapter } from './device.js';

export interface GPULimits {
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupsPerDimension: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}

export interface GPUCapabilities {
  workgroupSharedMemory: boolean;
  timestampQuery: boolean;
  subgroups: boolean;
  limits: GPULimits;
  platform: {
    browser: string;
    gpu: string;
  };
}

let capabilities: GPUCapabilities | null = null;

export async function detectCapabilities(): Promise<GPUCapabilities> {
  if (capabilities) return capabilities;
  const adapter = getAdapter();
  const l = adapter.limits;
  const limits: GPULimits = {
    maxComputeWorkgroupSizeX: l.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: l.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: l.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension,
    maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
    maxBufferSize: l.maxBufferSize,
  };
  capabilities = {
    workgroupSharedMemory: true,
    timestampQuery: false,
    subgroups: false,
    limits,
    platform: { browser: "unknown", gpu: "unknown" },
  };
  return capabilities;
}

export function getCapabilities(): GPUCapabilities {
  if (!capabilities) throw new Error("GPU capabilities not detected");
  return capabilities;
}