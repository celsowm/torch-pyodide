export type WebGPUDevice = GPUDevice;
export type WebGPUAdapter = GPUAdapter;
export type WebGPUBuffer = GPUBuffer;
export type WebGPUShaderModule = GPUShaderModule;
export type WebGPUComputePipeline = GPUComputePipeline;
export type WebGPUBindGroup = GPUBindGroup;
export type WebGPUBindGroupLayout = GPUBindGroupLayout;
export type WebGPUCommandEncoder = GPUCommandEncoder;
export type WebGPUCommandBuffer = GPUCommandBuffer;
export type WebGPUComputePassEncoder = GPUComputePassEncoder;
export type WebGPUQueue = GPUQueue;

/**
 * Check if we're running in Node.js environment.
 */
function isNode(): boolean {
  return typeof process !== 'undefined' && process.versions?.node !== undefined;
}

// Buffer usage flags - unified for browser and Node.js
let _BufferUsage: {
  MAP_READ: number;
  MAP_WRITE: number;
  COPY_SRC: number;
  COPY_DST: number;
  INDEX: number;
  VERTEX: number;
  UNIFORM: number;
  STORAGE: number;
  INDIRECT: number;
  QUERY_RESOLVE: number;
};

let _MapMode: {
  READ: number;
  WRITE: number;
};

if (isNode()) {
  // Will be initialized when the module is imported
  _BufferUsage = {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
  };
  _MapMode = {
    READ: 0x0001,
    WRITE: 0x0002,
  };
} else {
  // Browser WebGPU constants
  _BufferUsage = {
    MAP_READ: GPUBufferUsage?.MAP_READ ?? 0x0001,
    MAP_WRITE: GPUBufferUsage?.MAP_WRITE ?? 0x0002,
    COPY_SRC: GPUBufferUsage?.COPY_SRC ?? 0x0004,
    COPY_DST: GPUBufferUsage?.COPY_DST ?? 0x0008,
    INDEX: GPUBufferUsage?.INDEX ?? 0x0010,
    VERTEX: GPUBufferUsage?.VERTEX ?? 0x0020,
    UNIFORM: GPUBufferUsage?.UNIFORM ?? 0x0040,
    STORAGE: GPUBufferUsage?.STORAGE ?? 0x0080,
    INDIRECT: GPUBufferUsage?.INDIRECT ?? 0x0100,
    QUERY_RESOLVE: GPUBufferUsage?.QUERY_RESOLVE ?? 0x0200,
  };
  _MapMode = {
    READ: GPUMapMode?.READ ?? 0x0001,
    WRITE: GPUMapMode?.WRITE ?? 0x0002,
  };
}

export const BufferUsage = _BufferUsage;
export const MapMode = _MapMode;
