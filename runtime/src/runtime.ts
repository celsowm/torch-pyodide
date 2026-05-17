import elementwiseShader from "./vendor/torchjs/shaders/elementwise.wgsl";
import fillShader from "./vendor/torchjs/shaders/fill.wgsl";
import matmulShader from "./vendor/torchjs/shaders/matmul.wgsl";
import reduceSumShader from "./vendor/torchjs/shaders/reduce_sum.wgsl";

type TensorMeta = {
  id: number;
  buffer: GPUBuffer;
  shape: number[];
  dtype: string;
  length: number;
};

type TensorHandle = {
  id: number;
  shape: number[];
  dtype: string;
};

const unaryReluShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&output)) { return; }
  let v = input[idx];
  output[idx] = max(v, 0.0);
}
`;

const transpose2dShader = `
struct Dims {
  rows: u32,
  cols: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;
@compute @workgroup_size(256)
fn transpose2d(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = dims.rows * dims.cols;
  if (idx >= total) { return; }
  let r = idx / dims.cols;
  let c = idx % dims.cols;
  output[c * dims.rows + r] = input[idx];
}
`;

function product(values: number[]): number {
  return values.reduce((acc, value) => acc * value, 1);
}

function cloneHandle(meta: TensorMeta): TensorHandle {
  return {
    id: meta.id,
    shape: [...meta.shape],
    dtype: meta.dtype
  };
}

export class TorchPyodideRuntime {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private initialized = false;
  private nextId = 1;
  private tensors = new Map<number, TensorMeta>();
  private pipelineCache = new Map<string, GPUComputePipeline>();

  async init(gpuProvider?: GPU | null): Promise<void> {
    if (this.initialized) {
      return;
    }
    const gpu = gpuProvider === undefined ? globalThis.navigator?.gpu : gpuProvider;
    if (!gpu) {
      throw new Error("WebGPU unavailable in this browser.");
    }
    let adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      adapter = await gpu.requestAdapter();
    }
    if (!adapter) {
      throw new Error("Failed to request WebGPU adapter.");
    }
    const device = await adapter.requestDevice();
    this.adapter = adapter;
    this.device = device;
    this.initialized = true;
  }

  async tensorFromData(data: number[], shape: number[], dtype: string): Promise<TensorHandle> {
    this.ensureReady();
    this.assertDType(dtype);
    const length = product(shape);
    if (length !== data.length) {
      throw new Error(`tensorFromData expected ${length} values, got ${data.length}.`);
    }
    const typed = new Float32Array(data);
    const buffer = this.createStorageBuffer(typed.byteLength);
    this.device!.queue.writeBuffer(buffer, 0, typed);
    const meta = this.registerTensor(buffer, shape, dtype, length);
    return cloneHandle(meta);
  }

  async zeros(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, 0.0);
  }

  async ones(shape: number[], dtype: string): Promise<TensorHandle> {
    return this.fill(shape, dtype, 1.0);
  }

  async add(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "add");
  }

  async mul(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "mul");
  }

  async sub(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "sub");
  }

  async div(aId: number, bId: number): Promise<TensorHandle> {
    return this.elementwise(aId, bId, "div_op");
  }

  async relu(tensorId: number): Promise<TensorHandle> {
    this.ensureReady();
    const source = this.getTensor(tensorId);
    const out = this.createStorageBuffer(Math.max(4, source.length * 4));
    const pipeline = this.getPipeline(unaryReluShader, "relu");
    this.dispatch(pipeline, [source.buffer, out], [Math.ceil(source.length / 256), 1, 1]);
    await this.device!.queue.onSubmittedWorkDone();
    const meta = this.registerTensor(out, source.shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  async matmul(aId: number, bId: number): Promise<TensorHandle> {
    this.ensureReady();
    const a = this.getTensor(aId);
    const b = this.getTensor(bId);
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error("matmul currently supports only 2D tensors.");
    }
    const [m, kA] = a.shape;
    const [kB, n] = b.shape;
    if (kA !== kB) {
      throw new Error(`matmul dimension mismatch: ${kA} != ${kB}.`);
    }
    const outLength = m * n;
    const outBuffer = this.createStorageBuffer(outLength * 4);

    const dimsData = new Uint32Array([m, kA, n, 1]);
    const dimsBuffer = this.device!.createBuffer({
      size: dimsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(dimsBuffer, 0, dimsData);

    const pipeline = this.getPipeline(matmulShader, "matmul_2d");
    this.dispatch(
      pipeline,
      [a.buffer, b.buffer, outBuffer, dimsBuffer],
      [Math.ceil(outLength / 256), 1, 1]
    );
    await this.device!.queue.onSubmittedWorkDone();
    dimsBuffer.destroy();

    const out = this.registerTensor(outBuffer, [m, n], "float32", outLength);
    return cloneHandle(out);
  }

  async sum(tensorId: number): Promise<TensorHandle> {
    return this.reduce(tensorId, false);
  }

  async mean(tensorId: number): Promise<TensorHandle> {
    return this.reduce(tensorId, true);
  }

  async reshape(tensorId: number, shape: number[]): Promise<TensorHandle> {
    this.ensureReady();
    const source = this.getTensor(tensorId);
    const outLength = product(shape);
    if (outLength !== source.length) {
      throw new Error(`reshape mismatch: new shape has ${outLength} elements, expected ${source.length}.`);
    }
    const out = this.createStorageBuffer(Math.max(4, source.length * 4));
    const encoder = this.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(source.buffer, 0, out, 0, source.length * 4);
    this.device!.queue.submit([encoder.finish()]);
    await this.device!.queue.onSubmittedWorkDone();
    const meta = this.registerTensor(out, shape, source.dtype, source.length);
    return cloneHandle(meta);
  }

  async transpose2d(tensorId: number): Promise<TensorHandle> {
    this.ensureReady();
    const source = this.getTensor(tensorId);
    if (source.shape.length !== 2) {
      throw new Error("transpose2d currently supports only rank-2 tensors.");
    }
    const [rows, cols] = source.shape;
    const out = this.createStorageBuffer(Math.max(4, source.length * 4));
    const dimsData = new Uint32Array([rows, cols]);
    const dimsBuffer = this.device!.createBuffer({
      size: dimsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(dimsBuffer, 0, dimsData);
    const pipeline = this.getPipeline(transpose2dShader, "transpose2d");
    this.dispatch(pipeline, [source.buffer, out, dimsBuffer], [Math.ceil(source.length / 256), 1, 1]);
    await this.device!.queue.onSubmittedWorkDone();
    dimsBuffer.destroy();
    const meta = this.registerTensor(out, [cols, rows], source.dtype, source.length);
    return cloneHandle(meta);
  }

  async toList(tensorId: number): Promise<number[]> {
    this.ensureReady();
    const meta = this.getTensor(tensorId);
    const readBuffer = this.device!.createBuffer({
      size: meta.length * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const encoder = this.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(meta.buffer, 0, readBuffer, 0, meta.length * 4);
    this.device!.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const copied = readBuffer.getMappedRange();
    const values = Array.from(new Float32Array(copied.slice(0)));
    readBuffer.unmap();
    readBuffer.destroy();
    return values;
  }

  async destroy(tensorId: number): Promise<void> {
    const meta = this.tensors.get(tensorId);
    if (!meta) {
      return;
    }
    meta.buffer.destroy();
    this.tensors.delete(tensorId);
  }

  private async fill(shape: number[], dtype: string, value: number): Promise<TensorHandle> {
    this.ensureReady();
    this.assertDType(dtype);
    const length = product(shape);
    const out = this.createStorageBuffer(Math.max(4, length * 4));
    const params = new ArrayBuffer(8);
    const view = new DataView(params);
    view.setFloat32(0, value, true);
    view.setUint32(4, length, true);
    const paramBuffer = this.device!.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(paramBuffer, 0, params);

    const pipeline = this.getPipeline(fillShader, "fill");
    this.dispatch(pipeline, [out, paramBuffer], [Math.ceil(length / 256), 1, 1]);
    await this.device!.queue.onSubmittedWorkDone();
    paramBuffer.destroy();

    const meta = this.registerTensor(out, shape, dtype, length);
    return cloneHandle(meta);
  }

  private async elementwise(aId: number, bId: number, op: "add" | "mul" | "sub" | "div_op"): Promise<TensorHandle> {
    this.ensureReady();
    const a = this.getTensor(aId);
    const b = this.getTensor(bId);
    if (a.length !== b.length) {
      throw new Error(`Shape mismatch for ${op}: ${a.length} != ${b.length}.`);
    }
    if (a.shape.join(",") !== b.shape.join(",")) {
      throw new Error(`Shape mismatch for ${op}: ${a.shape} vs ${b.shape}.`);
    }
    const out = this.createStorageBuffer(Math.max(4, a.length * 4));
    const pipeline = this.getPipeline(elementwiseShader, op);
    this.dispatch(pipeline, [a.buffer, b.buffer, out], [Math.ceil(a.length / 256), 1, 1]);
    await this.device!.queue.onSubmittedWorkDone();
    const meta = this.registerTensor(out, a.shape, a.dtype, a.length);
    return cloneHandle(meta);
  }

  private async reduce(tensorId: number, asMean: boolean): Promise<TensorHandle> {
    this.ensureReady();
    const source = this.getTensor(tensorId);
    let currentBuffer = source.buffer;
    let currentLength = source.length;
    let temporaryToDestroy: GPUBuffer[] = [];

    while (currentLength > 1) {
      const groups = Math.ceil(currentLength / 256);
      const outBuffer = this.createStorageBuffer(Math.max(4, groups * 4));
      const paramData = new Uint32Array([currentLength]);
      const paramBuffer = this.device!.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      this.device!.queue.writeBuffer(paramBuffer, 0, paramData);

      const pipeline = this.getPipeline(reduceSumShader, "main");
      this.dispatch(pipeline, [currentBuffer, outBuffer, paramBuffer], [groups, 1, 1]);
      await this.device!.queue.onSubmittedWorkDone();
      paramBuffer.destroy();

      if (currentBuffer !== source.buffer) {
        temporaryToDestroy.push(currentBuffer);
      }
      currentBuffer = outBuffer;
      currentLength = groups;
    }

    let scalarValue = await this.readScalar(currentBuffer);
    if (asMean) {
      scalarValue /= source.length;
    }

    if (currentBuffer !== source.buffer) {
      temporaryToDestroy.push(currentBuffer);
    }
    for (const tmp of temporaryToDestroy) {
      tmp.destroy();
    }

    const outBuffer = this.createStorageBuffer(4);
    this.device!.queue.writeBuffer(outBuffer, 0, new Float32Array([scalarValue]));
    const meta = this.registerTensor(outBuffer, [], source.dtype, 1);
    return cloneHandle(meta);
  }

  private async readScalar(buffer: GPUBuffer): Promise<number> {
    const readBuffer = this.device!.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const encoder = this.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, 4);
    this.device!.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const view = new Float32Array(readBuffer.getMappedRange().slice(0));
    const value = view[0];
    readBuffer.unmap();
    readBuffer.destroy();
    return value;
  }

  private registerTensor(buffer: GPUBuffer, shape: number[], dtype: string, length: number): TensorMeta {
    const id = this.nextId++;
    const meta: TensorMeta = {
      id,
      buffer,
      shape: [...shape],
      dtype,
      length
    };
    this.tensors.set(id, meta);
    return meta;
  }

  private getTensor(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) {
      throw new Error(`Unknown tensor id: ${id}.`);
    }
    return meta;
  }

  private createStorageBuffer(size: number): GPUBuffer {
    return this.device!.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
  }

  private ensureReady() {
    if (!this.device || !this.initialized || !this.adapter) {
      throw new Error("Runtime not initialized. Call torch.init() first.");
    }
  }

  private assertDType(dtype: string) {
    if (dtype !== "float32") {
      throw new Error(`Only float32 is supported in MVP, received: ${dtype}.`);
    }
  }

  private getPipeline(shaderCode: string, entryPoint: string): GPUComputePipeline {
    const key = `${entryPoint}:${shaderCode}`;
    const cached = this.pipelineCache.get(key);
    if (cached) {
      return cached;
    }
    const module = this.device!.createShaderModule({ code: shaderCode });
    const pipeline = this.device!.createComputePipeline({
      layout: "auto",
      compute: {
        module,
        entryPoint
      }
    });
    this.pipelineCache.set(key, pipeline);
    return pipeline;
  }

  private dispatch(
    pipeline: GPUComputePipeline,
    buffers: GPUBuffer[],
    workgroups: [number, number, number]
  ): void {
    const bindGroup = this.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({
        binding,
        resource: { buffer, offset: 0, size: buffer.size }
      }))
    });
    const encoder = this.device!.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
    pass.end();
    this.device!.queue.submit([encoder.finish()]);
  }
}

export function installTorchRuntime(target: typeof globalThis = globalThis): TorchPyodideRuntime {
  const runtime = new TorchPyodideRuntime();
  (target as typeof globalThis & { __torch_pyodide_runtime__?: TorchPyodideRuntime }).__torch_pyodide_runtime__ =
    runtime;
  return runtime;
}
