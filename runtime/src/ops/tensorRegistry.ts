import { TensorMeta, TensorHandle, cloneHandle } from "./types.js";

export class TensorRegistry {
  private tensors = new Map<number, TensorMeta>();
  private nextId = 1;
  private allocated = 0;

  all(): Map<number, TensorMeta> {
    return this.tensors;
  }

  nextTensorId(): number {
    return this.nextId++;
  }

  memoryAllocated(): number {
    return this.allocated;
  }

  allocateBytes(bytes: number): void {
    this.allocated += bytes;
  }

  deallocateBytes(bytes: number): void {
    this.allocated = Math.max(0, this.allocated - bytes);
  }

  getTensorMeta(id: number): TensorMeta {
    const meta = this.tensors.get(id);
    if (!meta) throw new Error(`Unknown tensor id: ${id}.`);
    return meta;
  }

  registerTensor(buffer: GPUBuffer, shape: number[], dtype: string, length: number): number {
    const id = this.nextTensorId();
    const bytes = buffer.size;
    const meta: TensorMeta = { id, buffer, shape: [...shape], dtype, length, bytes };
    this.tensors.set(id, meta);
    this.allocated += bytes;
    return id;
  }

  registerTensorAsHandle(buffer: GPUBuffer, shape: number[], dtype: string, length: number): TensorHandle {
    const id = this.registerTensor(buffer, shape, dtype, length);
    return cloneHandle(this.tensors.get(id)!);
  }

  deleteTensor(id: number): TensorMeta | null {
    const meta = this.tensors.get(id);
    if (!meta) return null;
    this.tensors.delete(id);
    this.deallocateBytes(meta.bytes);
    return meta;
  }

  tensorHandle(meta: TensorMeta): { id: number; shape: number[]; dtype: string } {
    return cloneHandle(meta);
  }
}
