export class PipelineCache {
  private pipelines = new Map<string, GPUComputePipeline>();
  private shaderModules = new Map<string, GPUShaderModule>();

  clear(): void {
    this.pipelines.clear();
    this.shaderModules.clear();
  }

  getOrCreate(device: GPUDevice, shaderCode: string, entryPoint: string): GPUComputePipeline {
    const key = `${shaderCode.length}:${entryPoint}`;
    const cached = this.pipelines.get(key);
    if (cached) return cached;

    let module = this.shaderModules.get(shaderCode);
    if (!module) {
      module = device.createShaderModule({ code: shaderCode });
      this.shaderModules.set(shaderCode, module);
    }

    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint },
    });
    this.pipelines.set(key, pipeline);
    return pipeline;
  }
}
