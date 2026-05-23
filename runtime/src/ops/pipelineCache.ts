export class PipelineCache {
  private pipelines = new Map<string, GPUComputePipeline>();
  private shaderModules = new Map<string, GPUShaderModule>();

  private makePipelineKey(shaderCode: string, entryPoint: string): string {
    // Use full shader source in key to avoid collisions between distinct shaders
    // that happen to share the same source length.
    return `${entryPoint}::${shaderCode}`;
  }

  clear(): void {
    this.pipelines.clear();
    this.shaderModules.clear();
  }

  getOrCreate(device: GPUDevice, shaderCode: string, entryPoint: string): GPUComputePipeline {
    const key = this.makePipelineKey(shaderCode, entryPoint);
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
