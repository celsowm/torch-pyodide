import path from "node:path";
import { defineConfig } from "vite";

function wgslLoader() {
  return {
    name: "wgsl-loader",
    transform(code: string, id: string) {
      if (id.endsWith(".wgsl")) {
        return {
          code: `export default ${JSON.stringify(code)};`,
          map: null
        };
      }
      return null;
    }
  };
}

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? "/torch-pyodide/" : "/",
  plugins: [wgslLoader()],
  server: {
    fs: {
      allow: [path.resolve(__dirname, "..")]
    }
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      input: {
        demo: path.resolve(__dirname, "demo/index.html"),
        playground: path.resolve(__dirname, "playground/index.html")
      }
    }
  }
});
