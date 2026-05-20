import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 300000,
  webServer: {
    command: "npm run demo",
    port: 4173,
    cwd: ".",
    timeout: 120000
  },
  use: {
    baseURL: "http://127.0.0.1:4173"
  },
  projects: [
    {
      name: "gpu-headed",
      grep: /@webgpu/,
      use: {
        channel: "chromium",
        headless: false,
        launchOptions: {
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,UseSkiaRenderer"
          ]
        }
      }
    }
  ]
});
