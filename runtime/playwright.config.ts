import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 120000,
  webServer: {
    command: "npm run demo",
    port: 4173,
    cwd: "."
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
            "--enable-features=Vulkan,UseSkiaRenderer",
            "--start-minimized",
            "--window-position=2400,0"
          ]
        }
      }
    },
    {
      name: "chrome-webgpu",
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
