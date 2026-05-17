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
    headless: true,
    launchOptions: {
      args: [
        "--enable-unsafe-webgpu",
        "--use-angle=swiftshader-webgpu",
        "--enable-features=Vulkan"
      ]
    },
    baseURL: "http://127.0.0.1:4173"
  }
});
