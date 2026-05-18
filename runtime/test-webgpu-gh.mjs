import { chromium } from 'playwright';

const configs = [
  {
    name: 'Google Chrome headless=new with unsafe-webgpu + Mesa Vulkan',
    channel: 'chrome',
    args: [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--use-gl=egl',
      '--no-sandbox',
    ]
  },
  {
    name: 'Google Chrome headless=new with unsafe-webgpu + SwiftShader',
    channel: 'chrome',
    args: [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--no-sandbox',
    ]
  },
  {
    name: 'Google Chrome headless=new with unsafe-webgpu (no GL override)',
    channel: 'chrome',
    args: [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--no-sandbox',
    ]
  },
  {
    name: 'Google Chrome headed with Xvfb virtual display',
    channel: 'chrome',
    args: [
      '--no-sandbox',
      '--disable-gpu-sandbox',
    ]
  },
];

for (const cfg of configs) {
  console.log(`\n=== ${cfg.name} ===`);
  try {
    const browser = await chromium.launch({
      channel: cfg.channel,
      headless: cfg.name.includes('headless') ? true : false,
      args: cfg.args,
    });
    const page = await browser.newPage();
    await page.goto('about:blank');
    const result = await page.evaluate(async () => {
      if (!navigator.gpu) return { gpu: false, error: 'navigator.gpu is undefined' };
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { gpu: true, adapter: false, error: 'requestAdapter returned null' };
        const device = await adapter.requestDevice();
        const info = {
          name: adapter.name,
          features: Array.from(adapter.features).join(', '),
          deviceCreated: !!device,
        };
        return { gpu: true, adapter: true, info };
      } catch (e) {
        return { gpu: true, adapter: false, error: e.message };
      }
    });
    console.log(JSON.stringify(result, null, 2));
    await browser.close();
    if (result.gpu && result.adapter) {
      console.log('✓ SUCCESS: WebGPU is available!');
      process.exit(0);
    }
  } catch (e) {
    console.log(`Launch error: ${e.message}`);
  }
}

console.log('\n✗ All approaches failed. WebGPU not available.');
process.exit(1);
