import { chromium } from 'playwright';
import { execSync } from 'child_process';

// Find Chrome executable path
function findChrome() {
  try {
    // Try which/google-chrome
    const which = execSync('which google-chrome-stable 2>/dev/null || which google-chrome 2>/dev/null || which chromium-browser 2>/dev/null', { encoding: 'utf8' }).trim();
    if (which) return which;
  } catch (_) {}
  // Check common paths
  const common = [
    '/usr/bin/google-chrome-stable',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium-browser',
    '/snap/bin/chromium',
  ];
  for (const p of common) {
    try { execSync(`test -x ${p}`); return p; } catch (_) {}
  }
  return null;
}

const chromePath = findChrome();
console.log(`Chrome executable: ${chromePath || 'NOT FOUND'}`);
if (!chromePath) {
  console.log('Google Chrome not found. Exiting.');
  process.exit(1);
}

// Get Chrome version
try {
  const ver = execSync(`"${chromePath}" --version 2>/dev/null || true`, { encoding: 'utf8' }).trim();
  console.log(`Chrome version: ${ver}`);
} catch (_) {}

const configs = [
  {
    name: 'headless=new + unsafe-webgpu + Mesa egl',
    args: [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--use-gl=egl',
      '--no-sandbox',
    ]
  },
  {
    name: 'headless=new + unsafe-webgpu + SwiftShader',
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
    name: 'headless=new + unsafe-webgpu (no GL override)',
    args: [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--no-sandbox',
    ]
  },
];

for (const cfg of configs) {
  console.log(`\n=== ${cfg.name} ===`);
  try {
    const browser = await chromium.launch({
      executablePath: chromePath,
      headless: true,
      args: cfg.args,
    });
    const page = await browser.newPage();
    await page.goto('about:blank');
    const result = await page.evaluate(async () => {
      if (!navigator.gpu) {
        return { gpu: false, error: 'navigator.gpu is undefined' };
      }
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          return { gpu: true, adapter: false, error: 'requestAdapter returned null' };
        }
        const device = await adapter.requestDevice();
        return {
          gpu: true,
          adapter: true,
          info: {
            name: adapter.name,
            features: Array.from(adapter.features).join(', '),
            deviceCreated: !!device,
          }
        };
      } catch (e) {
        return { gpu: true, adapter: false, error: e.message };
      }
    });
    console.log(JSON.stringify(result, null, 2));
    await browser.close();
    if (result.gpu && result.adapter) {
      console.log('\n*** SUCCESS! ***');
      process.exit(0);
    }
  } catch (e) {
    console.log(`Launch error: ${e.message}`);
  }
}

console.log('\n*** All approaches failed ***');
process.exit(1);
