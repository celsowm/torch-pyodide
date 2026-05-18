import { chromium } from 'playwright';

const browser = await chromium.launch({
  channel: 'chromium',
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,UseSkiaRenderer'],
});

const page = await browser.newPage();
page.on('console', (msg) => console.log('[console]', msg.text()));
page.on('pageerror', (err) => console.log('[pageerror]', err.message));

await page.goto('http://127.0.0.1:4173/demo/index.html', { waitUntil: 'load' });
await page.waitForFunction(() => Boolean(window.__torchMvpStatus), null, { timeout: 120000 });
const status = await page.evaluate(() => window.__torchMvpStatus);
console.log('STATUS:', JSON.stringify(status, null, 2));

await browser.close();
