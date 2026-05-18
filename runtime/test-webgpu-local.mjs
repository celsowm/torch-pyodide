import { chromium } from 'playwright';

const chromePath = 'C:\\Users\\celso\\AppData\\Local\\ms-playwright\\chromium-1223\\chrome-win64\\chrome.exe';

const configs = [
  {
    name: 'minimal flags',
    args: ['--enable-unsafe-webgpu'],
    headless: false,
  },
  {
    name: 'headless=new minimal',
    args: ['--headless=new', '--enable-unsafe-webgpu'],
    headless: true,
  },
];

for (const cfg of configs) {
  console.log(`\n=== ${cfg.name} ===`);
  try {
    const browser = await chromium.launch({
      executablePath: chromePath,
      headless: cfg.headless,
      args: cfg.args,
    });
    const page = await browser.newPage();
    await page.goto('about:blank');
    // Check browser info
    const ua = await page.evaluate(() => navigator.userAgent);
    console.log('UA:', ua);
    // Check WebGPU with more detail
    const result = await page.evaluate(() => {
      const info = {
        hasNavigatorGPU: 'gpu' in navigator,
        hasGPU: !!navigator.gpu,
        navigatorKeys: Object.keys(navigator).filter(k => k.toLowerCase().includes('gpu')),
      };
      return info;
    });
    console.log(JSON.stringify(result, null, 2));
    await browser.close();
  } catch (e) {
    console.log('Launch error:', e.message);
  }
}
