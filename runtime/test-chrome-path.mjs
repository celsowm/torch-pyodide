import { chromium } from 'playwright';
import { execSync } from 'child_process';

try {
  const dirs = execSync('ls -la ~/.cache/ms-playwright/', { encoding: 'utf8' });
  console.log('Playwright cache contents:');
  console.log(dirs);
} catch(e) {
  console.log(e.message);
}

try {
  const ls = execSync('find ~/.cache/ms-playwright -name "chrome" -type f 2>/dev/null', { encoding: 'utf8' });
  console.log('\nChrome executables found:');
  console.log(ls);
} catch(e) {
  console.log('find failed:', e.message);
}
