const puppeteer = require('puppeteer-core');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/google-chrome',
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();

  const filePath = path.resolve(__dirname, 'slides.html');
  await page.goto('file://' + filePath, { waitUntil: 'networkidle0', timeout: 30000 });

  // Wait for fonts to load
  await new Promise(r => setTimeout(r, 2000));

  await page.pdf({
    path: path.resolve(__dirname, 'EPL_Prediction_Slides.pdf'),
    width: '1280px',
    height: '720px',
    printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    preferCSSPageSize: true,
  });

  console.log('PDF saved: EPL_Prediction_Slides.pdf');
  await browser.close();
})();
