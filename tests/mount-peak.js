/* Measure the peak number of mounted cells across viewport sizes and scroll
   patterns. Also checks pinned/body row alignment at fractional offsets. */
const { chromium } = require('playwright');

const URL = 'http://127.0.0.1:8148/index.html';

(async () => {
  const browser = await chromium.launch({
    executablePath: '/usr/bin/chromium',
    headless: true,
    args: ['--no-sandbox'],
  });

  const sizes = [
    [800, 600], [1024, 700], [1280, 800], [1440, 900], [1920, 1080],
  ];
  const report = [];

  for (const [width, height] of sizes) {
    const page = await browser.newPage({ viewport: { width, height } });
    await page.goto(URL);
    await page.waitForFunction(() => window.__wideTable && document.querySelector('.cell'));
    await page.locator('#grid').focus();

    let peak = 0;
    let peakRows = 0;
    const positions = [
      [0, 0], [150, 1500], [1234, 3000], [9000, 7000],
      [25094, 7433], [5000, 4000], [0, 7433], [25094, 0],
    ];
    for (const [top, left] of positions) {
      await page.evaluate(({ top, left }) => {
        const g = document.querySelector('#grid');
        g.scrollTop = top;
        g.scrollLeft = left;
      }, { top, left });
      await page.waitForTimeout(80);
      const s = await page.evaluate(() => window.__wideTable.getStats());
      peak = Math.max(peak, s.mountedCells);
      peakRows = Math.max(peakRows, s.mountedRows);
    }

    // Fractional scroll alignment: every mounted cell must share its row's top.
    await page.evaluate(() => {
      const g = document.querySelector('#grid');
      g.scrollTop = 7777.5;
      g.scrollLeft = 2222.25;
    });
    await page.waitForTimeout(120);
    const alignment = await page.evaluate(() => {
      const rows = new Map([...document.querySelectorAll('.row')].map(r => [r.dataset.rowIndex, r.getBoundingClientRect().top]));
      let maxDelta = 0;
      let pinnedCount = 0;
      for (const c of document.querySelectorAll('.row .cell')) {
        const delta = Math.abs(c.getBoundingClientRect().top - rows.get(c.parentElement.dataset.rowIndex));
        maxDelta = Math.max(maxDelta, delta);
        if (c.classList.contains('pinned')) pinnedCount += 1;
      }
      const gb = document.querySelector('#grid').getBoundingClientRect();
      const pinned = [...document.querySelectorAll('.row .cell.pinned')].map(c => c.getBoundingClientRect().x - gb.x);
      return {
        maxDelta,
        pinnedCount,
        pinnedX: [...new Set(pinned.map(x => Math.round(x)))].sort((a, b) => a - b),
      };
    });

    const dom = await page.evaluate(() => ({
      rows: document.querySelectorAll('.row').length,
      cells: document.querySelectorAll('.row .cell').length,
      headers: document.querySelectorAll('.header-cell').length,
    }));

    report.push({
      viewport: `${width}x${height}`,
      peakMountedCells: peak,
      peakMountedRows: peakRows,
      domRows: dom.rows,
      domDataCells: dom.cells,
      domHeaderCells: dom.headers,
      rowTopDelta: alignment.maxDelta,
      pinnedX: alignment.pinnedX,
    });
    await page.close();
  }

  console.log(JSON.stringify(report, null, 2));
  const worst = Math.max(...report.map(r => r.peakMountedCells));
  const worstRows = Math.max(...report.map(r => r.peakMountedRows));
  console.log(`\nMAX mounted data+header cells across viewports: ${worst}`);
  console.log(`MAX mounted rows across viewports: ${worstRows}`);
  const badAlign = report.filter(r => r.rowTopDelta > 0.01);
  if (badAlign.length) {
    console.error('Alignment failures', badAlign);
    process.exit(1);
  }
  const badPins = report.filter(r => JSON.stringify(r.pinnedX) !== JSON.stringify([0, 120]));
  if (badPins.length) {
    console.error('Pinned X drift', badPins);
    process.exit(1);
  }
  console.log('PASS  pinned columns at x=0,120 and rows aligned at fractional scroll');
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
