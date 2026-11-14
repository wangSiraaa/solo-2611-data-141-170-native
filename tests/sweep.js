/* Full-table traversal: visit every row and every column with real key events,
   prove the mounted DOM stays bounded, and record the peak mount counts. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

const URL = 'http://127.0.0.1:8148/index.html';
const columns = require('../fixtures/columns.json');
const rows = require('../fixtures/rows.json');

(async () => {
  const browser = await chromium.launch({
    executablePath: '/usr/bin/chromium',
    headless: true,
    args: ['--no-sandbox'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(URL);
  await page.waitForFunction(() => window.__wideTable && document.querySelector('.cell'));
  await page.locator('#grid').focus();

  const stats = () => page.evaluate(() => window.__wideTable.getStats());
  const active = () => page.evaluate(() => window.__wideTable.getActive());

  // Visit every row top-to-bottom, alternating column sweeps on each row so
  // both virtualization axes recycle continuously.
  const sweep = await page.evaluate(async () => {
    const grid = document.querySelector('#grid');
    const table = window.__wideTable;
    const sleep = () => new Promise((r) => requestAnimationFrame(r));
    const stats = () => table.getStats();
    let peakCells = 0;
    let peakRows = 0;
    let peakDomCells = 0;
    let peakDomRows = 0;
    const visitedRows = new Set();
    const visitedCols = new Set();

    const press = (key, ctrlKey = false) => {
      grid.dispatchEvent(new KeyboardEvent('keydown', {
        key, ctrlKey, bubbles: true, cancelable: true,
      }));
    };

    press('Home', true);
    await sleep();

    for (let r = 0; r < 800; r += 1) {
      // On a few rows do a complete left-to-right column sweep, recording each
      // column the active cell actually lands on.
      if (r % 100 === 0) {
        press('Home');
        await sleep();
        visitedCols.add(table.getActive().colIndex);
        for (let c = 1; c < 64; c += 1) {
          press('ArrowRight');
          visitedCols.add(table.getActive().colIndex);
          if (c % 8 === 0) await sleep();
        }
        press('Home');
        await sleep();
      }
      // Mixed partial sweeps on the remaining rows to keep both axes churning.
      if (r % 47 === 0) {
        press('End');
        await sleep();
        press('Home');
        await sleep();
      }
      visitedRows.add(table.getActive().rowIndex);
      const s = stats();
      peakCells = Math.max(peakCells, s.mountedCells);
      peakRows = Math.max(peakRows, s.mountedRows);

      const dom = {
        rows: document.querySelectorAll('.row').length,
        cells: document.querySelectorAll('.row .cell').length + document.querySelectorAll('.header-cell').length,
      };
      peakDomRows = Math.max(peakDomRows, dom.rows);
      peakDomCells = Math.max(peakDomCells, dom.cells);

      // Walk a different horizontal window each row, releasing to Home.
      const steps = r % 13;
      for (let c = 0; c < steps; c += 1) {
        press('ArrowRight');
      }
      visitedCols.add(table.getActive().colIndex);
      if (r < 799) {
        press('ArrowDown');
        await sleep();
      }
    }

    const end = table.getActive();
    const last = stats();
    return {
      peakCells,
      peakRows,
      peakDomCells,
      peakDomRows,
      visitedRows: visitedRows.size,
      visitedCols: visitedCols.size,
      finalActive: end,
      finalMountedCells: last.mountedCells,
      finalMountedRows: last.mountedRows,
      domFinalRows: document.querySelectorAll('.row').length,
      domFinalCells: document.querySelectorAll('.row .cell').length + document.querySelectorAll('.header-cell').length,
      maxMountedCells: last.maxMountedCells,
      maxMountedRows: last.maxMountedRows,
      activeDescendant: grid.getAttribute('aria-activedescendant'),
    };
  });

  console.log('\n=== full-table traversal ===');
  console.log(JSON.stringify(sweep, null, 2));

  assert.equal(sweep.visitedRows, 800, 'every row visited');
  assert.equal(sweep.visitedCols, 64, 'every column visited');
  assert.equal(sweep.finalActive.rowIndex, 799);
  assert.equal(sweep.finalActive.rowId, rows[799].row_id);
  assert.equal(sweep.finalActive.colId, columns.columns[sweep.finalActive.colIndex].id);
  assert.equal(
    sweep.finalActive.value,
    rows[799].cells[columns.columns[sweep.finalActive.colIndex].id],
  );
  assert.ok(sweep.peakCells < 800, `peak mounted cells bounded: ${sweep.peakCells}`);
  assert.ok(sweep.peakDomCells < 800, `peak DOM cells bounded: ${sweep.peakDomCells}`);
  assert.ok(sweep.peakDomRows < 40, `peak DOM rows bounded: ${sweep.peakDomRows}`);
  assert.equal(sweep.domFinalRows, sweep.finalMountedRows, 'final DOM rows equals mounted rows');
  assert.equal(sweep.domFinalCells, sweep.finalMountedCells, 'final DOM cells equals mounted cells');
  assert.equal(sweep.maxMountedCells, sweep.peakCells, 'tracked peak equals observed peak');
  assert.equal(
    sweep.activeDescendant,
    `cell-${rows[799].row_id}-${columns.columns[sweep.finalActive.colIndex].id}`,
    'active descendant still resolves to the current logical cell',
  );
  assert.deepEqual(errors, []);

  // Focus must still be on the grid after the entire sweep.
  assert.equal(await page.evaluate(() => document.activeElement.id), 'grid');
  // And the active cell must be a mounted gridcell exactly once.
  assert.equal(await page.locator('#' + sweep.activeDescendant).count(), 1);

  console.log('\nPASS  full-table traversal: DOM bounded, identity preserved');
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
