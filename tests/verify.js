/* Browser verification for the virtualized wide table. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

const URL = 'http://127.0.0.1:8148/index.html';
const columns = require('../fixtures/columns.json');
const rows = require('../fixtures/rows.json');
const ROW_H = columns.row_height_px;
const HEADER_H = 32;
const PINNED_W = columns.columns.filter(c => c.pinned).reduce((s, c) => s + c.width_px, 0);

const results = [];
function check(name, fn) {
  return Promise.resolve().then(fn).then(
    () => { results.push({ name, ok: true }); console.log(`PASS  ${name}`); },
    (err) => { results.push({ name, ok: false, err }); console.log(`FAIL  ${name}\n      ${err.message}`); },
  );
}

(async () => {
  const browser = await chromium.launch({
    executablePath: '/usr/bin/chromium',
    headless: true,
    args: ['--no-sandbox'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') pageErrors.push(msg.text());
  });

  await page.goto(URL);
  await page.waitForFunction(() => window.__wideTable && document.querySelector('.cell'));
  await page.locator('#grid').focus();

  const stats = () => page.evaluate(() => window.__wideTable.getStats());
  const activeInfo = () => page.evaluate(() => window.__wideTable.getActive());
  const gridBox = async () => page.locator('#grid').boundingBox();
  const rectOf = async (id) => page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  }, id);

  // ------------------------------------------------------------------
  await check('grid semantics and counts', async () => {
    assert.equal(await page.getAttribute('#grid', 'role'), 'grid');
    assert.equal(await page.getAttribute('#grid', 'aria-rowcount'), String(rows.length + 1));
    assert.equal(await page.getAttribute('#grid', 'aria-colcount'), String(columns.columns.length));
    assert.equal(await page.getAttribute('#grid', 'aria-readonly'), 'true');
    assert.equal(await page.getAttribute('#grid', 'tabindex'), '0');
  });

  await check('header cells use columnheader + aria-colindex', async () => {
    const headers = await page.$$eval('.header-cell', (nodes) => nodes.map((n) => ({
      col: n.getAttribute('aria-colindex'),
      label: n.getAttribute('aria-label'),
      pinned: n.classList.contains('pinned'),
      id: n.dataset.colId,
    })));
    assert.ok(headers.length >= 3);
    for (const h of headers) {
      const idx = Number(h.col) - 1;
      assert.equal(h.id, columns.columns[idx].id);
      assert.ok(h.label.startsWith(columns.columns[idx].label));
    }
    assert.deepEqual(headers.filter(h => h.pinned).map(h => h.id), ['c00', 'c01']);
  });

  await check('row semantics: rowindex over full table incl. header', async () => {
    const rowMeta = await page.$$eval('.row', (nodes) => nodes.map((n) => ({
      rowindex: Number(n.getAttribute('aria-rowindex')),
      rowId: n.dataset.rowId,
      cells: [...n.querySelectorAll('.cell')].map((c) => Number(c.getAttribute('aria-colindex'))),
    })));
    for (const r of rowMeta) {
      const dataIdx = r.rowindex - 2;
      assert.equal(rows[dataIdx].row_id, r.rowId);
      for (const colindex of r.cells) {
        assert.ok(colindex >= 1 && colindex <= 64);
      }
      // each column index appears at most once (no duplicate pinned copy)
      assert.equal(new Set(r.cells).size, r.cells.length, 'duplicate cells within a row');
    }
  });

  await check('each logical row has exactly one DOM cell per column index', async () => {
    const dup = await page.evaluate(() => {
      const out = [];
      for (const row of document.querySelectorAll('.row')) {
        const seen = new Map();
        for (const c of row.querySelectorAll('.cell')) {
          const k = c.dataset.colIndex;
          seen.set(k, (seen.get(k) || 0) + 1);
        }
        for (const [k, n] of seen) if (n > 1) out.push([row.dataset.rowId, k, n]);
      }
      return out;
    });
    assert.deepEqual(dup, []);
    // no second table/grid copy
    assert.equal(await page.locator('[role="grid"]').count(), 1);
  });

  await check('pinned columns stay aligned when scrolling horizontally', async () => {
    const before = await gridBox();
    await page.evaluate(() => { document.querySelector('#grid').scrollLeft = 1400; });
    await page.waitForTimeout(80);
    const pinned = await page.evaluate(() => {
      const grid = document.querySelector('#grid');
      const gb = grid.getBoundingClientRect();
      const cells = [...document.querySelectorAll('.row .cell.pinned')];
      return cells.map((c) => ({ id: c.dataset.colId, x: Math.round(c.getBoundingClientRect().x - gb.x), w: c.getBoundingClientRect().width }));
    });
    const byId = {};
    for (const c of pinned) byId[c.id] = byId[c.id] || [];
    for (const c of pinned) byId[c.id].push(c.x);
    assert.deepEqual([...new Set(byId.c00)].sort((a, b) => a - b), [0]);
    assert.deepEqual([...new Set(byId.c01)].sort((a, b) => a - b), [120]);
    // Pinned cells clip/overlay: their right edge is at pinned width.
    const maxRight = Math.max(...pinned.map(c => c.x + c.w));
    assert.ok(maxRight <= PINNED_W + 1, `pinned layer width ${maxRight}`);
    // header pinned too
    const hx = await page.evaluate(() => {
      const grid = document.querySelector('#grid');
      const gb = grid.getBoundingClientRect();
      return [...document.querySelectorAll('.header-cell.pinned')].map(h => Math.round(h.getBoundingClientRect().x - gb.x));
    });
    assert.deepEqual(hx, [0, 120]);
    assert.ok(before);
  });

  await check('scrolling horizontally actually swaps scrollable cell DOM', async () => {
    const idsBefore = await page.$$eval('.row .cell:not(.pinned)', ns => ns.map(n => n.id));
    await page.evaluate(() => { document.querySelector('#grid').scrollLeft = 4000; });
    await page.waitForTimeout(80);
    const idsAfter = await page.$$eval('.row .cell:not(.pinned)', ns => ns.map(n => n.id));
    const overlap = idsAfter.filter(id => idsBefore.includes(id));
    assert.ok(overlap.length < idsBefore.length / 2, 'non-pinned cells were not recycled');
    assert.ok(idsBefore.length < 64 * 20, 'all columns mounted');
  });

  await check('Ctrl+End jumps to last cell; real DOM and JSON agree', async () => {
    await page.locator('#grid').focus();
    await page.evaluate(() => { const g = document.querySelector('#grid'); g.scrollTop = 0; g.scrollLeft = 0; });
    await page.waitForTimeout(50);
    await page.keyboard.press('Control+End');
    await page.waitForTimeout(150);
    const info = await activeInfo();
    assert.deepEqual(
      { rowIndex: info.rowIndex, colIndex: info.colIndex, rowId: info.rowId, colId: info.colId, value: info.value },
      {
        rowIndex: rows.length - 1,
        colIndex: columns.columns.length - 1,
        rowId: rows.at(-1).row_id,
        colId: columns.columns.at(-1).id,
        value: rows.at(-1).cells[columns.columns.at(-1).id],
      },
    );
    const s = await stats();
    assert.equal(s.activeId, `cell-${rows.at(-1).row_id}-${columns.columns.at(-1).id}`);
    assert.equal(await page.getAttribute('#grid', 'aria-activedescendant'), s.activeId);
    assert.equal(await page.getAttribute('#' + s.activeId, 'role'), 'gridcell');
    // focus stayed on the grid (not body)
    assert.equal(await page.evaluate(() => document.activeElement.id), 'grid');
    // last cell visible in viewport geometry
    const cell = await rectOf('#' + s.activeId);
    const gb = await gridBox();
    assert.ok(cell.x >= gb.x && cell.x + Math.min(cell.w, 60) <= gb.x + gb.width + 1, 'last cell horizontally visible');
    assert.ok(cell.y >= gb.y && cell.y + cell.h <= gb.y + gb.height + 1, 'last cell vertically visible');
  });

  await check('Home / End / Ctrl+Home', async () => {
    await page.keyboard.press('Home');
    let i = await activeInfo();
    assert.equal(i.colIndex, 0);
    assert.equal(i.rowIndex, rows.length - 1);
    await page.keyboard.press('End');
    i = await activeInfo();
    assert.equal(i.colIndex, 63);
    await page.keyboard.press('Control+Home');
    i = await activeInfo();
    assert.deepEqual([i.rowIndex, i.colIndex], [0, 0]);
    const s = await stats();
    assert.equal(s.activeId, `cell-${rows[0].row_id}-c00`);
    assert.equal(s.scrollTop, 0);
    assert.equal(s.scrollLeft, 0);
  });

  await check('arrow navigation crossing unrendered region mounts before focus move', async () => {
    const s0 = await stats();
    const r0 = s0.active.rowIndex;
    // Jump many rows down with PageDown five times, check every stop mounted.
    for (let k = 0; k < 5; k += 1) {
      await page.keyboard.press('PageDown');
      const s = await stats();
      const aid = s.activeId;
      assert.ok(aid, 'active descendant set');
      assert.equal(await page.locator('#' + aid).count(), 1, 'active cell is mounted once');
      assert.equal(await page.getAttribute('#' + aid, 'aria-colindex'), String(s.active.colIndex + 1));
    }
    const s1 = await stats();
    assert.ok(s1.active.rowIndex > r0);
    // Right arrow across to a far column, then down; all mounted.
    for (let k = 0; k < 8; k += 1) await page.keyboard.press('ArrowRight');
    const s2 = await stats();
    assert.equal(await page.locator('#' + s2.activeId).count(), 1);
    await page.keyboard.press('ArrowDown');
    const s3 = await stats();
    assert.equal(await page.locator('#' + s3.activeId).count(), 1);
  });

  await check('DOM stays bounded after visiting many rows and columns', async () => {
    const before = await stats();
    // sweep down 500 rows in chunks with End to force column sweeps repeatedly
    for (let k = 0; k < 40; k += 1) {
      await page.keyboard.press('PageDown');
      if (k % 5 === 0) { await page.keyboard.press('End'); await page.keyboard.press('Home'); }
    }
    await page.keyboard.press('Control+End');
    await page.keyboard.press('Control+Home');
    await page.waitForTimeout(100);
    const after = await stats();
    const dom = await page.evaluate(() => ({
      rows: document.querySelectorAll('.row').length,
      cells: document.querySelectorAll('.cell').length,
      headers: document.querySelectorAll('.header-cell').length,
    }));
    assert.equal(dom.rows, after.mountedRows);
    assert.equal(dom.cells + dom.headers, after.mountedCells);
    assert.ok(dom.rows < 30, `rows bounded: ${dom.rows}`);
    assert.ok(dom.cells < 900, `cells bounded: ${dom.cells}`);
    assert.ok(after.maxMountedCells < 1200, `peak cells bounded: ${after.maxMountedCells}`);
    // no growth relative to a fresh view at the same position
    assert.ok(after.mountedCells <= before.maxMountedCells + 80, 'no unbounded growth');
  });

  await check('DOM order of cells stays logical after horizontal recycling', async () => {
    await page.evaluate(() => { document.querySelector('#grid').scrollLeft = 0; });
    await page.waitForTimeout(80);
    await page.evaluate(() => { document.querySelector('#grid').scrollLeft = 5000; });
    await page.waitForTimeout(80);
    await page.evaluate(() => { document.querySelector('#grid').scrollLeft = 0; });
    await page.waitForTimeout(80);
    const orders = await page.evaluate(() => [...document.querySelectorAll('.row')].map((row) => ({
      rowId: row.dataset.rowId,
      cols: [...row.querySelectorAll('.cell')].map((c) => Number(c.dataset.colIndex)),
      ids: [...row.querySelectorAll('.cell')].map((c) => c.id),
    })));
    assert.ok(orders.length > 0);
    for (const o of orders) {
      const sorted = [...o.cols].sort((a, b) => a - b);
      assert.deepEqual(o.cols, sorted, `row ${o.rowId} cell DOM order is not logical`);
      assert.equal(new Set(o.ids).size, o.ids.length);
    }
    const headerCols = await page.$$eval('.header-cell', ns => ns.map(n => Number(n.dataset.colIndex)));
    assert.deepEqual(headerCols, [...headerCols].sort((a, b) => a - b));
  });

  await check('null / empty / false / zero are distinguishable', async () => {
    // row 0: c03 null, c05 zero, c09 false ; row 1: c06 empty
    const grab = async (rowIndex, colIndex) => page.evaluate(({ rowIndex, colIndex }) => {
      const row = document.querySelector(`.row[data-row-index="${rowIndex}"]`);
      const cell = row && row.querySelector(`.cell[data-col-index="${colIndex}"]`);
      return cell && {
        text: cell.textContent,
        aria: cell.getAttribute('aria-label'),
        cls: cell.className,
        title: cell.title,
      };
    }, { rowIndex, colIndex });

    const go = async (rowIndex, colIndex) => {
      await page.keyboard.press('Control+Home');
      for (let i = 0; i < rowIndex; i += 1) await page.keyboard.press('ArrowDown');
      for (let i = 0; i < colIndex; i += 1) await page.keyboard.press('ArrowRight');
      const info = await activeInfo();
      assert.deepEqual([info.rowIndex, info.colIndex], [rowIndex, colIndex]);
    };
    await page.locator('#grid').focus();

    await go(0, 3);
    const nullCell = await grab(0, 3);
    assert.ok(nullCell.text.includes('null'));
    assert.ok(nullCell.cls.includes('value-null'));
    assert.ok(nullCell.aria.includes('空值'));

    await go(0, 5);
    const zeroCell = await grab(0, 5);
    assert.equal(zeroCell.text, '0');
    assert.ok(zeroCell.aria.includes('0'));

    await go(0, 9);
    const falseCell = await grab(0, 9);
    assert.equal(falseCell.text, 'false');
    assert.ok(falseCell.cls.includes('value-false'));

    await go(1, 6);
    const emptyCell = await grab(1, 6);
    assert.ok(emptyCell.text.includes('空字符串'));
    assert.ok(emptyCell.cls.includes('value-empty'));

    const texts = new Set([nullCell.text, zeroCell.text, falseCell.text, emptyCell.text]);
    assert.equal(texts.size, 4, 'four distinct presentations');
    const arias = new Set([nullCell.aria, zeroCell.aria, falseCell.aria, emptyCell.aria]);
    assert.equal(arias.size, 4, 'four distinct announcements');
  });

  await check('long text: full value in aria-label/title/status, not omitted', async () => {
    // row 0 col 10 is the longest value in row 0 (58 chars)
    await page.locator('#grid').focus();
    await page.keyboard.press('Control+Home');
    for (let i = 0; i < 10; i += 1) await page.keyboard.press('ArrowRight');
    const col = columns.columns[10];
    const full = String(rows[0].cells[col.id]);
    assert.ok(full.length > 30);
    const info = await activeInfo();
    assert.equal(info.colIndex, 10);
    assert.equal(info.rowIndex, 0);
    const aria = await page.getAttribute('#' + (await stats()).activeId, 'aria-label');
    assert.ok(aria.includes(full), 'aria-label contains full value');
    const status = await page.locator('#status').textContent();
    assert.ok(status.includes(full), 'status contains full value');
    const inspector = await page.locator('#active-value').textContent();
    assert.equal(inspector, full);
    const title = await page.getAttribute('#' + (await stats()).activeId, 'title');
    assert.ok(title.includes(full));
  });

  await check('container resize keeps active cell mounted and focus on grid', async () => {
    const before = await stats();
    await page.setViewportSize({ width: 760, height: 520 });
    await page.waitForTimeout(250);
    let s = await stats();
    assert.equal(await page.evaluate(() => document.activeElement.id), 'grid');
    assert.equal(await page.getAttribute('#grid', 'aria-activedescendant'), s.activeId);
    assert.equal(await page.locator('#' + s.activeId).count(), 1);
    assert.deepEqual(s.active, before.active);
    // scroll + resize combos
    await page.evaluate(() => { const g = document.querySelector('#grid'); g.scrollTop = 9000; g.scrollLeft = 3000; });
    await page.waitForTimeout(120);
    await page.setViewportSize({ width: 1100, height: 700 });
    await page.waitForTimeout(250);
    s = await stats();
    assert.equal(await page.evaluate(() => document.activeElement.id), 'grid');
    assert.equal(await page.locator('#' + s.activeId).count(), 1);
    assert.equal(await page.getAttribute('#grid', 'aria-activedescendant'), s.activeId);
  });

  await check('no focus loss to body through rapid keys and recycles', async () => {
    await page.locator('#grid').focus();
    for (let k = 0; k < 60; k += 1) {
      await page.keyboard.press(k % 2 ? 'ArrowDown' : 'ArrowRight');
    }
    await page.evaluate(() => { const g = document.querySelector('#grid'); g.scrollTop = 5000; g.scrollLeft = 2000; });
    await page.waitForTimeout(80);
    for (let k = 0; k < 30; k += 1) await page.keyboard.press(k % 3 ? 'PageUp' : 'ArrowLeft');
    const s = await stats();
    // The active cell may be forced off-viewport; it must still be mounted and
    // the active descendant must resolve to a real gridcell.
    assert.equal(await page.evaluate(() => document.activeElement.id), 'grid');
    assert.equal(await page.locator('#' + s.activeId).count(), 1);
  });

  await check('Tab leaves the grid; shift+Tab returns (no trap)', async () => {
    await page.locator('#grid').focus();
    await page.keyboard.press('Tab');
    const afterTab = await page.evaluate(() => ({
      id: document.activeElement.id,
      tag: document.activeElement.tagName,
      inGrid: document.querySelector('#grid').contains(document.activeElement),
    }));
    assert.equal(afterTab.inGrid, false, 'focus left the grid');
    assert.notEqual(afterTab.tag, 'BODY');
    await page.keyboard.press('Shift+Tab');
    const back = await page.evaluate(() => ({ id: document.activeElement.id }));
    assert.equal(back.id, 'grid');
    // Tab forward again just to be safe
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => document.activeElement.id === 'grid'), false);
  });

  await check('accessibility snapshot has one grid and no duplicate pinned cells', async () => {
    await page.locator('#grid').focus();
    await page.keyboard.press('Control+Home');
    const snap = await page.locator('#grid').ariaSnapshot();
    assert.match(snap, /^- grid "只读数据表"/m, 'grid role with accessible name');
    assert.match(snap, /- row /, 'row role present');
    assert.match(snap, /- columnheader "记录号/, 'columnheader present');
    assert.match(snap, /- gridcell /, 'gridcell role present');
    assert.equal((snap.match(/- grid "/g) || []).length, 1, 'exactly one grid');
    // Pinned cells exist exactly once per mounted row in the accessibility tree.
    const domPinned = await page.evaluate(() => {
      const row = document.querySelector('.row[aria-rowindex="2"]');
      const cells = [...row.querySelectorAll('.cell')];
      return {
        pinned: cells.filter(c => c.classList.contains('pinned')).length,
        ids: cells.map(c => c.id),
        names: cells.map(c => c.getAttribute('aria-label')),
      };
    });
    assert.equal(domPinned.pinned, 2, 'two pinned cells in one row');
    assert.equal(new Set(domPinned.ids).size, domPinned.ids.length);
    assert.equal(new Set(domPinned.names).size, domPinned.names.length);
    // aria snapshot row for the first data row lists each columnheader/gridcell name once
    const firstRowNames = await page.evaluate(() => {
      const row = document.querySelector('.row[aria-rowindex="2"]');
      return [...row.querySelectorAll('.cell')].map(c => c.getAttribute('aria-label'));
    });
    assert.ok(firstRowNames[0].includes('记录号'), 'pinned first cell is column c00');
  });

  await check('scroll position → logical cell mapping is consistent (pinned vs body)', async () => {
    // Mounted rows must all satisfy renderedY = header + index*rowH - scrollTop,
    // with pinned cells sharing the exact same y as scrollable cells.
    await page.locator('#grid').focus();
    await page.evaluate(({ rowH }) => {
      const g = document.querySelector('#grid');
      g.scrollTop = rowH * 137;
      g.scrollLeft = 1234;
    }, { rowH: ROW_H });

    let settled = null;
    for (let attempt = 0; attempt < 40 && !settled; attempt += 1) {
      await page.waitForTimeout(50);
      settled = await page.evaluate(({ rowH, headerH }) => {
        const grid = document.querySelector('#grid');
        const gb = grid.getBoundingClientRect();
        const rows = [...document.querySelectorAll('.row')];
        if (!rows.length) return null;
        let ok = true;
        for (const row of rows) {
          const index = Number(row.dataset.rowIndex);
          const expected = headerH + index * rowH - grid.scrollTop;
          if (Math.abs(row.getBoundingClientRect().y - gb.y - expected) > 1) ok = false;
        }
        if (!ok) return null;
        const firstAtHeader = rows
          .map((r) => ({ i: Number(r.dataset.rowIndex), y: r.getBoundingClientRect().y - gb.y }))
          .filter((r) => Math.abs(r.y - headerH) <= 1)
          .sort((a, b) => a.y - b.y)[0];
        const cells = [];
        for (const c of document.querySelectorAll('.row .cell')) {
          const rowEl = c.parentElement;
          const index = Number(c.dataset.colIndex);
          cells.push({
            rowIndex: Number(rowEl.dataset.rowIndex),
            colIndex: index,
            left: parseFloat(c.style.left),
            y: c.getBoundingClientRect().y - gb.y,
            rowY: rowEl.getBoundingClientRect().y - gb.y,
            pinned: c.classList.contains('pinned'),
            id: c.id,
            rowId: String(c.dataset.rowId),
          });
        }
        return {
          scrollTop: Math.round(grid.scrollTop),
          scrollLeft: Math.round(grid.scrollLeft),
          rowCount: rows.length,
          firstAtHeader: firstAtHeader && firstAtHeader.i,
          implied: Math.floor(grid.scrollTop / rowH),
          cells,
        };
      }, { rowH: ROW_H, headerH: HEADER_H });
      if (!settled) continue;
      // Reject a frame where cell y does not match its row y yet.
      if (settled.cells.some(c => Math.abs(c.y - c.rowY) > 1)) settled = null;
    }
    assert.ok(settled, 'mounted rows settled into a consistent mapping');
    assert.ok(settled.rowCount > 0);
    assert.ok(settled.firstAtHeader !== undefined, 'a row sits exactly below the header');
    assert.equal(settled.firstAtHeader, settled.implied,
      'row at header edge equals floor(scrollTop / rowH)');

    for (const c of settled.cells) {
      const expectedLeft = columns.columns
        .slice(0, c.colIndex)
        .reduce((sum, col) => sum + col.width_px, 0);
      assert.equal(c.left, expectedLeft, 'cell x follows configured column widths');
      assert.equal(c.id, `cell-${c.rowId}-${columns.columns[c.colIndex].id}`, 'stable cell id');
      assert.ok(rows[c.rowIndex], 'cell row exists');
      assert.equal(rows[c.rowIndex].row_id, c.rowId, 'cell row id matches JSON');
    }
    // Pinned and scrollable cells in the same row share one y.
    const perRowYs = new Map();
    for (const c of settled.cells) {
      if (!perRowYs.has(c.rowIndex)) perRowYs.set(c.rowIndex, new Set());
      perRowYs.get(c.rowIndex).add(Math.round(c.y * 100) / 100);
    }
    for (const [rowIndex, set] of perRowYs) {
      assert.equal(set.size, 1, `row ${rowIndex} has one vertical position: ${[...set]}`);
    }
  });

  await check('wheel scrolling then keyboard arrow keeps active descendant valid', async () => {
    await page.locator('#grid').focus();
    const gb = await gridBox();
    await page.mouse.move(gb.x + gb.width / 2, gb.y + gb.height / 2);
    await page.mouse.wheel(0, 1200);
    await page.waitForTimeout(150);
    await page.mouse.wheel(1000, 0);
    await page.waitForTimeout(150);
    const s = await stats();
    assert.equal(await page.getAttribute('#grid', 'aria-activedescendant'), s.activeId);
    assert.equal(await page.locator('#' + s.activeId).count(), 1);
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('ArrowRight');
    const s2 = await stats();
    assert.equal(await page.getAttribute('#grid', 'aria-activedescendant'), s2.activeId);
    assert.equal(await page.evaluate(() => document.activeElement.id), 'grid');
    assert.equal(await page.locator('#' + s2.activeId).count(), 1);
  });

  await check('page and row header indices match full table after deep scroll', async () => {
    // Keyboard-driven end-of-table jump, then let the scroll settle before
    // inspecting the mounted set.
    await page.locator('#grid').focus();
    await page.keyboard.press('Control+End');
    await page.waitForFunction(() => {
      const g = document.querySelector('#grid');
      return g.scrollTop === g.scrollHeight - g.clientHeight &&
        document.querySelector('.row[data-row-index="799"]') !== null;
    }, null, { timeout: 3000 });
    const s = await stats();
    await page.waitForFunction(() => window.__wideTable.getStats().scrollTop ===
      document.querySelector('#grid').scrollHeight - document.querySelector('#grid').clientHeight);
    const maxRowIndex = await page.evaluate(() => Math.max(...[...document.querySelectorAll('.row')].map(r => Number(r.dataset.rowIndex))));
    assert.equal(maxRowIndex, rows.length - 1);
    // last row's aria-rowindex accounts for the header row (799 + 2 = 801)
    const lastRowIndex = await page.getAttribute('.row[data-row-index="799"]', 'aria-rowindex');
    assert.equal(lastRowIndex, String(rows.length + 1));
    // The last row's last scrollable column is present with its true value.
    const lastCell = await page.evaluate(() => {
      const row = document.querySelector('.row[data-row-index="799"]');
      const cell = row.querySelector('.cell[data-col-index="63"]');
      return cell && { text: cell.textContent, aria: cell.getAttribute('aria-label') };
    });
    assert.ok(lastCell, 'last cell mounted');
    assert.equal(lastCell.text, String(rows[799].cells.c63));
    assert.ok(lastCell.aria.endsWith('类型 boolean'));
    assert.equal(s.rowCount, 800);
  });

  await check('no page errors', async () => {
    assert.deepEqual(pageErrors, []);
  });

  await browser.close();

  const failed = results.filter(r => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  if (failed.length) {
    for (const f of failed) console.log(`\n${f.name}\n${f.err && f.err.stack}`);
    process.exit(1);
  }
})().catch((e) => { console.error(e); process.exit(1); });
