/**
 * Read-only virtualized wide table.
 *
 * One scroll viewport (#grid) owns a single canvas:
 *   - one sticky header row (pinned cells are position:sticky; left)
 *   - one absolutely positioned DOM row per *mounted* logical row
 *   - each row holds its pinned cells plus only the scrollable cells currently
 *     in the horizontal window (+overscan)
 *
 * There is never a second copy of the row/table. Pinned columns are the same
 * DOM elements that would otherwise scroll; CSS sticky keeps them in place.
 *
 * Focus stays on the role=grid container; aria-activedescendant names the
 * current cell. Cell ids are built from stable row_id + column id, and mounted
 * DOM is keyed by (rowIndex, colIndex) so a reused DOM node can never change
 * logical identity while mounted.
 */

const ROW_OVERSCAN = 3;
const COL_OVERSCAN = 2;

const els = {
  grid: document.getElementById('grid'),
  status: document.getElementById('status'),
};

const state = {
  columns: [],
  rows: [],
  rowHeight: 32,
  headerHeight: 32,
  colStarts: [],        // canvas x of every column
  pinnedCols: [],       // column indices, in table order
  scrollableCols: [],   // column indices, in table order
  pinnedOffsets: new Map(), // colIndex -> left inset inside the pinned layer
  pinnedWidth: 0,
  totalWidth: 0,
  totalHeight: 0,
  active: { rowIndex: 0, colIndex: 0 },
  mountedRows: new Map(),      // rowIndex -> { el, cells: Map<colIndex, el> }
  mountedHeader: new Map(),    // colIndex -> el
  headerEl: null,
  canvas: null,
  activeEl: null,
  maxMountedCells: 0,
  maxMountedRows: 0,
  renderCount: 0,
};

/* ------------------------------------------------------------------ */
/* Loading                                                             */
/* ------------------------------------------------------------------ */

async function load() {
  const [columnsRes, rowsRes] = await Promise.all([
    fetch('fixtures/columns.json', { cache: 'no-store' }),
    fetch('fixtures/rows.json', { cache: 'no-store' }),
  ]);
  if (!columnsRes.ok || !rowsRes.ok) {
    throw new Error(`加载失败: columns=${columnsRes.status}, rows=${rowsRes.status}`);
  }
  const colConfig = await columnsRes.json();
  const rows = await rowsRes.json();

  state.columns = colConfig.columns;
  state.rows = rows;
  state.rowHeight = colConfig.row_height_px || 32;

  // Column geometry: same cumulative starts used by header and body.
  let x = 0;
  state.colStarts = state.columns.map((col) => {
    const start = x;
    x += col.width_px;
    return start;
  });
  state.totalWidth = x;

  state.columns.forEach((col, index) => {
    if (col.pinned) state.pinnedCols.push(index);
    else state.scrollableCols.push(index);
  });
  let pinnedOffset = 0;
  for (const index of state.pinnedCols) {
    state.pinnedOffsets.set(index, pinnedOffset);
    pinnedOffset += state.columns[index].width_px;
  }
  state.pinnedWidth = pinnedOffset;
  state.headerHeight = 32;
  state.totalHeight = state.headerHeight + rows.length * state.rowHeight;

  buildShell();
  updateMounts();
  applyActive({ scroll: false });
  renderStatus();
  installEvents();
}

function buildShell() {
  const grid = els.grid;
  grid.textContent = '';
  grid.style.setProperty('--header-h', `${state.headerHeight}px`);
  grid.setAttribute('aria-rowcount', String(state.rows.length + 1)); // + header
  grid.setAttribute('aria-colcount', String(state.columns.length));

  const canvas = document.createElement('div');
  canvas.className = 'canvas';
  canvas.setAttribute('role', 'presentation');
  canvas.style.width = `${state.totalWidth}px`;
  canvas.style.height = `${state.totalHeight}px`;
  canvas.style.overflowAnchor = 'none';

  const header = document.createElement('div');
  header.className = 'header-row';
  header.setAttribute('role', 'row');
  header.setAttribute('aria-rowindex', '1');
  header.setAttribute('aria-label', '列标题行');
  canvas.appendChild(header);

  grid.appendChild(canvas);
  state.canvas = canvas;
  state.headerEl = header;
}

function installEvents() {
  const grid = els.grid;
  grid.addEventListener('focus', onFocus);
  grid.addEventListener('blur', () => grid.classList.remove('is-focused'));
  grid.addEventListener('keydown', onKeyDown);
  grid.addEventListener('pointerdown', onPointerDown);
  grid.addEventListener('scroll', onScroll, { passive: true });

  if (typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => {
      // A resize can leave the active cell outside the new scrollport. Keep it
      // visible and mounted so aria-activedescendant never dangles.
      if (state.rows.length) ensureCellVisibleInViewport();
      updateMounts();
    });
    ro.observe(grid);
  } else {
    window.addEventListener('resize', () => {
      if (state.rows.length) ensureCellVisibleInViewport();
      updateMounts();
    });
  }
}

/* ------------------------------------------------------------------ */
/* Layout / ranges                                                     */
/* ------------------------------------------------------------------ */

function computeRanges() {
  const grid = els.grid;
  const rowH = state.rowHeight;
  const headerH = state.headerHeight;
  const scrollTop = grid.scrollTop;
  const scrollLeft = grid.scrollLeft;
  const clientH = grid.clientHeight;
  const clientW = grid.clientWidth;

  const rowCount = state.rows.length;

  // Rows whose box intersects the scrollport below the sticky header.
  const firstVisibleRow = Math.max(0, Math.floor(scrollTop / rowH));
  const lastVisibleRow = Math.min(
    rowCount - 1,
    Math.ceil((scrollTop + clientH - headerH) / rowH) - 1,
  );

  const firstRow = Math.max(0, firstVisibleRow - ROW_OVERSCAN);
  const lastRow = Math.min(rowCount - 1, lastVisibleRow + ROW_OVERSCAN);

  // Scrollable columns visible to the right of the pinned overlay.
  const viewLeft = scrollLeft + state.pinnedWidth;
  const viewRight = scrollLeft + clientW;
  let firstVisibleCol = state.scrollableCols.length;
  let lastVisibleCol = -1;

  for (let i = 0; i < state.scrollableCols.length; i += 1) {
    const colIndex = state.scrollableCols[i];
    const start = state.colStarts[colIndex];
    const width = state.columns[colIndex].width_px;
    if (start + width > viewLeft) {
      firstVisibleCol = i;
      break;
    }
  }
  for (let i = firstVisibleCol; i < state.scrollableCols.length; i += 1) {
    const colIndex = state.scrollableCols[i];
    const start = state.colStarts[colIndex];
    if (start < viewRight) lastVisibleCol = i;
    else break;
  }
  if (lastVisibleCol < firstVisibleCol) lastVisibleCol = firstVisibleCol - 1;

  const firstScrollablePos = Math.max(0, firstVisibleCol - COL_OVERSCAN);
  const lastScrollablePos = Math.min(
    state.scrollableCols.length - 1,
    lastVisibleCol + COL_OVERSCAN,
  );

  const colSet = new Set(state.pinnedCols);
  for (let i = firstScrollablePos; i <= lastScrollablePos; i += 1) {
    colSet.add(state.scrollableCols[i]);
  }
  // Keep the active column mounted so aria-activedescendant never dangles.
  colSet.add(state.active.colIndex);

  const colIndices = [...colSet].sort((a, b) => a - b);

  const rowSet = new Set();
  for (let i = firstRow; i <= lastRow; i += 1) rowSet.add(i);
  rowSet.add(state.active.rowIndex);

  return {
    rowIndices: [...rowSet].sort((a, b) => a - b),
    colIndices,
    firstRow,
    lastRow,
    firstScrollablePos,
    lastScrollablePos,
  };
}

/* ------------------------------------------------------------------ */
/* Mounting / reconciliation                                           */
/* ------------------------------------------------------------------ */

function updateMounts() {
  if (!state.canvas) return;
  const { rowIndices, colIndices } = computeRanges();
  const rowSet = new Set(rowIndices);
  const colSet = new Set(colIndices);

  // Remove rows no longer in range (and all their cells).
  for (const [rowIndex, rec] of state.mountedRows) {
    if (!rowSet.has(rowIndex)) {
      rec.el.remove();
      state.mountedRows.delete(rowIndex);
    }
  }

  for (const rowIndex of rowIndices) {
    let rec = state.mountedRows.get(rowIndex);
    if (!rec) {
      const row = state.rows[rowIndex];
      const el = document.createElement('div');
      el.className = 'row';
      el.setAttribute('role', 'row');
      el.setAttribute('aria-rowindex', String(rowIndex + 2)); // header is 1
      el.setAttribute('aria-label', `第 ${rowIndex + 1} 行，行 ID ${row.row_id}`);
      el.dataset.rowIndex = String(rowIndex);
      el.dataset.rowId = row.row_id;
      el.style.top = `${state.headerHeight + rowIndex * state.rowHeight}px`;
      el.style.height = `${state.rowHeight}px`;
      el.style.width = `${state.totalWidth}px`;
      state.canvas.appendChild(el);
      rec = { el, cells: new Map() };
      state.mountedRows.set(rowIndex, rec);
    }
    reconcileRowCells(rowIndex, rec, colIndices, colSet);
  }

  reconcileHeader(colIndices, colSet);
  ensureDomOrder(
    state.canvas,
    [state.headerEl, ...rowIndices.map((i) => state.mountedRows.get(i).el)],
  );
  ensureDomOrder(
    state.headerEl,
    colIndices.map((i) => state.mountedHeader.get(i)),
  );
  refreshActiveHighlight();

  const cellCount = countMountedCells();
  state.maxMountedCells = Math.max(state.maxMountedCells, cellCount);
  state.maxMountedRows = Math.max(state.maxMountedRows, state.mountedRows.size);
  state.renderCount += 1;
  renderStatus();
}

function reconcileRowCells(rowIndex, rec, colIndices, colSet) {
  for (const [colIndex, cellEl] of rec.cells) {
    if (!colSet.has(colIndex)) {
      cellEl.remove();
      rec.cells.delete(colIndex);
    }
  }

  for (const colIndex of colIndices) {
    let cellEl = rec.cells.get(colIndex);
    if (!cellEl) {
      cellEl = createCell(rowIndex, colIndex);
      rec.cells.set(colIndex, cellEl);
      rec.el.appendChild(cellEl);
    }
    // Invariant: a mounted node keeps its fixed logical key. It is never
    // reused for another row/column.
    if (
      cellEl.dataset.rowIndex !== String(rowIndex) ||
      cellEl.dataset.colIndex !== String(colIndex)
    ) {
      throw new Error('virtual cell identity mismatch');
    }
  }

  // Recycling appends replacement cells at the end; restore logical column
  // order so the accessibility tree (and DOM inspection) never sees a row's
  // cells out of order. Absolute positioning makes this visually a no-op.
  ensureDomOrder(rec.el, colIndices.map((i) => rec.cells.get(i)));
}

function createCell(rowIndex, colIndex) {
  const col = state.columns[colIndex];
  const row = state.rows[rowIndex];
  const value = row.cells[col.id];

  const cell = document.createElement('div');
  cell.className = `cell type-${col.type}`;
  if (col.pinned) cell.classList.add('pinned');
  cell.setAttribute('role', 'gridcell');
  cell.setAttribute('aria-colindex', String(colIndex + 1));
  cell.id = cellId(row.row_id, col.id);
  cell.dataset.rowIndex = String(rowIndex);
  cell.dataset.colIndex = String(colIndex);
  cell.dataset.rowId = row.row_id;
  cell.dataset.colId = col.id;
  cell.style.left = `${state.colStarts[colIndex]}px`;
  cell.style.width = `${col.width_px}px`;
  if (col.pinned) {
    // Sticky uses `left` as its inset from the scrollport. Give the element its
    // in-flow static position with a margin so pinned cells work even if the
    // pinned set is not perfectly contiguous.
    const inset = state.pinnedOffsets.get(colIndex);
    cell.style.left = `${inset}px`;
    const inFlowOffset = state.colStarts[colIndex] - inset;
    if (inFlowOffset !== 0) cell.style.marginLeft = `${inFlowOffset}px`;
  }
  cell.title = `${col.label}（${col.id}）: ${rawText(value)}`;

  applyValuePresentation(cell, value);
  cell.setAttribute(
    'aria-label',
    cellAnnouncement(col, value, rowIndex, colIndex),
  );
  return cell;
}

function applyValuePresentation(cell, value) {
  let text = '';
  cell.classList.remove(
    'value-null', 'value-empty', 'value-false', 'value-true', 'value-number',
  );

  if (value === null) {
    text = '(null)';
    cell.classList.add('value-null');
  } else if (value === '') {
    text = '(空字符串)';
    cell.classList.add('value-empty');
  } else if (value === false) {
    text = 'false';
    cell.classList.add('value-false');
  } else if (value === true) {
    text = 'true';
    cell.classList.add('value-true');
  } else if (typeof value === 'number') {
    text = String(value);
    cell.classList.add('value-number');
  } else {
    text = String(value);
  }
  cell.textContent = text;
}

function rawText(value) {
  if (value === null) return 'null';
  if (value === '') return '空字符串';
  return String(value);
}

function spokenValue(value) {
  if (value === null) return '空值 null';
  if (value === '') return '空字符串';
  if (value === false) return 'false，布尔假';
  if (value === true) return 'true，布尔真';
  if (value === 0) return '0，数字零';
  if (typeof value === 'number') return `${value}，数字`;
  return String(value);
}

function cellAnnouncement(col, value, rowIndex, colIndex) {
  return (
    `${col.label}（列 ${col.id}），值：${spokenValue(value)}；` +
    `第 ${rowIndex + 1} 行，共 ${state.rows.length} 行；` +
    `第 ${colIndex + 1} 列，共 ${state.columns.length} 列；类型 ${col.type}`
  );
}

function cellId(rowId, colId) {
  return `cell-${rowId}-${colId}`;
}

function reconcileHeader(colIndices, colSet) {
  for (const [colIndex, el] of state.mountedHeader) {
    if (!colSet.has(colIndex)) {
      el.remove();
      state.mountedHeader.delete(colIndex);
    }
  }
  for (const colIndex of colIndices) {
    let el = state.mountedHeader.get(colIndex);
    if (!el) {
      el = createHeaderCell(colIndex);
      state.mountedHeader.set(colIndex, el);
      state.headerEl.appendChild(el);
    }
  }
}

function createHeaderCell(colIndex) {
  const col = state.columns[colIndex];
  const el = document.createElement('div');
  el.className = 'header-cell';
  if (col.pinned) el.classList.add('pinned');
  el.setAttribute('role', 'columnheader');
  el.setAttribute('aria-colindex', String(colIndex + 1));
  el.setAttribute(
    'aria-label',
    `${col.label}，列 ${col.id}，第 ${colIndex + 1} 列，共 ${state.columns.length} 列`,
  );
  el.dataset.colIndex = String(colIndex);
  el.dataset.colId = col.id;
  el.style.left = `${state.colStarts[colIndex]}px`;
  el.style.width = `${col.width_px}px`;
  if (col.pinned) {
    const inset = state.pinnedOffsets.get(colIndex);
    el.style.left = `${inset}px`;
    const inFlowOffset = state.colStarts[colIndex] - inset;
    if (inFlowOffset !== 0) el.style.marginLeft = `${inFlowOffset}px`;
  }
  el.title = `${col.label}（${col.id}）`;

  const label = document.createElement('span');
  label.className = 'header-label';
  label.textContent = col.label;
  const id = document.createElement('span');
  id.className = 'header-id';
  id.textContent = col.id;
  el.append(label, id);
  return el;
}

function countMountedCells() {
  let count = 0;
  for (const rec of state.mountedRows.values()) count += rec.cells.size;
  count += state.mountedHeader.size;
  return count;
}

/**
 * Keep the DOM child order equal to logical row/column order, and normalize the
 * DOM attributes of reused nodes. Rows and cells are absolutely positioned so
 * the re-append is visually a no-op; it exists so the accessibility tree's
 * traversal order matches the data.
 */
function ensureDomOrder(container, ordered) {
  const current = [...container.children];
  if (current.length === ordered.length) {
    let same = true;
    for (let i = 0; i < ordered.length; i += 1) {
      if (current[i] !== ordered[i]) {
        same = false;
        break;
      }
    }
    if (same) return;
  }
  const fragment = document.createDocumentFragment();
  for (const el of ordered) fragment.appendChild(el);
  container.appendChild(fragment);
}

/* ------------------------------------------------------------------ */
/* Active cell / focus                                                 */
/* ------------------------------------------------------------------ */

function onScroll() {
  // Scrolling never re-targets the active cell: its stable row_id/column id is
  // preserved and it is always kept mounted, so aria-activedescendant can
  // never silently point at a different logical row.
  updateMounts();
}

function refreshActiveHighlight() {
  if (state.activeEl) state.activeEl.classList.remove('is-active');
  const el = cellElement(state.active.rowIndex, state.active.colIndex);
  if (el) {
    el.classList.add('is-active');
    state.activeEl = el;
  } else {
    state.activeEl = null;
  }
  if (el) els.grid.setAttribute('aria-activedescendant', el.id);
}

function cellElement(rowIndex, colIndex) {
  const rec = state.mountedRows.get(rowIndex);
  if (!rec) return null;
  return rec.cells.get(colIndex) || null;
}

function applyActive({ scroll = true } = {}) {
  if (!state.rows.length) return;
  const { rowIndex, colIndex } = state.active;
  if (scroll) ensureCellVisible(rowIndex, colIndex);
  updateMounts();

  const el = cellElement(rowIndex, colIndex);
  if (!el) {
    // Should be unreachable: the active row and column are always forced into
    // the mounted set.
    throw new Error(`active cell ${rowIndex}/${colIndex} was not mounted`);
  }
  els.grid.setAttribute('aria-activedescendant', el.id);
  state.activeEl = el;
  refreshActiveHighlight();
  renderStatus();
}

function onFocus() {
  if (!state.rows.length) return;
  els.grid.classList.add('is-focused');
  // Focus returns to the grid after Tab, resizing, or programmatic focus.
  ensureCellVisible(state.active.rowIndex, state.active.colIndex);
  updateMounts();
  const el = cellElement(state.active.rowIndex, state.active.colIndex);
  if (el) {
    els.grid.setAttribute('aria-activedescendant', el.id);
    state.activeEl = el;
    refreshActiveHighlight();
  }
  renderStatus();
}

function moveActive(rowIndex, colIndex) {
  const clampedRow = clamp(rowIndex, 0, state.rows.length - 1);
  const clampedCol = clamp(colIndex, 0, state.columns.length - 1);
  if (
    clampedRow === state.active.rowIndex &&
    clampedCol === state.active.colIndex
  ) {
    // Still make sure the target is visible/mounted after a resize.
    ensureCellVisible(clampedRow, clampedCol);
    updateMounts();
    refreshActiveHighlight();
    return;
  }
  state.active = { rowIndex: clampedRow, colIndex: clampedCol };
  applyActive({ scroll: true });
}

/**
 * Jump to a logical cell with explicit scroll coordinates. Used for
 * Ctrl+Home / Ctrl+End, where the browser's own scroll position must line up
 * with the target even at the very end of a 8675px-wide canvas.
 */
function goToCell(rowIndex, colIndex, { scrollTop, scrollLeft }) {
  state.active = {
    rowIndex: clamp(rowIndex, 0, state.rows.length - 1),
    colIndex: clamp(colIndex, 0, state.columns.length - 1),
  };
  els.grid.scrollTop = scrollTop;
  els.grid.scrollLeft = scrollLeft;
  updateMounts();
  const el = cellElement(state.active.rowIndex, state.active.colIndex);
  if (!el) throw new Error('goToCell target not mounted');
  els.grid.setAttribute('aria-activedescendant', el.id);
  state.activeEl = el;
  refreshActiveHighlight();
  renderStatus();
}

function ensureCellVisible(rowIndex, colIndex) {
  const grid = els.grid;
  const rowH = state.rowHeight;
  const headerH = state.headerHeight;

  // A row's canvas y is headerHeight + rowIndex*rowHeight (the canvas starts
  // after the sticky header). Its rendered y is that minus scrollTop; keep it
  // between the sticky header and the bottom edge of the scrollport.
  const contentTop = headerH + rowIndex * rowH;
  const contentBottom = contentTop + rowH;

  if (contentTop - grid.scrollTop < headerH) {
    grid.scrollTop = rowIndex * rowH;
  } else if (contentBottom - grid.scrollTop > grid.clientHeight) {
    grid.scrollTop = contentBottom - grid.clientHeight;
  }

  // Horizontal: pinned columns are always visible; scrollable columns need to
  // clear the pinned overlay on the left.
  const colInfo = state.columns[colIndex];
  if (!colInfo.pinned) {
    const left = state.colStarts[colIndex];
    const right = left + colInfo.width_px;
    const visibleLeft = grid.scrollLeft + state.pinnedWidth;
    const visibleRight = grid.scrollLeft + grid.clientWidth;

    if (left < visibleLeft) {
      grid.scrollLeft = left - state.pinnedWidth;
    } else if (right > visibleRight) {
      const available = grid.clientWidth - state.pinnedWidth;
      // A column wider than the usable (non-pinned) viewport cannot fit; align
      // its start so the value is readable rather than hiding it under pins.
      grid.scrollLeft = colInfo.width_px > available
        ? left - state.pinnedWidth
        : right - grid.clientWidth;
    }
  }
}

function ensureCellVisibleInViewport() {
  ensureCellVisible(state.active.rowIndex, state.active.colIndex);
}

function onKeyDown(event) {
  if (event.altKey) return;

  const { rowIndex, colIndex } = state.active;
  const page = Math.max(
    1,
    Math.floor((els.grid.clientHeight - state.headerHeight) / state.rowHeight) - 1,
  );
  let handled = true;

  switch (event.key) {
    case 'ArrowRight':
      moveActive(rowIndex, colIndex + 1);
      break;
    case 'ArrowLeft':
      moveActive(rowIndex, colIndex - 1);
      break;
    case 'ArrowDown':
      moveActive(rowIndex + 1, colIndex);
      break;
    case 'ArrowUp':
      moveActive(rowIndex - 1, colIndex);
      break;
    case 'PageDown':
      moveActive(rowIndex + page, colIndex);
      break;
    case 'PageUp':
      moveActive(rowIndex - page, colIndex);
      break;
    case 'Home':
      if (event.ctrlKey || event.metaKey) {
        goToCell(0, 0, { scrollTop: 0, scrollLeft: 0 });
      } else {
        moveActive(rowIndex, 0);
      }
      break;
    case 'End':
      if (event.ctrlKey || event.metaKey) {
        goToCell(state.rows.length - 1, state.columns.length - 1, {
          scrollTop: els.grid.scrollHeight,
          scrollLeft: els.grid.scrollWidth,
        });
      } else {
        moveActive(rowIndex, state.columns.length - 1);
      }
      break;
    default:
      handled = false;
  }

  if (handled) {
    event.preventDefault();
    event.stopPropagation();
  }
  // Tab / Shift+Tab intentionally fall through: the grid is a single tab stop
  // and the browser moves focus out normally (no keyboard trap).
}

function onPointerDown(event) {
  const cell = event.target.closest('.cell[role="gridcell"]');
  if (!cell || !els.grid.contains(cell)) return;
  event.preventDefault();
  const rowIndex = Number(cell.dataset.rowIndex);
  const colIndex = Number(cell.dataset.colIndex);
  state.active = { rowIndex, colIndex };
  els.grid.focus({ preventScroll: true });
  applyActive({ scroll: false });
  refreshActiveHighlight();
}

/* ------------------------------------------------------------------ */
/* Status panel                                                        */
/* ------------------------------------------------------------------ */

function renderStatus() {
  const { rowIndex, colIndex } = state.active;
  if (!state.rows.length) return;
  const row = state.rows[rowIndex];
  const col = state.columns[colIndex];
  const value = row.cells[col.id];

  const parts = [
    `第 ${rowIndex + 1}/${state.rows.length} 行`,
    `第 ${colIndex + 1}/${state.columns.length} 列`,
    `列名 ${col.label}`,
    `列ID ${col.id}`,
    `行ID ${row.row_id}`,
    `类型 ${col.type}`,
    `值 ${rawText(value)}`,
  ];
  els.status.textContent = parts.join(' · ');

  const meta = document.getElementById('active-meta');
  if (meta) {
    meta.textContent =
      `第 ${rowIndex + 1} 行（${row.row_id}）· 第 ${colIndex + 1} 列·` +
      ` ${col.label}（${col.id}）· 类型 ${col.type}`;
  }
  const val = document.getElementById('active-value');
  if (val) val.textContent = rawText(value);

  const stats = document.getElementById('dom-stats');
  if (stats) {
    stats.textContent =
      `已挂载 DOM：${state.mountedRows.size} 行 / ${countMountedCells()} 个单元格` +
      `（峰值 ${state.maxMountedRows} 行 / ${state.maxMountedCells} 个单元格）`;
  }
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/* Debug/automation surface. Not used by the UI itself. */
window.__wideTable = {
  getStats() {
    return {
      mountedRows: state.mountedRows.size,
      mountedCells: countMountedCells(),
      maxMountedRows: state.maxMountedRows,
      maxMountedCells: state.maxMountedCells,
      renderCount: state.renderCount,
      scrollTop: els.grid.scrollTop,
      scrollLeft: els.grid.scrollLeft,
      active: { ...state.active },
      activeId: els.grid.getAttribute('aria-activedescendant'),
      rowCount: state.rows.length,
      colCount: state.columns.length,
    };
  },
  getActive() {
    const { rowIndex, colIndex } = state.active;
    const row = state.rows[rowIndex];
    const col = state.columns[colIndex];
    return {
      rowIndex,
      colIndex,
      rowId: row.row_id,
      colId: col.id,
      value: row.cells[col.id],
      label: col.label,
    };
  },
};

load().catch((error) => {
  console.error(error);
  els.grid.innerHTML = '';
  const box = document.createElement('div');
  box.className = 'loading error';
  box.textContent = `无法加载数据：${error.message}。请通过 fixtures/serve.py 启动 HTTP 服务。`;
  els.grid.appendChild(box);
});
