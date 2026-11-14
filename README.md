# 只读宽表浏览器（800 行 × 64 列）

零依赖的本地静态页面：原生 JavaScript（ES module）+ HTML + CSS。不使用任何已完成的虚拟表格组件。

## 启动

```sh
# 在仓库根目录
python3 fixtures/serve.py --port 8148
# 浏览器打开 http://127.0.0.1:8148/index.html
```

必须经 HTTP 打开（页面用 `fetch` 读取 `fixtures/*.json`，`file://` 会被 CORS 拦截）。

## 实现要点

**单一滚动容器，两层虚拟化**

* `#grid` 是唯一滚动视口，内部一个尺寸为 `总宽 × (表头 + 800×行高)` 的 canvas。
* 纵向：只挂载视口内行 ± 3 行 overscan；`active` 行额外强制挂载。
* 横向：只挂载固定列 + 视口内可滚动列 ± 2 列 overscan；`active` 列额外强制挂载。
  因此每行不会创建全部 64 个单元格。
* 行与网格都在 canvas 中绝对定位；列 `left` 与 `width` 直接来自 `columns.json`
  （`宽度配置 + 固定行高`），因此表头与正文使用同一套几何。

**固定列与固定表头**

* 固定列不是复制出来的第二张表：正文里每行只有一份 DOM，前两列的单元格是
  `position: sticky; left: <pinned offset>`；表头单元格同理，表头行 `sticky; top: 0`。
  滚动时固定列贴住视口左侧，且可访问树中每个逻辑单元格只出现一次。
* 固定区与正文共享同一行元素和同一条 `top = headerHeight + rowIndex × rowHeight`，
  不会出现两套行位置。

**焦点与键盘**

* `role="grid"` 容器是唯一的 tab stop，真实焦点始终在容器上；
  当前单元格通过 `aria-activedescendant`（`cell-<row_id>-<col_id>`）指向。
* 逻辑位置用 `{rowIndex, colIndex}` 保存，单元格 id 用稳定的 `row_id` + 列 id 构成。
* 方向键、Home/End（行首/行尾）、Ctrl+Home/End（全表首/尾）、PageUp/PageDown。
* 移动到尚未挂载的单元格时，先设置滚动位置 → `updateMounts()` 挂载 → 再写
  `aria-activedescendant`，因此目标一定已存在。
* 滚动回收、容器缩放（`ResizeObserver`）都会重新挂载并保活 active 单元格；
  active 单元格不会被 unprefix 到 body。Tab/Shift+Tab 不被拦截，可正常离开/返回。

**可访问性**

* `grid` 上 `aria-rowcount=801`（800 数据行 + 1 表头行）、`aria-colcount=64`、
  `aria-readonly=true`。
* 数据行 `aria-rowindex` = 数据序号 + 2（表头为 1）；单元格与表头
  `aria-colindex` 为 1–64。
* 单元格 `aria-label` 形如
  `字段 10（列 c10），值：…；第 1 行，共 800 行；第 11 列，共 64 列；类型 text`，
  含列名、完整值、行列位置与类型。长文本在 `aria-label`、`title` 和页面
  “完整值”面板中均为全文，不截断成省略号。
* 值显示可区分：`null → (null)`（斜体/橙色）、`"" → (空字符串)`（灰色）、
  `false`（红色）/`true`（绿色）、数字（蓝色右对齐），朗读文本也各不相同。
* 深滚动后 DOM 子节点顺序仍按逻辑行列排序，可访问树遍历顺序稳定。

**只读**

无编辑、筛选、排序、图表；`aria-readonly="true"`。

## 验证（浏览器自动化）

安装 Playwright（本环境已装；其他环境 `npm i playwright`，用系统
`/usr/bin/chromium`，无需下载浏览器）：

```sh
python3 fixtures/serve.py --port 8148 &   # 先启动服务
node tests/verify.js        # 21 项：语义、键盘、滚动、焦点、可访问属性
node tests/sweep.js         # 用真实按键走完 800 行 × 64 列并统计挂载峰值
node tests/mount-peak.js    # 5 种视口下的挂载峰值与固定列/行对齐
```

`tests/verify.js` 会真实打开 `http://127.0.0.1:8148/index.html`，通过键盘事件、
滚动、窗口缩放和对 DOM/AX 树的检查来验证，不只是对数组切片做断言。

### 实测结果（本机 Chromium 156）

* `tests/verify.js`：21/21 通过。
* 全表遍历（`tests/sweep.js`，键盘逐行走完 800 行并多次横向扫完 64 列）：
  * 访问行 800 / 列 64，最终活动单元格 `{rowIndex: 799, rowId: R0018, colId: c06,
    value 与 rows.json 一致}`。
  * 挂载峰值 **312 个单元格 / 23 行**；结束后回落到 **220 / 19**。
  * 页面 DOM 峰值与峰值统计一致（312 单元格），不随访问过的行列累积。
* 多视口峰值（`tests/mount-peak.js`，含 1920×1080）：

  | 视口 | 峰值挂载单元格 | 峰值挂载行 |
  | --- | --- | --- |
  | 800×600 | 198 | 17 |
  | 1024×700 | 252 | 20 |
  | 1280×800 | 350 | 24 |
  | 1440×900 | 420 | 27 |
  | 1920×1080 | 627 | 33 |

  最大挂载单元格数为 **627**（视口内可见行列 + overscan），远小于
  `800 × 64 = 51200`。
* 固定列在任意水平滚动/分数偏移下始终位于 `x = 0` 与 `x = 120`；同一行内
  固定单元格与可滚动单元格 `top` 完全一致（0px 偏差）。

## 文件

```
index.html        页面结构（skip link、grid、状态/完整值面板）
styles.css        布局、sticky 固定列/表头、值类型样式
app.js            加载、虚拟化、挂载回收、键盘/焦点、状态面板
tests/verify.js   21 项浏览器行为与可访问性检查
tests/sweep.js    全表键盘遍历 + 挂载上限证明
tests/mount-peak.js 多视口峰值与对齐测量
```
