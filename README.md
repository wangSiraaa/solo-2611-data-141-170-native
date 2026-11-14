# 题 148 输入材料

用本地 HTTP 服务加载 columns.json 和 rows.json。可用 Node 20 的普通前端框架；也可运行 `python3 fixtures/serve.py --port 8148` 为当前目录提供静态文件。输入次序就是浏览次序，row_id 不等于位置。只提供数据，不提供表格组件。自动验收可安装 Playwright npm 包并指定 executablePath=/usr/bin/chromium；本批原生环境提供系统 Chromium，不必下载浏览器。

这些文件是输入与必要环境，不含题目核心实现、参考输出或独立答案。相对路径以题目仓库根为准。所有数据为合成数据。
