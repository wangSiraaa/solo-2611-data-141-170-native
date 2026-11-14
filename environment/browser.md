# 本机浏览器验证

本批运行环境已有 Node.js 20 和 Chromium，浏览器可执行文件为 `/usr/bin/chromium`。仅涉及页面的题目需要浏览器；其他题目按各自 README 安装必要的普通依赖。

如使用 Playwright，可在项目内安装 `playwright`，启动时设置 `executablePath: '/usr/bin/chromium'` 和 `headless: true`，无需重复下载浏览器。隔离环境若提示 sandbox 不可用，可显式传入 `args: ['--no-sandbox']`。浏览器的下载、缓存、截图、依赖及测试输出均放在项目内忽略目录。

以实际页面运行、键盘操作及 DOM/可访问属性检查验证页面行为。环境自检通过不代表题目实现或功能验收通过。
