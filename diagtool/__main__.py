"""模块执行入口：python3 -m diagtool ..."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
