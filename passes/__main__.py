"""Allow ``python -m passes`` as an alias for the ``passes`` console script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
