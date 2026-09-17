from __future__ import annotations

import argparse

from .app import Reader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read your HN Rerank profile in the terminal"
    )
    parser.add_argument(
        "--server",
        help="Server URL, including deployment prefix (for example https://host/hn/)",
    )
    parser.add_argument("--version", action="version", version="hn-rerank 0.1.0")
    args = parser.parse_args()
    try:
        app = Reader(server=args.server)
    except ValueError as exc:
        parser.error(str(exc))
    app.run()


if __name__ == "__main__":
    main()
