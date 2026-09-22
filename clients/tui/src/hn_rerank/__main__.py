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
    parser.add_argument(
        "--prefetch",
        type=int,
        default=10,
        metavar="N",
        help="Fetch cached summaries for the next N stories (default 10; 0 disables).",
    )
    parser.add_argument("--version", action="version", version="hn-rerank 0.1.0")
    args = parser.parse_args()
    if args.prefetch < 0:
        parser.error("--prefetch must be 0 or greater")
    try:
        app = Reader(server=args.server, prefetch=args.prefetch)
    except ValueError as exc:
        parser.error(str(exc))
    app.run()


if __name__ == "__main__":
    main()
