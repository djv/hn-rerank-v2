from __future__ import annotations

import argparse

from .app import DEFAULT_PREFETCH, DEFAULT_PREFETCH_GENERATE, DEFAULT_SERVER, Reader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read your HN Rerank profile in the terminal"
    )
    parser.add_argument(
        "--server",
        default=None,
        # None keeps a saved profile's server; only an explicit flag overrides it.
        help=f"Server URL, including deployment prefix (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=DEFAULT_PREFETCH,
        metavar="N",
        help="Fetch cached summaries for the next N stories (default 20; 0 disables).",
    )
    parser.add_argument(
        "--prefetch-generate",
        type=int,
        default=DEFAULT_PREFETCH_GENERATE,
        metavar="N",
        help="Generate missing summaries in the next N navigation targets (default 3; 0 uses cache only).",
    )
    parser.add_argument("--version", action="version", version="hn-rerank 0.1.0")
    args = parser.parse_args()
    if args.prefetch < 0:
        parser.error("--prefetch must be 0 or greater")
    if args.prefetch_generate < 0:
        parser.error("--prefetch-generate must be 0 or greater")
    try:
        app = Reader(
            server=args.server,
            prefetch=args.prefetch,
            prefetch_generate=args.prefetch_generate,
        )
    except ValueError as exc:
        parser.error(str(exc))
    app.run()


if __name__ == "__main__":
    main()
