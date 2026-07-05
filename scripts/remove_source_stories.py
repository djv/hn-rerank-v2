"""Delete all stories matching a given source, preserving feedback-guarded stories.

Usage:
    uv run python scripts/remove_source_stories.py --source rss_reddit_gis
"""

from __future__ import annotations

import argparse
import sys

from database import Database


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    db = Database()
    with db.conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM stories WHERE source = ?", (args.source,)
        ).fetchone()[0]
        protected = conn.execute(
            "SELECT COUNT(*) FROM stories WHERE source = ? "
            "AND id IN (SELECT story_id FROM feedback)",
            (args.source,),
        ).fetchone()[0]
        to_delete = total - protected

        print(f"Source: {args.source}")
        print(f"  Total stories:     {total}")
        print(f"  Feedback-guarded:  {protected}")
        print(f"  Will be deleted:   {to_delete}")

        if protected > 0:
            print(f"\n  Feedback rows that will become orphaned: {protected}")
            rows = conn.execute(
                "SELECT s.id, s.title, f.action FROM stories s "
                "JOIN feedback f ON s.id = f.story_id "
                "WHERE s.source = ?",
                (args.source,),
            ).fetchall()
            for r in rows:
                print(f"    id={r[0]} action={r[2]} title={r[1][:60]}")

        if to_delete == 0:
            print("Nothing to delete.")
            return

        if not args.yes:
            ans = input(f"\nDelete {to_delete} stories? [y/N] ")
            if ans.lower() != "y":
                print("Aborted.")
                sys.exit(1)

        with conn:
            conn.execute(
                "DELETE FROM feedback WHERE story_id IN "
                "(SELECT id FROM stories WHERE source = ?)",
                (args.source,),
            )
            cursor = conn.execute(
                "DELETE FROM stories WHERE source = ? "
                "AND id NOT IN (SELECT story_id FROM feedback)",
                (args.source,),
            )
        print(f"Deleted {cursor.rowcount} stories (and cascaded data).")


if __name__ == "__main__":
    main()
