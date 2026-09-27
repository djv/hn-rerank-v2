from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from scripts.audit_publication import audit, hostname


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.Jack-Clark.net/post", "jack-clark.net"),
        ("https://author.substack.com/post", "author.substack.com"),
        ("https://[bad", None),
        (None, None),
        ("javascript:foo", None),
    ],
)
def test_hostname(url: str | None, expected: str | None) -> None:
    assert hostname(url) == expected


def test_audit_is_user_scoped_read_only_and_counts_crossposts(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE users (id INTEGER);
        CREATE TABLE stories (id INTEGER, title TEXT, url TEXT, source TEXT,
                              text_content TEXT, self_text TEXT, article_body TEXT);
        CREATE TABLE feedback (user_id INTEGER, story_id INTEGER, action TEXT, updated_at REAL);
        INSERT INTO users VALUES (1), (2);
        INSERT INTO stories VALUES
          (10,'Newsletter','https://example.org/1','rss_example','text','self','body'),
          (11,'Crosspost','https://example.org/1','hn','text','',''),
          (12,'Other','https://other.org/1','hn','text','','');
        INSERT INTO feedback VALUES (1,10,'up',1),(1,11,'up',2),(1,12,'down',3),(2,10,'down',4);
        """)
    before = path.read_bytes()
    result = audit(path, 1, 10)
    assert result.training_counts == {"up": 2, "down": 1}
    assert result.publication_counts == {"up": 2}
    assert result.duplicate_urls == {"https://example.org/1": [10, 11]}
    assert result.text_chars == 4
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="User"):
        audit(path, 3, 10)


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        audit(path, 1, 10)
    assert not path.exists()
