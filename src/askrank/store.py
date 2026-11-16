"""SQLite storage. Raw answers are saved BEFORE extraction, so if the extractor
crashes or improves later you can re-extract without paying for new answers.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from askrank.schemas import Answer, Extraction

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    project     TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS answers (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    query_id       TEXT NOT NULL,
    query_text     TEXT NOT NULL,
    run_index      INTEGER NOT NULL,
    text           TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    extracted      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS mentions (
    id        INTEGER PRIMARY KEY,
    answer_id INTEGER NOT NULL REFERENCES answers(id),
    name      TEXT NOT NULL,
    matched   TEXT,
    position  INTEGER NOT NULL,
    sentiment TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cited_urls (
    answer_id INTEGER NOT NULL REFERENCES answers(id),
    url       TEXT NOT NULL,
    domain    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answers_run ON answers(run_id);
CREATE INDEX IF NOT EXISTS idx_mentions_answer ON mentions(answer_id);
"""


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


class Store:
    def __init__(self, path: str | Path = "askrank.db"):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def start_run(self, project: str, config: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (project, started_at, config_json) VALUES (?, ?, ?)",
            (project, datetime.now(UTC).isoformat(), json.dumps(config)),
        )
        self.conn.commit()
        return cur.lastrowid

    def save_answer(self, run_id: int, a: Answer) -> int:
        cur = self.conn.execute(
            "INSERT INTO answers (run_id, provider, model, query_id, query_text, run_index,"
            " text, citations_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, a.provider, a.model, a.query_id, a.query_text, a.run_index, a.text,
             json.dumps([c.model_dump() for c in a.citations]), a.created_at.isoformat()),
        )
        self.conn.commit()
        return cur.lastrowid

    def save_extraction(self, answer_id: int, ex: Extraction) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM mentions WHERE answer_id = ?", (answer_id,))
            self.conn.execute("DELETE FROM cited_urls WHERE answer_id = ?", (answer_id,))
            self.conn.executemany(
                "INSERT INTO mentions (answer_id, name, matched, position, sentiment)"
                " VALUES (?, ?, ?, ?, ?)",
                [(answer_id, m.name, m.matched, m.position, m.sentiment.value)
                 for m in ex.mentions],
            )
            self.conn.executemany(
                "INSERT INTO cited_urls (answer_id, url, domain) VALUES (?, ?, ?)",
                [(answer_id, u, domain_of(u)) for u in ex.cited_urls],
            )
            self.conn.execute("UPDATE answers SET extracted = 1 WHERE id = ?", (answer_id,))

    def latest_run_id(self, project: str | None = None) -> int | None:
        sql, args = "SELECT MAX(id) FROM runs", ()
        if project:
            sql, args = sql + " WHERE project = ?", (project,)
        row = self.conn.execute(sql, args).fetchone()
        return row[0] if row else None

    def answers(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM answers WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()

    def mentions(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT m.*, a.provider, a.model, a.query_id FROM mentions m"
            " JOIN answers a ON a.id = m.answer_id WHERE a.run_id = ?",
            (run_id,),
        ).fetchall()

    def cited_domains(self, run_id: int) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT c.domain, COUNT(DISTINCT c.answer_id) AS n FROM cited_urls c"
            " JOIN answers a ON a.id = c.answer_id WHERE a.run_id = ?"
            " GROUP BY c.domain ORDER BY n DESC, c.domain",
            (run_id,),
        ).fetchall()
        return [(r["domain"], r["n"]) for r in rows]
