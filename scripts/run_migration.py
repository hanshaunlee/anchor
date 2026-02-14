#!/usr/bin/env python3
"""
Run a DB migration from db/migrations/ without needing psql.

Usage:
  python scripts/run_migration.py [migration_name]
  python scripts/run_migration.py 006_agent_runs_step_trace

Loads DATABASE_URL from environment or .env. Uses psycopg2 if installed;
otherwise prints the SQL so you can run it in Supabase Dashboard → SQL Editor.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env from repo root if present
def _load_dotenv() -> None:
    root = Path(__file__).resolve().parent.parent
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k.strip(), v)


def main() -> int:
    _load_dotenv()
    name = sys.argv[1] if len(sys.argv) > 1 else "006_agent_runs_step_trace"
    if not name.endswith(".sql"):
        name = name + ".sql"
    root = Path(__file__).resolve().parent.parent
    path = root / "db" / "migrations" / name
    if not path.exists():
        print(f"Migration not found: {path}", file=sys.stderr)
        return 1
    sql = path.read_text()
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL not set. Set it in .env or environment.", file=sys.stderr)
        _print_sql_and_instructions(sql, name)
        return 1

    try:
        import psycopg2
    except ImportError:
        print("psycopg2 not installed. Install with: pip install psycopg2-binary", file=sys.stderr)
        _print_sql_and_instructions(sql, name)
        return 1

    try:
        conn = psycopg2.connect(url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.close()
        print(f"Ran migration: {name}")
        return 0
    except Exception as e:
        err = str(e).lower()
        if "connection refused" in err or "could not connect" in err or "connection" in err and "failed" in err:
            print("Database not reachable (is local Postgres/Supabase running?).", file=sys.stderr)
            print("For hosted Supabase: run the SQL below in Dashboard → SQL Editor.", file=sys.stderr)
            print("Or set DATABASE_URL to your project's connection string (Project Settings → Database).", file=sys.stderr)
        else:
            print(f"Migration failed: {e}", file=sys.stderr)
        _print_sql_and_instructions(sql, name)
        return 1


def _print_sql_and_instructions(sql: str, name: str) -> None:
    print("\nRun this SQL manually (Supabase Dashboard → SQL Editor):\n")
    print("---")
    print(sql.strip())
    print("---")
    print(f"\nOr with a working DB: psql $DATABASE_URL -f db/migrations/{name}")


if __name__ == "__main__":
    sys.exit(main())
