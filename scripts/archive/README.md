# Scripts archive

Optional or one-off scripts preserved for validation and reference. Not referenced by Makefile or main docs. Run from repo root with appropriate `PYTHONPATH` if needed.

- **stress_supervisor_matrix.py** — Stress test: run supervisor in dry_run for multiple synthetic scenarios (urgency scam, new contact bursty), assert invariants, write `demo_out/stress/stress_results.json`. Usage: `PYTHONPATH=apps/api:. python scripts/archive/stress_supervisor_matrix.py`
