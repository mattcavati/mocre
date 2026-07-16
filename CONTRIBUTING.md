# Contributing to MOCRE-1

Thanks for your interest. MOCRE-1 is a research-grade earthquake-forecasting model built in the
open, and its guiding value is **intellectual honesty over impressive-looking numbers**. If you
contribute, keep that bar.

## The one rule that matters

**Report negative and null results as first-class outcomes.** The development log
([`docs/DEVLOG.md`](docs/DEVLOG.md)) is full of bugs caught, overfits rejected, and "not
statistically distinguishable from ETAS-only" verdicts kept in plain sight. A PR that makes a
metric look better by removing a holdout, dropping a significance test, or tuning to the metric
will be rejected even if the number goes up. A PR that finds our model is *worse* than we claimed,
with evidence, is exactly what we want.

## Getting set up

The heavy data tree (~19 GB) is **not** committed — it is rebuilt from public sources:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cd src
python3 ingest_catalog.py        # USGS ComCat
python3 ingest_gnss.py           # Nevada Geodetic Lab
python3 pipeline.py ../config/segment_sjc.json
```

Everything under `data/`, `out/`, `public/` and `.deploy/` is generated — never commit it (the
`.gitignore` already blocks it; don't force it in).

## Style

- **Code and code comments in English; docs may be Spanish or English.**
- No secrets, ever. No API keys, tokens, `.env` files, service-account JSON. The gateway reads
  everything from environment variables — keep it that way.
- Prefer a real fix at the root cause over a parameter tweak. If you must approximate, say so in a
  comment (see `src/stress_graph.py` for the tone).
- New fuzzy rules must respect monotonicity as an **ordered chain**, not "all consequents ≥ 0"
  (that exact over-constraint was a real bug — see DEVLOG F5).

## Security

Found a vulnerability? Please **do not** open a public issue. Follow the private reporting
instructions in [`SECURITY.md`](SECURITY.md).
