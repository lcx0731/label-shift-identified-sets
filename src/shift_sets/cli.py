"""Read one explicit known-channel problem and print an audit as JSON."""

import argparse
import json
from pathlib import Path
import sys

from .core import audit, multinomial_bands


def _unique(pairs):
    d = {}
    for key, value in pairs:
        if key in d:
            raise ValueError(f"duplicate JSON field: {key}")
        d[key] = value
    return d


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.input.read_text(), object_pairs_hook=_unique)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON must be an object")
        allowed = {"channel", "counts", "alpha", "lower", "upper", "contrasts"}
        if set(data)-allowed or "channel" not in data:
            raise ValueError("unknown field or missing channel")
        if "counts" in data:
            if "lower" in data or "upper" in data:
                raise ValueError("choose counts OR explicit lower/upper bands")
            bands = multinomial_bands(data["counts"], data.get("alpha", .05))
        else:
            if "alpha" in data or not {"lower", "upper"} <= set(data):
                raise ValueError("explicit mode needs lower/upper and no alpha")
            bands = {"lower": data["lower"], "upper": data["upper"],
                     "method": "user-supplied-no-coverage-claim"}
        result = audit(data["channel"], bands["lower"], bands["upper"],
                       data.get("contrasts"))
        result["bands"] = bands
        result["scope"] = "Fixed known channel; target sampling only in counts mode."
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "feasible" else 2
    except (ValueError, TypeError, OSError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 1
    except ArithmeticError as exc:
        print(f"numerical error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
