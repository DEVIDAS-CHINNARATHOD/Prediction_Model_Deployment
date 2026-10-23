"""Send the included sample request to a running or deployed model API."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.getenv("MODEL_API_TOKEN")
SAMPLE = Path(__file__).resolve().parent / "examples" / "predict_request.json"


def main() -> None:
    body = SAMPLE.read_bytes()
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    request = Request(f"{BASE_URL}/predict", data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except HTTPError as exc:
        print(f"API returned HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
        raise SystemExit(1) from exc
    except URLError as exc:
        print(f"Could not reach {BASE_URL}: {exc.reason}")
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
