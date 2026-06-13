"""One-time real-data acquisition. Network call is injectable so tests stay offline."""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import urllib.request

# martj42/international_results — canonical free dataset (1872–present).
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (trusted URL)
        return resp.read()

def download_results(dest: str, url: str = RESULTS_URL, *,
                     fetch: Callable[[str], bytes] = _http_get) -> str:
    """Download the results CSV to `dest`. `fetch` is injectable for testing.
    Returns the path written."""
    data = fetch(url)
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)
