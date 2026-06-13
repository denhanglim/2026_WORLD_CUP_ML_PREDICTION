from pathlib import Path
from wc2026.data.fetch import download_results, RESULTS_URL
from wc2026.data.results import load_results

def test_download_writes_file_and_is_loadable(tmp_path):
    fixture_bytes = Path("tests/fixtures/mini_results.csv").read_bytes()
    calls = {}
    def fake_fetch(url):
        calls["url"] = url
        return fixture_bytes
    dest = tmp_path / "raw" / "results.csv"
    out = download_results(str(dest), fetch=fake_fetch)
    assert Path(out).exists()
    assert calls["url"] == RESULTS_URL          # default URL used
    df = load_results(out)                       # downloaded file is loadable
    assert len(df) == 8

def test_download_respects_custom_url(tmp_path):
    seen = {}
    def fake_fetch(u):
        seen["u"] = u
        return b"date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n2020-01-01,A,B,1,0,Friendly,c,d,False\n"
    download_results(str(tmp_path / "r.csv"), url="http://x/y.csv", fetch=fake_fetch)
    assert seen["u"] == "http://x/y.csv"
