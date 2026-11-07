#!/usr/bin/env python3
"""UFCash: picks and opening-line value bets for a UFC card.

  ufcash setup        one time (and whenever you want the latest fights): download the public UFC data (~200 MB) and build the features
  ufcash card         predict the bouts in data/upcoming.csv: pick, win probability, and value bets at the odds you entered

Edit data/upcoming.csv with the card (fighter names as on ufcstats.com) and the American moneylines you can bet now.
Value bets come from the opening-line model: bet early in the week, at or near the opening price. Paper-trade first; see README.md.
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
UA = "Mozilla/5.0 (compatible; ufcash)"
GRECO = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
SOURCES = [(f"{GRECO}/{f}.csv", RAW / "greco" / f"{f}.csv") for f in
           ("ufc_event_details", "ufc_fight_details", "ufc_fight_results", "ufc_fight_stats", "ufc_fighter_details", "ufc_fighter_tott")] + [
    ("https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master/ufc-master.csv", RAW / "ultimate" / "ufc-master.csv"),
    ("https://raw.githubusercontent.com/nlbair/ufc-data/main/data/ufc_all_fights_wide.parquet", RAW / "nlbair" / "ufc_all_fights_wide.parquet"),
    ("https://github.com/jadevit/fight-canon-db/releases/latest/download/canon.db", RAW / "canon" / "canon.db"),
]
FEATURES = ["build_data.py", "odds.py", "canon_feats.py", "features.py"]
if os.name == "nt":
    os.system("")
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def step(msg):
    print(f"\033[36;1m>\033[0m {msg}" if COLOR else f"> {msg}", flush=True)


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def run(script):
    out = None if os.environ.get("UFCASH_VERBOSE") else subprocess.DEVNULL
    r = subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, stdout=out, stderr=None if out is None else subprocess.PIPE, text=True)
    if r.returncode:
        print((r.stderr or "")[-3000:])
        sys.exit(f"{script} failed; rerun with UFCASH_VERBOSE=1 for details")


def setup():
    t0 = time.time()
    step("Downloading the public UFC data: ufcstats mirror, pro records of 373k MMA fights, BestFightOdds lines (~200 MB)")
    for url, dest in SOURCES:
        download(url, dest)
    build()
    print(f"Setup done in {time.time() - t0:.0f} s. Next: edit data/upcoming.csv, then run: ufcash card")


def build():
    step("Building leakage-free pre-fight features for every bout (~30 s)")
    for s in FEATURES:
        run(s)


def card():
    if not (RAW / "greco" / "ufc_fight_results.csv").exists():
        sys.exit("run `ufcash setup` first")
    build()
    subprocess.run([sys.executable, str(ROOT / "src" / "predict.py")], cwd=ROOT, check=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "setup":
        setup()
    elif cmd == "card":
        card()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
