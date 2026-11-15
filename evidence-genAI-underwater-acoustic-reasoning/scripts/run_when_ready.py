"""Await a supervised stage marker without extending the research deadline."""
import argparse,subprocess,time
from pathlib import Path
from underwater_reasoning.runtime import budget_check
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--marker',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    marker=ROOT/a.marker
    while not marker.exists():budget_check();time.sleep(10)
    if marker.read_text().strip()!='0':raise SystemExit('Previous stage failed; inspect its real error before continuing')
    budget_check();raise SystemExit(subprocess.call(a.command,cwd=ROOT))
