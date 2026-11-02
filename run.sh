#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 -m venv .venv
if [ "$(uname -s)" = Darwin ]; then
  CFLAGS='-include stdio.h' .venv/bin/python -m pip install -r requirements.txt
else
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python upgrade.py
.venv/bin/python verify_control.py > results_control/validation.txt
.venv/bin/python make_control_video.py
.venv/bin/python make_control_report.py
