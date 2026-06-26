#!/bin/bash
# Double-clickable launcher (macOS) for the NiceGUI front-end.
# The project's deps (nicegui, vinted_scraper, sentence-transformers, …) live in
# Python 3.13 only — the shell's default `python3` is 3.11 and would fail.
cd "$(dirname "$0")" || exit 1
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
[ -x "$PY" ] || PY=python3
echo "Vinted Tracker (NiceGUI) → http://localhost:8080"
echo "(chiudi la finestra per fermare il server)"
echo
"$PY" -m webgui
code=$?
echo
echo "── server terminato (codice $code) ──"
read -n 1 -s -r -p "Premi un tasto per chiudere…"
echo
