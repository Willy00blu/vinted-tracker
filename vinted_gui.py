#!/usr/bin/env python3
"""Vinted Tracker — nuova GUI (NiceGUI).

Apri questo file in VS Code e premi ▶ (Run Python File) per avviare l'interfaccia.
Si apre il browser su http://localhost:8080.

NB: l'interprete di VS Code deve essere **Python 3.13** (lo stesso con cui gira
gui.py) — è l'unico che ha nicegui + le dipendenze del progetto.

Sostituisce la vecchia gui.py (Tkinter), che resta come fallback.
"""

from webgui.app import run

if __name__ in {"__main__", "__mp_main__"}:
    try:
        run()
    except KeyboardInterrupt:
        print("\nServer fermato.")
