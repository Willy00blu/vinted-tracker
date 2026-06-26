"""NiceGUI front-end for the Vinted Tracker.

The backend (core/, the monitor subprocesses, the shared SQLite DB) is unchanged;
this package is only the presentation layer, replacing the legacy Tkinter gui.py.

    python -m webgui      # launches the web UI (opens a native window)

Modules:
    services  — data access (configs, DB, coverage, network) + monitor process
                management (start/stop/restart/test/status, live-log streaming).
    app       — the NiceGUI app: shell (header, tabs, sidebar, live log) + views.
"""
