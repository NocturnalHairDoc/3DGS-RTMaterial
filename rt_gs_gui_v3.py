"""Canonical launcher for the 3DGS-RTMaterial GUI.

The implementation lives in :mod:`viewer.gui.pbr_viewer`. Keeping this small
root entry point preserves the documented ``python rt_gs_gui_v3.py`` command.
"""

from importlib import import_module
from runpy import run_module


def main():
    """Run the packaged V3 viewer as a script."""
    run_module("viewer.gui.pbr_viewer", run_name="__main__")


def __getattr__(name):
    """Lazily expose implementation symbols for import compatibility."""
    return getattr(import_module("viewer.gui.pbr_viewer"), name)


if __name__ == "__main__":
    main()
