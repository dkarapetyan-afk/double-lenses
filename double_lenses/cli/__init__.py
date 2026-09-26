"""
Command-Line Interfaces for Double Lenses.
"""


def main():
    """Entry point — lazy-imports to avoid loading heavy modules on package import."""
    from double_lenses.cli.run_cluster import main as _main

    return _main()


__all__ = ["main"]
