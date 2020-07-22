"""
Functions that are only useful within the LLNL environment
"""

import os
import sys


def is_lc_system():
    """
    Use heuristic to determine if we're running on an LC system.
    """
    # TODO: This won't work on Lassen, which doesn't support lustre.

    return os.path.exists('/p/lustre1')
