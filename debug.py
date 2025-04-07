# debug.py
# encoding: utf-8


"""
Debugging utilities
"""

import os
import sys

DEBUG = ((hasattr(sys, 'gettrace')  # check if in debug mode
          and sys.gettrace() is not None)
         or os.environ.get('ALTDBG'))  # or ALTDBG env var is set
print(f"""
DEBUG TOOL IMPORTED
DEBUG MODE IS {"ACTIVE" if DEBUG else "INACTIVE"}
"""
      )

DEBUG_INITIATOR = "breakpoint"


class DebugStartException(Exception):
    """Exception raised to start debugging"""
    pass


def debug_print(*args, **kwargs):
    """print if in debug mode"""
    if DEBUG:
        print(*args, **kwargs)


def cond_breakpoint():
    """breakpoint if in debug mode"""
    if DEBUG:
        import pdb
        pdb.set_trace()
        if DEBUG_INITIATOR == 'breakpoint':
            raise breakpoint()
        elif DEBUG_INITIATOR == 'raise':
            raise DebugStartException('Debugging started')
        elif DEBUG_INITIATOR == 'quit':
            sys.exit(1)
