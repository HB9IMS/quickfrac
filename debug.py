# debug.py
# encoding: utf-8


"""
Debugging utilities
"""

import os
import sys
import hb9imsutils

DEBUG = ((hasattr(sys, 'gettrace')  # check if in debug mode
          and sys.gettrace() is not None)
         or os.environ.get('ALTDBG'))  # or ALTDBG env var is set
print(f"""
DEBUG TOOL IMPORTED
DEBUG MODE IS {"ACTIVE" if DEBUG else "INACTIVE"}
""")

VARIABLES = hb9imsutils.Namespace(c=0)

DEBUG_INITIATOR = "breakpoint"


class DebugStartException(Exception):
    """Exception raised to start debugging"""
    pass


def debug_print(*args, **kwargs):
    """print if in debug mode"""
    if DEBUG:
        print(*args, **kwargs)


def tick_counter():
    """increment the counter at VARIABLES.c and return the new value"""
    VARIABLES.c += 1
    return VARIABLES.c


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
