import os
import sys

# Make sure the test suite always runs against this source checkout, not
# against an installed copy of gfpgan elsewhere on the system.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
