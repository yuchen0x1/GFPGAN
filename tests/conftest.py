import os
import sys

# Make sure the test suite always runs against this source checkout, not
# against an installed copy of gfpgan elsewhere on the system.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import gfpgan before any test module: this installs the basicsr/torchvision
# compatibility shims (gfpgan/_compat.py). Test modules such as
# test_gfpgan_model.py import basicsr directly, which fails on
# torchvision >= 0.17 without the shim.
import gfpgan  # noqa: F401, E402
