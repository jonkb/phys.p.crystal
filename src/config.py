""" Establish a few constants shared across the whole program
"""

import os

# Paths
src_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.realpath(os.path.join(src_dir, ".."))
res_dir = os.path.realpath(os.path.join(root_dir, "res"))
data_dir = os.path.realpath(os.path.join(root_dir, "data"))
