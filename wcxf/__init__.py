from .classes import *
from . import matchers

# read all EFTs and bases from the wcxf-bases submodule

import os
import glob

_root = os.path.abspath(os.path.dirname(__file__))
all_efts = glob.glob(os.path.join(_root, 'bases', '*.eft.yml'))
all_bases = glob.glob(os.path.join(_root, 'bases', '*.basis.yml'))
child_bases = glob.glob(os.path.join(_root, 'bases', 'child', '*.basis.yml'))

for eft in all_efts:
    with open(eft, 'r') as f:
        EFT.load(f)

for basis in all_bases + child_bases:
    with open(basis, 'r') as f:
        Basis.load(f)
