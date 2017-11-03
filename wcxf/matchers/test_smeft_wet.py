import unittest
import numpy as np
import numpy.testing as npt
import yaml
import json
import pkgutil
import wcxf

np.random.seed(89)

# generate a random WC instance for the SMEFT Warsaw basis
C_SMEFT_random = {}
basis = wcxf.Basis['SMEFT', 'Warsaw']
for sector, wcs in basis.sectors.items():
    for name, d in wcs.items():
         C_SMEFT_random[name] = 1e-6*np.random.rand()
         if 'real' not in d or d['real'] == False:
             C_SMEFT_random[name] += 1j*1e-6*np.random.rand()
v_SMEFT_random = wcxf.WC.dict2values(C_SMEFT_random)
wc_SMEFT_random = wcxf.WC('SMEFT', 'Warsaw', scale=160,
                          values=v_SMEFT_random)

class TestSMEFTWET(unittest.TestCase):
    def test_smeft_wet(self):
        wc_WET_random = wc_SMEFT_random.match('WET', 'JMS')
        wc_WET_random.validate()
