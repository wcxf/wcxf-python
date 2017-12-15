import unittest
import numpy as np
import numpy.testing as npt
import yaml
import json
import pkgutil
import wcxf

np.random.seed(89)

# generate a random WC instance for the SMEFT Warsaw basis
C_Warsaw_random = {}
basis = wcxf.Basis['SMEFT', 'Warsaw']
for sector, wcs in basis.sectors.items():
    for name, d in wcs.items():
         C_Warsaw_random[name] = 1e-6*np.random.rand()
         if 'real' not in d or d['real'] == False:
             C_Warsaw_random[name] += 1j*1e-6*np.random.rand()
v_Warsaw_random = wcxf.WC.dict2values(C_Warsaw_random)
wc_Warsaw_random = wcxf.WC('SMEFT', 'Warsaw', scale=160,
                          values=v_Warsaw_random)

class TestSMEFTWET(unittest.TestCase):
    def test_warsaw_jms(self):
        wc = wc_Warsaw_random.match('WET', 'JMS')
        self.assertEqual(wc.eft, 'WET')
        self.assertEqual(wc.basis, 'JMS')
        wc.validate()

    def test_warsaw_flavio(self):
        wc = wc_Warsaw_random.match('WET', 'flavio')
        self.assertEqual(wc.eft, 'WET')
        self.assertEqual(wc.basis, 'flavio')
        wc.validate()

    def test_warsaw_EOS(self):
        wc = wc_Warsaw_random.match('WET', 'EOS')
        self.assertEqual(wc.eft, 'WET')
        self.assertEqual(wc.basis, 'EOS')
        wc.validate()

    def test_warsaw_Bern(self):
        wc = wc_Warsaw_random.match('WET', 'Bern')
        self.assertEqual(wc.eft, 'WET')
        self.assertEqual(wc.basis, 'Bern')
        wc.validate()
