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

class TestWarsawMass(unittest.TestCase):
    def test_smeft_mass(self):
        wc_Warsawmass_random = wc_Warsaw_random.translate('Warsaw mass')
        p = {'Vub': 3.6e-3}  # pass a parameter, but not all
        wc_Warsawmass_random = wc_Warsaw_random.translate('Warsaw mass', p)
        wc_Warsawmass_random.validate()
        # almost all WCs should actually stay the same
        for k, v in wc_Warsaw_random.dict.items():
            if k.split('_')[0] not in ['uphi', 'uG', 'uW', 'uB', 'llphiphi']:
                self.assertEqual(wc_Warsawmass_random.dict[k], v,
                                 msg="Not equal for {}".format(k))
        for i in range(3):
            for j in range(3):
                if i > j:
                    # the off-diagonal neutrino mass matrix elements
                    # must vanish in the mass basis, i.e. be absent
                    self.assertTrue('llphiphi_{}{}'.format(i+1, j+1) not in wc_Warsawmass_random.dict)


class TestWarsawUp(unittest.TestCase):
    def test_warsaw_up(self):
        wc_Warsawup_random = wc_Warsaw_random.translate('Warsaw up')
        wc_Warsawup_random.validate()
        # translate back and check that nothing changed
        wc_roundtrip = wc_Warsawup_random.translate('Warsaw')
        for k, v in wc_Warsaw_random.dict.items():
            self.assertAlmostEqual(v, wc_roundtrip.dict[k], places=12,
                                   msg="Failed for {}".format(k))
