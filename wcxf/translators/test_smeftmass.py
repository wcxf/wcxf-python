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

class TestSMEFTMass(unittest.TestCase):
    def test_smeft_mass(self):
        wc_SMEFTmass_random = wc_SMEFT_random.translate('Warsaw mass')
        wc_SMEFTmass_random.validate()
        # almost all WCs should actually stay the same
        for k, v in wc_SMEFT_random.dict.items():
            if k.split('_')[0] not in ['uphi', 'uG', 'uW', 'uB', 'llphiphi']:
                self.assertEqual(wc_SMEFTmass_random.dict[k], v,
                                 msg="Not equal for {}".format(k))
        for i in range(3):
            for j in range(3):
                if i > j:
                    # the off-diagonal neutrino mass matrix elements
                    # must vanish in the mass basis, i.e. be absent
                    self.assertTrue('llphiphi_{}{}'.format(i+1, j+1) not in wc_SMEFTmass_random.dict)
