import unittest
import numpy as np
import numpy.testing as npt
import wcxf
from . import wet

np.random.seed(87)

def get_random_wc(eft, basis, cmax=1e-2):
    """Generate a random Wilson coefficient instance for a given basis."""
    basis_obj = wcxf.Basis[eft, basis]
    _wc = {}
    for s in basis_obj.sectors.values():
        for name, d in s.items():
            _wc[name] = cmax*np.random.rand()
            if 'real' not in d or not d['real']:
                _wc[name] += 1j*cmax*np.random.rand()
    return wcxf.WC(eft, basis, 80., wcxf.WC.dict2values(_wc))


class TestTranslateWET(unittest.TestCase):
    def test_scalar2array(self):
        d = {'bla_123': 3, 'blo': 5j}
        da = wet._scalar2array(d)
        self.assertEqual(da['blo'], 5j)
        self.assertIn('bla', da)
        self.assertEqual(da['bla'].shape, (3, 3, 3))
        self.assertTrue(np.isnan(da['bla'][0, 0, 0]))
        self.assertEqual(da['bla'][0, 1, 2], 3)


class TestJMS2Flavio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.flavio_wc = jms_wc.translate('flavio')

    def test_validate(self):
        self.flavio_wc.validate()

    def test_nan(self):
        for k, v in self.flavio_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        fkeys = set(self.flavio_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET', 'flavio'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")

    def test_incomplete_input(self):
        # generate and input WC instance with just 1 non-zero coeff.
        jms_wc = wcxf.WC('WET', 'JMS', 80, {'VddRR_2323': {'Im': -1}})
        flavio_wc = jms_wc.translate('flavio')
        flavio_wc.validate()
        # the output WC instance should contain only one as well
        self.assertEqual(list(flavio_wc.dict.keys()), ['CVRR_bsbs'])


class TestJMS2Bern(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.bern_wc = jms_wc.translate('Bern')

    def test_validate(self):
        self.bern_wc.validate()

    def test_nan(self):
        for k, v in self.bern_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        bkeys = set(self.bern_wc.values.keys())
        bkeys_all = set([k for s in wcxf.Basis['WET', 'Bern'].sectors.values()
                         for k in s
                         if 'b' in k]) # for the time being, only look at b operators
        self.assertSetEqual(bkeys_all - bkeys, set(), msg="Missing coefficients")


class TestJMS2EOS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.eos_wc = jms_wc.translate('EOS')

    def test_validate(self):
        self.eos_wc.validate()

    def test_nan(self):
        for k, v in self.eos_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        fkeys = set(self.eos_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET', 'EOS'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")

class TestJMS2FormFLavor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jms_wc = get_random_wc('WET', 'JMS')
        cls.formflavor_wc = jms_wc.translate('formflavor')

    def test_validate(self):
        self.formflavor_wc.validate()

    def test_nan(self):
        for k, v in self.formflavor_wc.dict.items():
            self.assertFalse(np.isnan(v), msg="{} is NaN".format(k))

    def test_missing(self):
        fkeys = set(self.formflavor_wc.values.keys())
        fkeys_all = set([k for s in wcxf.Basis['WET', 'formflavor'].sectors.values()
                         for k in s])
        self.assertSetEqual(fkeys_all - fkeys, set(), msg="Missing coefficients")
