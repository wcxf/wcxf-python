import unittest
import wcxf


class TestWET(unittest.TestCase):
    def test_bern(self):
        basis = wcxf.Basis['WET', 'Bern']
        # comparing no. of WCs to table 1 of arXiv:1704.06639
        self.assertEqual(len(basis.sectors['sbsb']), 8)
        self.assertEqual(len(basis.sectors['dbdb']), 8)
        self.assertEqual(len(basis.sectors['ubenu']), 5*3)
        self.assertEqual(len(basis.sectors['ubmunu']), 5*3)
        self.assertEqual(len(basis.sectors['ubtaunu']), 5*3)
        self.assertEqual(len(basis.sectors['cbenu']), 5*3)
        self.assertEqual(len(basis.sectors['cbmunu']), 5*3)
        self.assertEqual(len(basis.sectors['cbtaunu']), 5*3)
        self.assertEqual(len(basis.sectors['sbuc']), 20)
        self.assertEqual(len(basis.sectors['sbcu']), 20)
        self.assertEqual(len(basis.sectors['dbuc']), 20)
        self.assertEqual(len(basis.sectors['dbcu']), 20)
        self.assertEqual(len(basis.sectors['sbsd']), 10)
        self.assertEqual(len(basis.sectors['dbds']), 10)
        self.assertEqual(len(basis.sectors['sb']), 57*2)
        self.assertEqual(len(basis.sectors['sbnunu']), 2*9)
        self.assertEqual(len(basis.sectors['db']), 57*2)
        self.assertEqual(len(basis.sectors['dbnunu']), 2*9)
        for l1 in ['e', 'mu', 'tau']:
            for l2 in ['e', 'mu', 'tau']:
                if l1 != l2:
                    self.assertEqual(len(basis.sectors['sb'+l1+l2]), 10)
                    self.assertEqual(len(basis.sectors['db'+l1+l2]), 10)

    def test_jms(self):
        basis = wcxf.Basis['WET', 'JMS']
        all_wc = {k: v for sk, sv in basis.sectors.items() for k, v in sv.items()}
        def isreal(wc):
            if 'real' in all_wc[wc] and all_wc[wc]['real']:
                return True
            else:
                return False
        def assert_len(s, n):
            L = sum([1 if isreal(k) else 2 for k in all_wc if s in k])
            try:
                self.assertEqual(L, n,
                             msg="Failed for {}".format(s))
            except Exception as exc:
                print(exc)
        # compare individual counts to tables 11-17 in arXiv:1709.04486
        assert_len('egamma_', 9*2)
        assert_len('ugamma_', 4*2)
        assert_len('dgamma_', 9*2)
        assert_len('uG_', 4*2)
        assert_len('dG_', 9*2)
        assert_len('VnunuLL_', 36)
        assert_len('VeeLL_', 36)
        assert_len('VnueLL_', 81)
        assert_len('VnuuLL_', 36)
        assert_len('VnudLL_', 81)
        assert_len('VeuLL_', 36)
        assert_len('VedLL_', 81)
        assert_len('VnueduLL_', 2*54)
        assert_len('VuuLL_', 10)
        assert_len('VddLL_', 45)
        assert_len('V1udLL_', 36)
        assert_len('V8udLL_', 36)
        assert_len('VeeRR_', 36)
        assert_len('VeuRR_', 36)
        assert_len('VedRR_', 81)
        assert_len('VuuRR_', 10)
        assert_len('VddRR_', 45)
        assert_len('V1udRR_', 36)
        assert_len('V8udRR_', 36)
        assert_len('VnueLR_', 81)
        assert_len('VeeLR_', 81)
        assert_len('VnuuLR_', 36)
        assert_len('VnudLR_', 81)
        assert_len('VeuLR_', 36)
        assert_len('VedLR_', 81)
        assert_len('VueLR_', 36)
        assert_len('VdeLR_', 81)
        assert_len('VnueduLR_', 2*54)
        assert_len('V1uuLR_', 16)
        assert_len('V8uuLR_', 16)
        assert_len('V1udLR_', 36)
        assert_len('V8udLR_', 36)
        assert_len('V1duLR_', 36)
        assert_len('V8duLR_', 36)
        assert_len('V1ddLR_', 81)
        assert_len('V8ddLR_', 81)
        assert_len('V1udduLR_', 2*36)
        assert_len('V8udduLR_', 2*36)
        assert_len('SeuRL_', 2*36)
        assert_len('SedRL_', 2*81)
        assert_len('SnueduRL_', 2*54)
        assert_len('SeeRR', 2*45)
        assert_len('SeuRR', 2*36)
        assert_len('TeuRR', 2*36)
        assert_len('SedRR', 2*81)
        assert_len('TedRR', 2*81)
        assert_len('SnueduRR', 2*54)
        assert_len('TnueduRR', 2*54)
        assert_len('S1uuRR', 2*10)
        assert_len('S8uuRR', 2*10)
        assert_len('S1udRR', 2*36)
        assert_len('S8udRR', 2*36)
        assert_len('S1ddRR', 2*45)
        assert_len('S8ddRR', 2*45)
        assert_len('S1udduRR', 2*36)
        assert_len('S8udduRR', 2*36)
        # compare total count to table 22
        ntot = sum([1 if isreal(k) else 2 for k in all_wc])
        self.assertEqual(ntot, 1+87+186+76+21+66+76+90+252+266+171+45+342+254+1+66+156+51+15+51+51+72+207+216+171+45+342+254+9+9+26+26)
