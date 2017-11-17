import unittest
import wcxf


class TestWET(unittest.TestCase):
    def test_afgv(self):
        basis = wcxf.Basis['WET', 'AFGV']
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
        self.assertEqual(len(basis.sectors['bsnunu']), 2*9)
        self.assertEqual(len(basis.sectors['db']), 57*2)
        self.assertEqual(len(basis.sectors['bdnunu']), 2*9)
        for l1 in ['e', 'mu', 'tau']:
            for l2 in ['e', 'mu', 'tau']:
                if l1 != l2:
                    self.assertEqual(len(basis.sectors['sb'+l1+l2]), 10)
                    self.assertEqual(len(basis.sectors['db'+l1+l2]), 10)
