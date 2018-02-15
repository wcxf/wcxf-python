import unittest
import os
from tempfile import mkdtemp
import subprocess
import pylha
from shutil import rmtree


my_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(my_path, '..', 'data')


class TestSMEFTsim(unittest.TestCase):
    def test_wcxf2smeftsim(self):
        tmpdir = mkdtemp()
        # use default settings
        res = subprocess.run(['wcxf2smeftsim',
                              os.path.join(data_path, 'test.Warsaw_mass.yml')],
                             cwd=tmpdir)
        # check return code
        self.assertEqual(res.returncode, 0, msg="Command failed")
        # check if file is present
        outf = os.path.join(tmpdir, 'wcxf2smeftsim_param_card.dat')
        self.assertTrue(os.path.isfile, outf)
        # check if can be imported as LHA
        with open(outf, 'r') as f:
            out = pylha.load(f)
        # check string is not empty
        self.assertTrue(out)
        # remove tmpdir
        rmtree(tmpdir)
