#!/usr/bin/env python3

"""Test program for the SFTP class."""

import unittest
import random

def randbytes(s):
    return bytes(random.getrandbits(8) for _ in range(s))

class TestSFTP(unittest.TestCase):
    def setUp(self):
        global sftp
        try:
            from obliv import sftp
            from . import get_ssh_info
        except ImportError:
            self.skipTest("Error importing sftp module. Maybe paramiko is not installed?")

        self.info = get_ssh_info.load_info()
        if self.info is None:
            self.skipTest('Could not load ssh info. Run "python3 -m tests.get_ssh_info" to set it up.')

        random.seed(0xf00dface)

    def test_sftp(self):
        n1, n2 = ['test_sftp_' + str(x) for x in random.sample(range(100,1000), 2)]
        v1, v2, v3 = [randbytes(random.randrange(100)) for _ in range(3)]

        # open and write (n1,v1) and (n2,v2)
        with sftp.SFTP(self.info) as s:
            s.write(n1, v1)
            s.write(n2, v2)

            # check
            self.assertEqual(s.read(n2), v2)
            self.assertEqual(s.read(n1), v1)

        # open again
        with sftp.SFTP(self.info) as s:
            # check
            self.assertEqual(s.read(n1), v1)
            self.assertEqual(s.read(n2), v2)

            # change something
            s.write(n1, v3)

            # check
            self.assertEqual(s.read(n1), v3)
            self.assertEqual(s.read(n2), v2)

        # open again
        with sftp.SFTP(self.info) as s:
            # check
            self.assertEqual(s.read(n2), v2)
            self.assertEqual(s.read(n1), v3)

            # clean-up
            s.delete(n1)
            s.delete(n2)


if __name__ == '__main__':
    unittest.main()
