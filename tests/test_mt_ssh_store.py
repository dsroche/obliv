#!/usr/bin/env python3

"""Test program for the fstore class."""

import unittest
import tempfile
import random

def randbytes(s):
    return bytes(random.getrandbits(8) for _ in range(s))

class TestSSHStore(unittest.TestCase):
    def setUp(self):
        global mt_ssh_store
        try:
            from obliv import mt_ssh_store
            from . import get_ssh_info
        except ImportError:
            self.skipTest("Error importing sftp module. Maybe paramiko is not installed?")

        self.info = get_ssh_info.load_info()
        if self.info is None:
            self.skipTest('Could not load ssh info. Run "python3 -m tests.get_ssh_info" to set it up.')

        self.dirname = "test_ssh_store"
        random.seed(0xf00dface)

    def test_ssh_store_small(self):
        n = 21 # number of files
        s = 213 # size of each file in bytes
        check = [randbytes(s) for _ in range(n)]
        
        with mt_ssh_store.mt_ssh_store(self.info, self.dirname) as fs:
            # insert everything
            for dat in check:
                fs.append(dat)

            # change 2 values
            i1, i2 = random.sample(range(n), 2)
            check[i1] = randbytes(s)
            check[i2] = randbytes(s)
            fs[i1] = check[i1]
            fs[i2] = check[i2]

            # check everything in random order
            for i in random.sample(range(n), n):
                self.assertEqual(check[i], fs[i])
        
        # re-open
        with mt_ssh_store.mt_ssh_store(self.info, self.dirname, size=n) as fs:
            # check everything in random order
            for i in random.sample(range(n), n):
                self.assertEqual(check[i], fs[i])

            # delete some things
            for _ in range(3):
                del check[-1]
                del fs[-1]
                n -= 1

            self.assertEqual(len(fs), n)

            # insert some new things
            for _ in range(5):
                check.append(randbytes(s))
                fs.append(check[-1])
                n += 1

            self.assertEqual(len(fs), n)

            # check everything in random order
            for i in random.sample(range(n), n):
                self.assertEqual(check[i], fs[i])

    def test_ssh_store_1thread(self):
        n = 100 # number of files
        m = 63 # something less than n
        s = 19 # size of each file in bytes
        check1 = [randbytes(s) for _ in range(n)]
        check2 = [randbytes(s) for _ in range(n)]

        with mt_ssh_store.mt_ssh_store(self.info, self.dirname, nthreads=1) as fs:
            # insert half
            fs.extend(check1[:m])

            # check first half
            for _ in range(n):
                i = random.randrange(m)
                self.assertEqual(check1[i], fs[i])

            # change first half
            for i in random.sample(range(m), m):
                fs[i] = check2[i]

            # insert second half
            for x in check1[m:]:
                fs.append(x)

            # check once
            for _ in range(n):
                i = random.randrange(n)
                if i < m:
                    self.assertEqual(check2[i], fs[i])
                else:
                    self.assertEqual(check1[i], fs[i])

            # change to check2 completely
            for i in random.sample(range(n), n):
                fs[i] = check2[i]

            # check again
            for _ in range(2*n):
                i = random.randrange(n)
                self.assertEqual(check2[i], fs[i])

            # delete everything
            while fs:
                del fs[-1]


if __name__ == '__main__':
    unittest.main()
