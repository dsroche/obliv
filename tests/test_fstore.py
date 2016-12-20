#!/usr/bin/env python3

"""Test program for the fstore class."""

import unittest
import tempfile
import random
from obliv import fstore

def randbytes(s):
    return bytes(random.getrandbits(8) for _ in range(s))

class TestFstore(unittest.TestCase):
    def setUp(self):
        tdobj = tempfile.TemporaryDirectory()
        self.addCleanup(tdobj.cleanup)
        self.dirname = tdobj.name
        random.seed(0xf00dface)

    def test_fstore_small(self):
        n = 21 # number of files
        s = 213 # size of each file in bytes
        check = [randbytes(s) for _ in range(n)]
        
        with fstore.fstore(self.dirname) as fs:
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
        with fstore.fstore(self.dirname) as fs:
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

    def test_fstore_sizes(self):
        maxn = 100 # max number of files
        s = 51 # size of each file in bytes
        check = [randbytes(s) for _ in range(maxn)]
        
        for _ in range(100):
            n = random.randrange(maxn)

            with fstore.fstore(self.dirname) as fs:
                # insert everything
                fs.extend(check[:n])

            with fstore.fstore(self.dirname) as fs:
                # check len, first and last
                self.assertEqual(n, len(fs))
                if n:
                    self.assertEqual(check[0], fs[0])
                    self.assertEqual(check[n-1], fs[-1])

                # delete everything
                while fs:
                    del fs[-1]

if __name__ == '__main__':
    unittest.main()
