#!/usr/bin/env python3

"""Test program for the stash implementations."""

import unittest
import random
from obliv.stash import SkipStash, DictStash, ListStash

def randbytes(s):
    return bytes(random.getrandbits(8) for _ in range(s))

class TestSFTP(unittest.TestCase):
    def setUp(self):
        random.seed(0xf00dface)

    def generic_test(self, cls):
        """Performs tests on the given stash class."""
        n = 10000 # how many items to insert

        stash = cls()
        copy = {}
        for i in range(n):
            ident = randbytes(3)
            val = str(i).encode()
            stash[ident] = val
            copy[ident] = val

        self.assertEqual(len(stash), len(copy))

        citer = iter(copy)
        for ii in range(n//10):
            ident = next(citer)
            val = ("changed " + str(ii)).encode()
            stash[ident] = val
            copy[ident] = val

        def krtest(k1, k2):
            self.assertEqual(set(stash.keys_range(k1,k2)), set(k for k in copy if k1 <= k < k2))

        krtest(next(citer), next(citer))
        krtest(next(citer), b'\xff' * 10)
        krtest(bytes(0), next(citer))

        todel = [next(citer) for _ in range(n//10)]
        for ident in todel:
            del stash[ident]
            del copy[ident]

        self.assertEqual(len(stash), len(copy))
        self.assertEqual(set(stash.items()), set(copy.items()))

    def test_skipstash(self):
        self.generic_test(SkipStash)

    def test_dictstash(self):
        self.generic_test(DictStash)

    def test_liststash(self):
        self.generic_test(ListStash)


if __name__ == '__main__':
    unittest.main()
