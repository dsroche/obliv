#!/usr/bin/env python3

"""Test program for hirb.Hirb"""

import unittest
import random
from obliv import hirb
from obliv.nopcipher import PyRand

def randbytes(s):
    return bytes(random.getrandbits(8) for _ in range(s))

class TestHirb(unittest.TestCase):
    def setUp(self):
        random.seed(0xf00dface)
        self.rf = PyRand(0xbad00bad)

    def test_hirb(self):
        n = 400 # how many items to insert
        s = 10 # average size of each item

        rand_label = lambda: random.getrandbits(100)
        rand_value = lambda: randbytes(random.randrange(s-5, s+5))

        # create check data of size n
        check = {}
        while len(check) < n:
            check[str(rand_label())] = rand_value()

        # create hirb
        h = hirb.create_hirb(n, s, 1024, randfile=self.rf)
        hirb_height = h.height
        voram_height = h.oram.L
        opcount = 0

        # insert all
        for k,v in check.items():
            h[k] = v
            opcount += 1

        # check lookups
        for _ in range(n//4):
            k = random.choice(list(check))
            self.assertEqual(h[k], check[k])
            opcount += 1

        # change some stuff
        for _ in range(n//4):
            k = random.choice(list(check))
            check[k] = rand_value()
            h[k] = check[k]
            opcount += 1

        # remove some stuff
        for _ in range(n//8):
            k = random.choice(list(check))
            del check[k]
            del h[k]
            opcount += 1

        # check all lookups again
        self.assertEqual(len(h), len(check))
        for k,v in check.items():
            self.assertEqual(h[k], v)
            opcount += 1

        # check obliviousness
        self.assertTrue(all(k < 2*(voram_height+1) for k in h.oram.counts))
        self.assertEqual(sum(h.oram.counts.values()), opcount*hirb_height)


if __name__ == '__main__':
    unittest.main()
