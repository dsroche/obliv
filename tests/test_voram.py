#!/usr/bin/env python3

"""Test program for voram.Voram"""

import unittest
import random
from obliv import voram
from obliv.nopcipher import PyRand

def randbytes(s):
    return bytes(random.getrandbits(8) for _ in range(s))

class TestVoram(unittest.TestCase):
    def setUp(self):
        random.seed(0xf00dface)
        self.rf = PyRand(0xbad00bad)

    def test_voram(self):
        n = 400 # how many items to insert
        s = 10 # average size of each item

        blobs = [randbytes(round(random.expovariate(1/s))) for _ in range(2*n)]

        # create voram
        v = voram.create_voram(n, s, randfile=self.rf)
        height = v.L
        opcount = 0

        # insert all
        refs = []
        check = {}
        for i in range(n):
            r = v.create()
            opcount += 1
            r.set(blobs[i])
            check[r.ident] = blobs[i]
            refs.append(r)

        # check obliviousness
        self.assertEqual({(height+1):opcount}, v.counts)

        # check some insertions
        for i in random.sample(range(n), n//2):
            oldident = refs[i].ident
            self.assertEqual(refs[i].get(), check[oldident])
            opcount += 1
            newident = refs[i].ident
            self.assertNotEqual(oldident, newident)
            check[newident] = check[oldident]
            del check[oldident]

        # check obliviousness
        self.assertEqual({(height+1):opcount}, v.counts)

        # re-assign some
        for i in range(n//2):
            ind = random.randrange(n)
            oldident = refs[ind].ident
            dat = blobs[n+i]
            refs[ind].set(dat)
            opcount += 1
            newident = refs[ind].ident
            self.assertNotEqual(oldident, newident)
            check[newident] = dat
            del check[oldident]

        # check obliviousness
        self.assertEqual({(height+1):opcount}, v.counts)

        # remove some
        for i in range(n//2):
            ind = random.randrange(len(refs))
            oldident = refs[ind].ident
            refs[ind].destroy()
            opcount += 1
            del check[oldident]
            del refs[ind]

        # check obliviousness
        self.assertEqual({(height+1):opcount}, v.counts)

        # add some
        for i in range(n//2):
            r = v.create()
            dat = blobs[n + n//2 + i]
            r.set(dat)
            opcount += 1
            check[r.ident] = dat
            refs.append(r)

        # check obliviousness
        self.assertEqual({(height+1):opcount}, v.counts)

        # check all
        self.assertEqual(set(r.ident for r in refs), set(check))
        for r in refs:
            oldid = r.ident
            self.assertEqual(r.get(), check[oldid])
            opcount += 1
            check[r.ident] = check[oldid]
            del check[oldid]

        # check obliviousness
        self.assertEqual({(height+1):opcount}, v.counts)

        # remove all
        for r in refs:
            del check[r.ident]
            r.destroy()
            opcount += 1

        # check everything was removed
        self.assertFalse(check)

        # check obliviousness
        self.assertEqual({(height+1):opcount}, v.counts)


if __name__ == '__main__':
    unittest.main()
