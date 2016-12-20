#!/usr/bin/python3

'''No-op ciphers and related methods.

The stuff here is a drop-in replacement for pycrypto's cipher
classes. The utility is only to improve efficiency for testing
purposes - it's not even that useful for debugging because it
is literally a no-op cipher!
'''

import random

class NopCipher:
    '''A replacement for PyCrypto's Crypto.Cipher.AES.
    
    By default, the parameters imitate AES256.
    '''

    key_size = (32,)
    block_size = 16
    IV = b'\x00' * 16

    MODE_CBC = True
    MODE_CFB = True
    MODE_ECB = True

    @classmethod
    def new(cls, key=None, mode=None, IV=None):
        return NopCipher()

    def decrypt(self, string):
        return string

    def encrypt(self, string):
        return string

class PyRand:
    '''A file-like view on Python's random number generator.
    
    This is compatible with PyCrypto's Crypto.Random.'''

    def __init__(self, seed = None):
        self._rand = random.Random(seed)

    def read(self, nbytes):
        return bytes(self._rand.randrange(255) for _ in range(nbytes))

