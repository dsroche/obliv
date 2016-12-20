#!/usr/bin/env python3

from collections import MutableMapping
from bisect import bisect_left, insort_left
import obliv

class ListStash(MutableMapping):
    """Simple data structure to hold the stash.

    Stash must support a dictionary-like interface mapping positions to
    bytes objects. In addition, must support 1-dimensional range search
    of keys via the keys_range() method.

    See SkipStash for a solution with better asymptotic performance.
    """

    def __init__(self):
        super().__init__()
        self._store = []

    def __len__(self):
        return len(self._store)

    def find(self, key):
        """Searches for the given id tag in the stash.

        The index of that id is returned, or a KeyError is raised.
        """
        ind = bisect_left(self._store, (key, b''))
        if ind < len(self._store) and self._store[ind][0] == key:
            return ind
        else:
            raise KeyError("key not found in stash: " + str(key))

    def keys_range(self, start, stop):
        """Iterates over all keys in range(start, stop)"""
        ind = bisect_left(self._store, (start, b''))
        while ind < len(self._store) and self._store[ind][0] < stop:
            yield self._store[ind][0]
            ind += 1

    def __getitem__(self, key):
        return self._store[self.find(key)][1]

    def __setitem__(self, key, value):
        try:
            self._store[self.find(key)] = (key, value)
        except KeyError:
            insort_left(self._store, (key, value))

    def __delitem__(self, key):
        del self._store[self.find(key)]

    def __iter__(self):
        return (k for k,v in self._store)

class DictStash(dict):
    """Simple dictionary-based stash used for testing.

    Insertion, lookup, and removal are O(1) time, but the critical
    keys_range method is O(n).

    See SkipStash for a solution with better asymptotic complexity.
    """

    def keys_range(self, start, stop):
        """Iterates over all keys in range(start, stop)"""
        for k in self:
            if start <= k < stop:
                yield k

class SkipStash(MutableMapping):
    """Skip-list based dictionary for the stash.

    Maps positions to bytes objects.
    Suffixes of the positions will determine the heights.
    """

    def __init__(self):
        super().__init__()
        self._heads = [None]
        self._size = 0

    def __len__(self):
        return self._size

    @staticmethod
    def _height(ident):
        '''Returns the height calculated from the id string'''
        addon = 0
        for single_byte in reversed(ident):
            # bit magic below to get lowest set bit
            lowest_bit = (single_byte & -single_byte).bit_length() - 1
            if lowest_bit >= 0:
                return addon + lowest_bit
            addon += 8
        return addon
    
    def __getitem__(self, key):
        links = self._heads
        level = len(self._heads) - 1
        while level >= 0:
            if links[level] is None or key < links[level].key:
                level -= 1
            elif key > links[level].key:
                links = links[level].links
            else:
                return links[level].value
        raise KeyError("key not found in stash: " + obliv.voram.idstr(key))

    def __setitem__(self, key, value):
        key_level = self._height(key)
        while key_level >= len(self._heads):
            self._heads.append(None)

        links = self._heads
        level = len(self._heads) - 1
        while level >= key_level:
            if links[level] is None or key < links[level].key:
                level -= 1
            elif key > links[level].key:
                links = links[level].links
            else:
                links[level].value = value
                return
        
        # now we insert at this level and below
        level = key_level
        newnode = SkipNode(key, value, key_level+1)
        self._size += 1
        while True:
            newnode.links[level] = links[level]
            links[level] = newnode
            if level == 0: break
            level -= 1
            while links[level] is not None and links[level].key < key:
                links = links[level].links

    def __delitem__(self, key):
        links = self._heads
        level = len(self._heads) - 1
        removed = False

        while level >= 0:
            if links[level] is None or key < links[level].key:
                level -= 1
            elif key > links[level].key:
                links = links[level].links
            else:
                removed = True
                links[level] = links[level].links[level]
                level -= 1

        if not removed:
            raise KeyError("key not removed from stash: " + obliv.voram.idstr(key))
        else:
            self._size -= 1

    def __iter__(self):
        links = self._heads
        while links[0]:
            yield links[0].key
            links = links[0].links

    def keys_range(self, start, stop):
        """Iterates over all keys in range(start, stop)"""
        links = self._heads
        level = len(self._heads) - 1
        while level >= 0:
            if links[level] is None or start <= links[level].key:
                level -= 1
            elif start > links[level].key:
                links = links[level].links
        while links[0] is not None and links[0].key < stop:
            yield links[0].key
            links = links[0].links


class SkipNode:
    """A node in the SkipStash."""
    def __init__(self, key, value, height):
        self.key = key
        self.value = value
        self.links = [None] * height


