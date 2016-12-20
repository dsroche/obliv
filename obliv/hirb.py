#!/usr/bin/env python3

"""HIRB tree with ORAM storage"""

import bisect
import random
import collections
import math
import hashlib
import pickle
import sys

from . import voram
from .progbar import ProgressBar

def estimate_B(nodesize, idlen, keylen, lenlen, hashlen, valsize):
    """Returns (B, LeafB) for the Hirb with ORAM parameters as specified"""
    node_data = nodesize - 2*keylen
    chunk_header = idlen + lenlen
    # node_data >= SIZE_FACTOR*(chunk_header + (B+1)*idlen + B*hash_len + B*valsize)
    B = ( (node_data - Hirb.SIZE_FACTOR * (chunk_header + idlen))
          // (Hirb.SIZE_FACTOR * (idlen + hashlen + valsize))
        )
    # node_data >= SIZE_FACTOR*(chunk_header + LeafB*hash_len + LeafB*valsize)
    LeafB = ( (node_data - Hirb.SIZE_FACTOR * chunk_header)
              // (Hirb.SIZE_FACTOR * (hashlen + valsize))
            )
    return B, LeafB


def create_hirb(items_limit, value_size, bucket_size,
    info_stream=None, simulate=False,
    # ORAM options
    levels=None, stash=None, storage=None, cipher=None, randfile=None,
    keylen=None, idlen=None, IV=None, immediate=True,
    # Hirb options
    initial_height=None, salt=None, hash_alg='sha1', initial_data={}, verbose=False):
    
    """Creates a new Hirb and a new underlying ORAM to store it.

    Parameters are automagically chosen for optimal wonderfulness.
    Only three parameters are required here:

    items_limit is the most number of items that will be stored in
    the Hirb. It's probably OK if you go slightly over this limit.

    value_size is the size, in bytes, of the largest value you expect
    to store in the Hirb. It's probably OK if you store a couple values
    that are bigger than it.

    bucket_size is the size of buckets in the to-be-created ORAM.

    If info_stream is not None, the details of ORAM and Hirb construction
    will be printed to it.

    If simulate is True, no Hirb or ORAM is actually created.

    The next set of options (levels, stash, storage, cipher, randfile,
    keylen, idlen, IV, immediate) default to some optimal-ish values,
    or are passed directly to the ORAM constructor if specified.

    The next set of options (initial_height, salt, hash_alg) default to
    some optimal-ish values, or are passed directly to the Hirb
    constructor if specified.

    initial_data is a dictionary containing the values to initially store
    in the HIRB.
    """
    # estimate the number of bits per value
    keylen = 32 if (keylen is None and cipher is None) else cipher.key_size[-1]
    lenlen = ((bucket_size - 2*keylen - 2).bit_length() + 7) // 8 # copied from ORAM
    hashlen = 20 if hash_alg is None else hashlib.new(hash_alg).digest_size

    B, LeafB = 1, 1
    while True:
        nblobs = math.ceil(items_limit*((LeafB-1)/LeafB**2 + 1/(B*LeafB)))
        idlen_try = idlen if idlen else voram.estimate_idlen(nblobs)
        nextB, nextLeafB = estimate_B(bucket_size, idlen_try, keylen, lenlen, hashlen, value_size)
        if nextLeafB <= LeafB:
            break
        B, LeafB = nextB, nextLeafB

    da_oram = voram.create_voram(
        blobs_limit=nblobs,
        blob_size=None, # note: ignored because bucket size is fixed
        info_stream=info_stream,
        simulate=simulate,
        levels=levels,
        nodesize=bucket_size,
        stash=stash,
        storage=storage,
        cipher=cipher,
        randfile=randfile,
        keylen=keylen,
        idlen=idlen,
        IV=IV,
        immediate=immediate,
        verbose=verbose
    )

    if initial_height is None:
        if B == 1:
            initial_height = voram.clog(items_limit/LeafB, 2) + 1
        else:
            initial_height = voram.clog((B-1)*items_limit/LeafB, B)

    if info_stream:
        print("""
Creating Hirb Tree with:
    height              {} levels
    leaf node size      {} items
    internal node size  {} items
    total max size      {} items
    expected node size  {:.1f} KB
""".format(
            initial_height+1,
            LeafB,
            B,
            items_limit,
            LeafB*(hashlen+value_size) / 1024,
        ), file=info_stream)
    
    if not simulate:
        return Hirb(da_oram, value_size,
            initial_height=initial_height,
            salt=salt,
            hash_alg=hash_alg,
            initial_data=initial_data,
            verbose=verbose
        )


class Hirb(collections.MutableMapping):
    """Class for secure B-tree.

    Works like the randomized B-tree, except:
    1) There is an ORAM storage back-end (must be provided).
    2) A secret seed is stored to generate random element heights.
    3) All operations fetch exactly 2*height nodes from the ORAM.

    Specifically, the access pattern on ANY operation is
    [read, read, write, write]^height.
    Only 2 nodes are stored in cache at any given time.

    NOTE: The root node, as well as other class instance information,
    is NOT stored in the voram. If you want to store the entire thing in
    ORAM, you have to put the Hirb instance itself in the voram, e.g.:
        my_hirb = Hirb(my_oram, 4, 4)
        my_hirb[10] = 20
        hirb_ref = my_oram.create()
        hirb_ref.set(my_hirb)
        del my_hirb
        # ...
        my_hirb = hirb_ref.get()
        my_hirb[10] # returns 20
    """

    """SIZE_FACTOR is a parameter that will affect the stash overflow.
    It is set to be the EXPECTED number of HIRB tree nodes that would fit
    in a single ORAM bucket.
    If it's too small, the ORAM stash will overflow (bad).
    If it's too large, the ORAM will be unnecessarily large and wasteful.
    """
    SIZE_FACTOR = 6

    def __init__(self, da_oram, valsize, 
            initial_height = 0, salt = None, hash_alg = 'sha1',
            initial_data = {}, verbose=False):
        """Creates a new HIRB, using the specified ORAM instance.
        
        valsize is the "typical" maximum size, in bytes, of any
        value that will be stored in the ORAM. It's okay if a few insertions
        go over this bound, but if you do that too much the ORAM stash will
        overflow.

        salt is the salt to use when generating pseudo-random heights.
        If salt is None or not given, a random salt will be generated.

        hash_alg should be a hash function defined in the hashlib library.

        initial_data is a map containing the initial key-value settings
        for the HIRB. NOTE: an efficient algorithm is used to set up the
        HIRB with this data, that could reveal the number of HIRB tree nodes
        to an attacker. If this is undesirable, use the update() method instead
        after creating the HIRB.

        verbose prints information about time left for initialization

        """
        if valsize < 1:
            raise ValueError("valsize must be positive")
        if initial_height < 0:
            raise ValueError("initial_height must be nonnegative")

        self._oram = da_oram

        self._hash_alg = hash_alg
        self._hasher = hashlib.new(self._hash_alg)
        self._hash_len = self._hasher.digest_size

        if salt is None:
            self._salt = self.oram.randfile.read(self._hasher.block_size)
        else: self._salt = salt
        self._hasher.update(self._salt)

        # calculating B so that SIZE_FACTOR number of size-B nodes fit an ORAM node.
        # node_data >= SIZE_FACTOR*(chunk_header + (B+1)*idlen + B*hash_len + B*valsize)
        self._B, self._LeafB = estimate_B(
            self.oram.Z, self.oram.idlen, self.oram._keylen, self.oram._lenlen,
            self._hash_len, math.ceil(valsize))
      
        if (self._B < 1):
            raise ValueError("""
Not enough room in ORAM nodes to store this HIRB.
If B=1 (the minimum), then we need {} size-1 nodes to fit in an ORAM node.
But a size-1 node has {} chunk header bytes, {} bytes for 2 child idents,
{} bytes for the hash and {} bytes for the value.
That's a total of {} bytes, but the provided ORAM can only store {} bytes of data.
                """.strip().format(
                    self.SIZE_FACTOR,
                    self.oram.chunk_header,
                    2*self.oram.idlen,
                    self._hash_len,
                    valsize,
                    self.SIZE_FACTOR*(self.oram.chunk_header+2*self.oram.idlen+self._hash_len+valsize),
                    self.oram.node_data
                ))

        self._size = 0          # the actual number of items stored in the tree
        # height will increase when we reach this size.
        self._grow_at = math.ceil(self._LeafB / 4)

        # create nodes for leftmost path of the tree
        self._height = 0
        tower = []
        while self._height < initial_height:
            tower.append(HirbInternal(child=None))
            self._grow_at *= self._B
            self._height += 1
        tower.append(HirbLeaf())

        if verbose:
            pb = ProgressBar(initial_data.keys())
            pb.start()
        
        # insert initial values and write nodes to ORAM as needed
        for lhash, label in sorted(
                (self.get_hash(label), label) for label in initial_data.keys()):

            if verbose:
                pb += 1

            lheight = self.get_height(lhash)
            # lheight nodes need to be closed off and written to ORAM.
            for _ in range(lheight):
                child_ref = self.oram.create(read=False, sync=True)
                child_ref.set(tower.pop())
                tower[-1].children.append(child_ref.ident)
            # add this item to the node at the proper height
            tower[-1].lhashes.append(lhash)
            tower[-1].values.append(initial_data[label])
            # now generate lheight new nodes and continue
            for h in range(lheight):
                tower.append(HirbLeaf() if h == lheight-1 else HirbInternal(child=None))
            assert len(tower) == self._height + 1

        # close off and write back all nodes in the tower
        while len(tower) > 1:
            child_ref = self.oram.create(read=False, sync=True)
            child_ref.set(tower.pop())
            tower[-1].children.append(child_ref.ident)
        self._root = tower[0]

        if verbose:
            pb.finish()


    @property
    def height(self):
        return self._height

    @property
    def oram(self):
        return self._oram

    @property
    def store(self):
        return self.oram.store

    def __len__(self):
        """Used by the built-in "len(...)" function to get the size"""
        return self._size

    def get_hash(self, label):
        """Returns the hash of the given label.

        This is used to determine the height of that label in the HIRB,
        and the same hash is also used as the sorting order for insertion
        and retrieval.
        """
        labelhash = self._hasher.copy()
        labelhash.update(pickle.dumps(label))
        return labelhash.digest()

    def get_height(self, label_hash):
        """Returns the pseudo-randomly-chosen height of the given label hash."""
        prng = random.Random(label_hash)
        if self.height == 0 or prng.randrange(self._LeafB) > 0:
            return 0 # most things go in a leaf!
        ht = 1
        while ht < self.height and prng.randrange(self._B) == 0:
            ht += 1
        return ht

    def lookup(self, label, disguise=True):
        """Performs ORAM fetches to find the given label.

        Returns a tuple (value, found), where found is True/False depending
        on whether the thing was actually found in the HIRB.

        Only height ORAM fetches are necessary for this operation.
        If disguise is True (the default), height additional "dummy" operations
        are performed so that this looks identical to any other HIRB operation.
        """
        curnode = self._root
        level = self.height
        lhash = self.get_hash(label)
        retval = None
        cur_ref = None
        while True:
            # search for label in this node
            child_ind = bisect.bisect_left(curnode.lhashes, lhash)
            if child_ind < len(curnode) and curnode.lhashes[child_ind] == lhash:
                # lhash was found in this node
                retval = (curnode.values[child_ind], True)
                break
            if level == 0:
                # reached a leaf node without finding it; it's not there
                retval = (None, False)
                break
            level -= 1
            child_ref = self.oram.get_ref(curnode.children[child_ind], sync=False)
            # the next line writes the NEW ident for the child, before fetching it
            curnode.children[child_ind] = child_ref.pregen()
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()
            curnode = child_ref.get()
            cur_ref = child_ref
            if disguise:
                self.oram.dummy_op(sync=False)
        self.oram.finalize()

        # item found in the middle of tree; do extra ops the rest of the way down.
        for _ in range(level):
            self.oram.dummy_op(sync=False)
            if disguise: self.oram.dummy_op(sync=False)
            self.oram.finalize()

        return retval

    def __contains__(self, label):
        """Used by Python's "in" operator to test membership"""
        return self.lookup(label)[1]

    def __getitem__(self, label):
        """Used by the [] operator when accessed as an r-value"""
        got = self.lookup(label)
        if got[1]:
            return got[0]
        else:
            raise KeyError(str(label) + " is not in the HIRB.")

    def __setitem__(self, label, value):
        """Used by the [] operator when accessed as an l-value.
        
        Always performs exactly 2*height ORAM accesses.
        """
        # If size is too big, we grow the height of the whole HIRB
        if self._size == self._grow_at:
            oldroot = self._root
            oldroot_ref = self.oram.create(read=False, sync=False)
            self._root = HirbInternal(oldroot_ref.ident)
            self._height += 1
            self._grow_at *= self._B

            # go through label hashes in root and decide whether each gets promoted
            index = 0
            while index < len(oldroot.lhashes):
                rootlhash = oldroot.lhashes[index]
                newlevel = self.get_height(rootlhash)
                assert newlevel >= self._height - 1
                if newlevel == self._height:
                    self._root.lhashes.append(rootlhash)
                    self._root.values.append(oldroot.values[index])
                    nextroot = oldroot.split(rootlhash)
                    oldroot_ref.set(oldroot)
                    oldroot = nextroot
                    oldroot_ref = self.oram.create(read=False, sync=False)
                    self._root.children.append(oldroot_ref.ident)
                    index = 0
                else:
                    index += 1
            oldroot_ref.set(oldroot)

        self.oram.finalize()
        lhash = self.get_hash(label)
        thislevel = self.get_height(lhash)

        curnode = self._root
        level = self._height
        cur_ref = None

        # First go down the tree to the level of insertion
        while level > thislevel:
            child_ind = bisect.bisect(curnode.lhashes, lhash)
            child_ref = self.oram.get_ref(curnode.children[child_ind], sync=False)
            # the next line writes the NEW ident for the child, before fetching it
            curnode.children[child_ind] = child_ref.pregen()
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()
            curnode = child_ref.get()
            cur_ref = child_ref
            level -= 1
            self.oram.dummy_op(sync=False)

        child_ind = bisect.bisect_left(curnode.lhashes, lhash)
        if child_ind < len(curnode) and curnode.lhashes[child_ind] == lhash:
            # the lhash is already in the HIRB; just change the value.
            curnode.values[child_ind] = value
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()
            for _ in range(level):
                self.oram.dummy_op(sync=False)
                self.oram.dummy_op(sync=False)
                self.oram.finalize()
            return

        # insert the lhash at this level
        curnode.lhashes.insert(child_ind, lhash)
        curnode.values.insert(child_ind, value)
        self._size += 1

        # the next loop splits nodes down to the leaf level
        sibling_ref = None
        while level > 0:
            # create a new Ref for the node that WILL be split off the child
            stepchild_ident = self.oram.new_ident()
            if sibling_ref:
                sibling.children[0] = stepchild_ident
                sibling_ref.set(sibling)
            else:
                curnode.children.insert(child_ind+1, stepchild_ident)

            # set new ident for the next child down
            child_ref = self.oram.get_ref(curnode.children[child_ind], sync=False)
            curnode.children[child_ind] = child_ref.pregen()
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()

            # go down a level
            curnode = child_ref.get()
            cur_ref = child_ref
            sibling_ref = self.oram.create(sync=False, ident=stepchild_ident)
            level -= 1

            # split this node
            sibling = curnode.split(lhash)
            child_ind = bisect.bisect_left(curnode.lhashes, lhash)

        if cur_ref:
            cur_ref.set(curnode)
            if sibling_ref:
                sibling_ref.set(sibling)
            self.oram.finalize()


    def __delitem__(self, label):
        """Used by the built-in del operator, like "del T[0]".
        
        Always performs exactly 2*height ORAM accesses.
        """
        self.oram.finalize()

        curnode = self._root
        level = self.height
        cur_ref = None
        child_ind = None
        lhash = self.get_hash(label)

        while True:
            # search for lhash in this node
            child_ind = bisect.bisect_left(curnode.lhashes, lhash)
            if child_ind < len(curnode) and curnode.lhashes[child_ind] == lhash:
                # lhash was found in this node
                del curnode.lhashes[child_ind]
                del curnode.values[child_ind]
                self._size -= 1
                break
            if level == 0:
                # reached a leaf node without finding it; it's not there
                raise KeyError(str(label) + " not found in HIRB to delete")

            child_ref = self.oram.get_ref(curnode.children[child_ind], sync=False)
            # the next line writes the NEW ident for the child, before fetching it
            curnode.children[child_ind] = child_ref.pregen()
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()
            curnode = child_ref.get()
            cur_ref = child_ref
            level -= 1
            self.oram.dummy_op(sync=False)

        # now the lhash has just been deleted from curnode
        # the next loop merges all the children, down to the leaves
        sibling_ref = None
        while level > 0:
            if sibling_ref:
                sibling_ref.destroy()
                sibling_ref = self.oram.get_ref(stepchild_ident, sync=False)
            else:
                sibling_ref = self.oram.get_ref(curnode.children[child_ind+1], sync=False)
                del curnode.children[child_ind+1]

            # set new ident for the next child down
            child_ref = self.oram.get_ref(curnode.children[child_ind], sync=False)
            curnode.children[child_ind] = child_ref.pregen()
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()

            # go down a level
            curnode = child_ref.get()
            cur_ref = child_ref
            sibling = sibling_ref.get()
            level -= 1

            # merge curnode with sibling
            child_ind = len(curnode)
            stepchild_ident = curnode.merge(sibling)

        if cur_ref:
            cur_ref.set(curnode)
            if sibling_ref:
                sibling_ref.destroy()
            self.oram.finalize()


    def __iter__(self):
        """Iteration over a HIRB inherently leaks information"""
        raise NotImplementedError("can't iterate over ORAM-backed HIRB.")

    def __getstate__(self):
        """Used by pickle.dump() serialization"""
        state = self.__dict__.copy()
        del state['_hasher']
        del state['_hash_len']
        return state

    def __setstate__(self, state):
        """Used by pickle.load() unserialization"""
        self.__dict__.update(state)
        self._hasher = hashlib.new(self._hash_alg)
        self._hash_len = self._hasher.digest_size
        self._hasher.update(self._salt)


class HirbLeaf:
    """A leaf node in a HIRB Tree."""

    def __init__(self):
        self.lhashes = []
        self.values = []

    def __len__(self):
        return len(self.lhashes)

    def split(self, wedge):
        """Splits this node according to the given lhash, and returns the new sibling.
        
        If wedge is present in the node, it is removed from this node and the new
        sibling.
        """

        index = bisect.bisect_left(self.lhashes, wedge)
        if index < len(self.lhashes) and self.lhashes[index] == wedge:
            copy_from = index + 1
        else:
            copy_from = index

        sibling = HirbLeaf()
        sibling.lhashes.extend(self.lhashes[copy_from:])
        sibling.values.extend(self.values[copy_from:])
        del self.lhashes[index:]
        del self.values[index:]

        return sibling

    def merge(self, sibling):
        self.lhashes.extend(sibling.lhashes)
        self.values.extend(sibling.values)
        return None

class HirbInternal(HirbLeaf):
    """An internal node in a HIRB Tree."""

    def __init__(self, child):
        super(HirbInternal, self).__init__()
        if child is None:
            self.children = []
        else:
            self.children = [child]

    def split(self, wedge):
        """Splits this node according to the given lhash, and returns the new sibling.
        
        If wedge is present in the node, it is removed from this node and the new
        sibling.
        """

        index = bisect.bisect_left(self.lhashes, wedge)
        if index < len(self.lhashes) and self.lhashes[index] == wedge:
            copy_from = index + 1
        else:
            copy_from = index

        sibling = HirbInternal(self.children[copy_from])
        sibling.lhashes.extend(self.lhashes[copy_from:])
        sibling.values.extend(self.values[copy_from:])
        sibling.children.extend(self.children[copy_from+1:])
        del self.lhashes[index:]
        del self.values[index:]
        del self.children[index+1:]

        return sibling

    def merge(self, sibling):
        self.lhashes.extend(sibling.lhashes)
        self.values.extend(sibling.values)
        self.children.extend(sibling.children[1:])
        return sibling.children[0]
