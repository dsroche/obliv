#!/usr/bin/env python3

"""Ordered HIRB for range searches"""

import abc
import random
import collections
import math
import hashlib
import pickle
import sys
import functools

from obliv import voram
from obliv import progbar

@functools.total_ordering
class NEG_INF:
    """A dummy class whose objects are less than anything."""
    def __le__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, NEG_INF)

@functools.total_ordering
class POS_INF:
    """A dummy class whose objects are greater than anything."""
    def __ge__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, POS_INF)

def estimate_B(nodesize, idlen, keylen, lenlen, labsize, hashlen, valsize):
    """Returns (B, LeafB) for the Ohirb with ORAM parameters as specified"""
    node_data = nodesize - 2*keylen
    chunk_header = idlen + lenlen

    # node_data >= SIZE_FACTOR*(chunk_header + (B+1)*idlen + B*labsize + B*hashlen)
    B = ((node_data - Ohirb.SIZE_FACTOR * (chunk_header + idlen))
         / (Ohirb.SIZE_FACTOR * (idlen + labsize + hashlen)))
    if B < 1:
        raise ValueError("Need ORAM blocksize at least {} bytes to fit {} internal OHIRB nodes."
            .format(Ohirb.SIZE_FACTOR * (chunk_header + 2*idlen + labsize + hashlen), Ohirb.SIZE_FACTOR))

    # node_data >= SIZE_FACTOR*(chunk_header + LeafB*labsize + LeafB*valsize)
    LeafB = ((node_data - Ohirb.SIZE_FACTOR * chunk_header)
             / (Ohirb.SIZE_FACTOR * (labsize + valsize)))
    if LeafB < 1:
        raise ValueError("Need ORAM blocksize at least {} bytes to fit {} OHIRB leaf nodes."
            .format(Ohirb.SIZE_FACTOR * (chunk_header + labsize + valsize), Ohirb.SIZE_FACTOR))

    return B, LeafB

def create_ohirb(items_limit, label_size, value_size, bucket_size,
    info_stream = sys.stdout, simulate=False,
    # oram options
    levels=None, stash=None, storage=None, cipher=None, randfile=None,
    keylen=None, idlen=None, IV=None, immediate=None,
    # hirb options
    initial_height=None, salt=None, hash_alg='md5', initial_data={}, verbose=False):
    
    """Creates a new OHirb and a new underlying ORAM to store it.

    Parameters are automagically chosen for optimal wonderfulness.
    Only three parameters are required here:

    items_limit is the most number of items that will be stored in
    the Ohirb. It's probably OK if you go slightly over this limit.

    label_size and value_size are respectively the size in bytes of
    a "typical" label and value that you expect to be stored in the OHirb.

    bucket_size is the size of buckets in the to-be-created ORAM.

    If info_stream is not None, the details of ORAM and Ohirb construction
    will be printed to it.

    If simulate is True, no Ohirb or ORAM is actually created.

    The next set of options (levels, stash, storage, cipher, randfile,
    keylen, idlen, IV, immediate) default to some optimal-ish values,
    or are passed directly to the ORAM constructor if specified.

    The next set of options (initial_height, salt, hash_alg) default to
    some optimal-ish values, or are passed directly to the Ohirb
    constructor if specified.

    initial_data is a list of (label,value) pairs containing the entries
    to initially store in the HIRB.
    """
    # estimate the number of bits per value
    keylen = 32 if (keylen is None and cipher is None) else cipher.key_size[-1]
    lenlen = ((bucket_size - 2*keylen - 2).bit_length() + 7) // 8 # copied from ORAM
    hashlen = 16 if hash_alg is None else hashlib.new(hash_alg).digest_size

    B, LeafB = 2.0, 1.0
    while True:
        leaves = max(1.0, items_limit / LeafB)
        internals = (leaves - 1) / (B - 1)
        nblobs = round(leaves + internals)
        idlen_try = idlen if idlen else voram.estimate_idlen(nblobs)
        nextB, nextLeafB = estimate_B(bucket_size, idlen_try, keylen, lenlen, label_size, hashlen, value_size)
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
        if leaves <= 1:
            initial_height = 0
        else:
            initial_height = voram.clog(leaves, B)

    if info_stream:
        print("""
Creating OHirb Tree with:
    height              {} levels
    leaf node size      {:.3f} items
    internal node size  {:.3f} items
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
        return Ohirb(da_oram, label_size, value_size,
            initial_height=initial_height,
            salt=salt,
            hash_alg=hash_alg,
            initial_data=initial_data,
            verbose=verbose
        )


class Ohirb:
    """Class for Ordered History-Independent Randomized B-tree.

    Just like the HIRB, but entries are sorted first by the label itself,
    and range operations are supported.
    """

    SIZE_FACTOR = 6

    def __init__(self, da_oram, labelsize, valsize, 
            initial_height = 0, salt = None, hash_alg = 'md5',
            initial_data = [], verbose=False):
        """Creates a new OHIRB, using the specified ORAM instance.
        
        labsize and valsize are (respectively) the "typical" maximum number
        of bytes of any label or value that will be stored.

        salt is the salt to use when generating pseudo-random heights.
        If salt is None or not given, a random salt will be generated.

        hash_alg should be a hash function defined in the hashlib library.

        initial_data is a map containing the initial key-value settings
        for the OHIRB. 
        
        verbose prints information about time left for initialization
        """
        if labelsize < 1:
            raise ValueError("labelsize must be positive")
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
            self._hash_len, labelsize, valsize)
      
        self._size = 0
        self._grow_at = self._LeafB

        # create nodes for leftmost path of the tree
        self._height = 0
        tower = [OhirbLeaf()]
        trefs = []
        while self._height < initial_height:
            trefs.append(self.oram.create())
            tower.append(OhirbInternal(trefs[-1].ident))
            self._grow_at *= self._B
            self._height += 1

        sdata = [(label, self.get_hash(label, value), value) 
                 for (label, value) in initial_data]
        sdata.sort()

        has_pbar = verbose and sdata
        if has_pbar:
            pbar = progbar.ProgressBar(len(sdata))
            pbar.start()

        for label, lvhash, value in sdata:
            # insert into leaf node
            tower[0].insert(label, value, self.get_hash)
            self._size += 1

            height = self.get_height(lvhash)
            assert height < len(tower)
            # height nodes need to be closed off and written to ORAM.
            for i in range(height):
                trefs[i].set(tower[i])
                trefs[i] = self.oram.create(read=False)
                if i == 0:
                    tower[i] = OhirbLeaf()
                else:
                    tower[i] = OhirbInternal(trefs[i-1].ident)
            if height > 0:
                # add to internal node
                tower[height].insert(len(tower[height]), label, lvhash, trefs[height-1].ident)
            if has_pbar: pbar += 1

        # close off and write back nodes in the tower
        for i in range(self._height):
            trefs[i].set(tower[i])

        self._root = tower[self._height]
        self.oram.finalize()

        if has_pbar: pbar.finish()


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

    def get_hash(self, label, value):
        """Returns the hash of the given label/value pair.
        """
        myhash = self._hasher.copy()
        myhash.update(pickle.dumps(label))
        myhash.update(pickle.dumps(value))
        return myhash.digest()

    def get_height(self, label_hash):
        """Returns the pseudo-randomly-chosen height of the given label hash."""
        prng = random.Random(label_hash)
        if self.height == 0 or prng.random() * self._LeafB >= 1:
            return 0 # most things go in a leaf!
        ht = 1
        while ht < self.height and prng.random() * self._B < 1:
            ht += 1
        return ht

    def _descend(self, node, ind):
        """Goes down one level from the given node to the child at the given index.

        The node must be an internal node. A vORAM reference to the relevant
        child is returned, and also pregen() is called to update the reference
        in the parent node.
        """
        oldid = node.get_child(ind)
        ref = self.oram.get_ref(oldid, sync=False)
        node.set_child(ind, ref.pregen())
        return ref
    
    def grow(self):
        """Height is increased by 1."""
        old = self._root
        oldref = self.oram.create(read=False, sync=False)
        self._root = OhirbInternal(oldref.ident)
        self._height += 1
        self._grow_at *= self._B

        # go through contents of old root and see what gets promoted
        i = 0
        while i < len(old):
            newlevel = self.get_height(old.get_lvhash(i, self.get_hash))
            assert self._height - 1 <= newlevel <= self._height
            if newlevel == self._height:
                # promotion to the new root
                label = old.get_label(i)
                lvhash = old.get_lvhash(i, self.get_hash)
                if isinstance(old, OhirbInternal):
                    childid = old.remove(i)
                    right, check = old.split(i, lambda : childid)
                    assert check == childid
                else:
                    right, _ = old.split(i+1, None)
                oldref.set(old)
                old = right
                oldref = self.oram.create(read=False, sync=False)
                self._root.insert(len(self._root), label, lvhash, oldref.ident)
                i = 0
            else:
                i += 1
        oldref.set(old)
        self.oram.finalize()

    def insert(self, label, value):
        """Inserts a new (label,value) pair into the OHIRB.

        Always performs exactly 2*height ORAM accesses.
        """
        # If size is too big, we grow the height of the whole HIRB
        if self._size >= self._grow_at:
            self.grow()

        # now to the regular insertion
        lvhash = self.get_hash(label, value)
        thislevel = self.get_height(lvhash)

        curnode = self._root
        level = self._height
        cur_ref = None

        # First go down the tree to the level of insertion
        while level > thislevel:
            child_ref = self._descend(curnode, curnode.find(label, lvhash, self.get_hash))
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()
            curnode = child_ref.get()
            cur_ref = child_ref
            level -= 1
            self.oram.dummy_op(sync=False)

        if level > 0:
            # insert into internal node
            ind = curnode.find(label, lvhash, self.get_hash)
            right_id = self.oram.new_ident()
            try:
                curnode.insert(ind, label, lvhash, right_id)
            except ValueError:
                # duplicate insertion
                if cur_ref:
                    cur_ref.set(curnode)
                    self.oram.finalize()
                for _ in range(level):
                    self.oram.dummy_op(sync=False)
                    self.oram.dummy_op(sync=True)
                raise

            while level > 0:
                child_ref = self._descend(curnode, ind)
                if cur_ref:
                    cur_ref.set(curnode)
                    self.oram.finalize()

                # go down a level
                curnode = child_ref.get()
                cur_ref = child_ref
                right_ref = self.oram.create(sync=False, ident=right_id)
                level -= 1

                # split this node to create new right sibling node
                ind = curnode.find(label, lvhash, self.get_hash)
                right_node, right_id = curnode.split(ind, self.oram.new_ident)
                right_ref.set(right_node)

        # now insert the actual item in the leaf
        assert isinstance(curnode, OhirbLeaf)
        try:
            curnode.insert(label, value, self.get_hash)
            self._size += 1
        except ValueError:
            raise
        finally:
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()

    def range_search(self, label1, label2, limit=1, val1=None, val2=None):
        """Performs a range search between (label1, val1) and (label2, val2).

        If either label or value is None, that indicates an open-ended search.

        limit is used to control the cost of searching.
        It is *not* a strict upper or lower bound on the size of the returned set.
        If the range contains at least limit items,
        then the expected number of returned items is at least limit.

        Two range searches with the same limit are indistinguishable
        from each other, and from some number of other operations depending on
        the limit and the OHIRB parameters.

        Returned is a pair: a list of (label, value) pairs in the range,
        and a True/False value indicating whether the entire range was traversed.
        """
        if label1 is None or val1 is None:
            if label1 is None:
                label1 = NEG_INF()
            hash1 = NEG_INF()
        else:
            hash1 = self.get_hash(label1, val1)

        if label2 is None or val2 is None:
            if label2 is None:
                label2 = POS_INF()
            hash2 = POS_INF()
        else:
            hash2 = self.get_hash(label2, val2)

        # check for invalid range
        if label1 > label2 or (label1 == label2 and hash1 > hash2):
            raise ValueError("range endpoints are inverted; no results possible")

        # this is the number of extra read/writes in order to get enough results
        remain_pairs = round(1 + (limit - 1) / self._LeafB / 2) * self.height

        lnode = rnode = self._root
        lref = rref = None
        middles = []
        all_found = True

        for remaining_levels in range(self.height, 0, -1):
            # first deal with the endpoints
            if lref:
                rnode = lnode = lref.get()
                remain_pairs -= 1
            if rref:
                rnode = rref.get()
            ind1 = lnode.find(label1, hash1, self.get_hash)
            next_lref = self._descend(lnode, ind1)
            ind2 = rnode.find(label2, hash2, self.get_hash)
            if lnode != rnode or ind1 != ind2:
                next_rref = self._descend(rnode, ind2)
            else:
                next_rref = None
            
            # determine the number of *new* descent paths that could be added
            remain_pairs -= (len(middles) + 1) // 2
            addpaths = 2 * (remain_pairs // remaining_levels) - len(middles) - 2
            next_mids = []

            # pick next middles under endpoints
            pool = []
            if lnode == rnode:
                pool.extend((lnode, j) for j in range(ind1+1, ind2))
            else:
                pool.extend((lnode, j) for j in range(ind1+1, len(lnode)+1))
                pool.extend((rnode, j) for j in range(0, ind2))
            all_found = all_found and addpaths >= len(pool)
            for u, j in random.sample(pool, min(addpaths, len(pool))):
                next_mids.append(self._descend(u, j))
                addpaths -= 1

            # write back the endpoints
            if lref:
                lref.set(lnode)
                if rref:
                    rref.set(rnode)
                else:
                    self.oram.dummy_op(sync=False)
                self.oram.finalize()
            lref = next_lref
            rref = next_rref
            
            # now deal with the middle nodes, 2 at a time
            for j in range(0, len(middles), 2):
                ref1 = middles[j]
                node1 = ref1.get()
                pool = [(node1, k) for k in range(len(node1)+1)]
                addpaths += 1
                if j + 1 < len(middles):
                    ref2 = middles[j+1]
                    node2 = ref2.get()
                    pool.extend((node2, k) for k in range(len(node2)+1))
                    addpaths += 1
                else:
                    ref2 = node2 = None
                all_found = all_found and addpaths >= len(pool)
                for u, k in random.sample(pool, min(addpaths, len(pool))):
                    next_mids.append(self._descend(u, k))
                    addpaths -= 1
                ref1.set(node1)
                if ref2:
                    ref2.set(node2)
                else:
                    self.oram.dummy_op(sync=False)
                self.oram.finalize()

            middles = next_mids
            random.shuffle(middles)

        # gather results from leaf nodes
        if lref is None:
            # one node only
            assert rref is None and not middles and all_found
            return list(lnode.range_search(label1, hash1, label2, hash2, self.get_hash)), True

        middles = [lref] + middles
        if rref:
            middles.append(rref)
        results = []

        for j in range(0, len(middles), 2):
            node1 = middles[j].get()
            results.extend(node1.range_search(label1, hash1, label2, hash2, self.get_hash))
            if j + 1 < len(middles):
                node2 = middles[j+1].get()
                results.extend(node2.range_search(label1, hash1, label2, hash2, self.get_hash))
            else:
                self.oram.dummy_op(sync=False)
            self.oram.finalize()
            remain_pairs -= 1

        assert remain_pairs >= 0
        for _ in range(remain_pairs):
            self.oram.dummy_op(sync=False)
            self.oram.dummy_op(sync=True)

        return results, all_found


    def remove(self, label, value=None):
        """Removes and returns the the given (label, value) pair.

        If the value is unspecified or None, an arbitrary element with
        the given label is removed.

        A ValueError is raised if no such element exists.
        """
        if value is None:
            lvhash = NEG_INF()
        else:
            lvhash = self.get_hash(label, value)

        curnode = self._root
        level = self._height
        cur_ref = None
        merge_ref = None
        ind = None
        ind_check = None

        while level > 0:
            # check if item exists on this level
            ind = curnode.find(label, lvhash, self.get_hash)
            if merge_ref:
                assert ind == ind_check
            elif (ind < len(curnode) and curnode.get_label(ind) == label and
                    (value is None or lvhash == curnode.get_lvhash(ind, self.get_hash))):
                # remove from node
                lvhash = curnode.get_lvhash(ind, self.get_hash)
                merge_ref = self.oram.get_ref(curnode.remove(ind), sync=False)

            # move down to next level
            child_ref = self._descend(curnode, ind)
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()
            level -= 1

            curnode = child_ref.get()
            cur_ref = child_ref
            if merge_ref:
                ind_check = len(curnode)
                merge_node = merge_ref.get()
                merge_ref.destroy()
                merge_ref = self.oram.get_ref(curnode.merge(merge_node), sync=False)
            else:
                self.oram.dummy_op(sync=False)

        # find and remove item from leaf node
        ind = curnode.find(label, lvhash, self.get_hash)
        if merge_ref:
            assert ind == ind_check - 1
        if (ind >= len(curnode) or curnode.get_label(ind) != label or
                (value is not None and lvhash != curnode.get_lvhash(ind, self.get_hash))):
            # not in the leaf
            if cur_ref:
                cur_ref.set(curnode)
                self.oram.finalize()
            raise ValueError("item to be deleted was not found in Ohirb.")

        res = curnode.get_label(ind), curnode.get_value(ind)
        curnode.remove(ind)
        self._size -= 1
        if cur_ref:
            cur_ref.set(curnode)
            self.oram.finalize()
        return res

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

class OhirbNodeBase(metaclass=abc.ABCMeta):
    """Base class for internal and leaf nodes."""

    def __init__(self):
        self.labels = []

    def __len__(self):
        return len(self.labels)
    
    def get_label(self, index):
        return self.labels[index]

    @abc.abstractmethod
    def get_lvhash(self, index, hasher): pass

    def find(self, label, lvhash, hasher):
        """Works similarly to bisect.bisect_left."""
        lo, hi = 0, len(self.labels)
        while lo < hi:
            mid = (lo + hi) // 2
            if (self.labels[mid] < label
                or (self.labels[mid] == label
                    and self.get_lvhash(mid, hasher) < lvhash)):
                lo = mid + 1
            else:
                hi = mid
        return lo

    @abc.abstractmethod
    def split(self, index, idgen): 
        """Creates a new right sibling with all items from given index on.

        idgen is a callback to create a new identifier for the sibling's leftmost child.
        If this is used, the identifier is returned also (otherwise None).
        """
        pass

    @abc.abstractmethod
    def remove(self, index):
        """Removes the item at the given index and returnes the newly-orphaned
        child id (if any)."""
        pass

    @abc.abstractmethod
    def merge(self, sibling):
        """Merges this node with the given right sibling node and returns the
        newly-orphaned child id (if any)."""
        pass

class OhirbLeaf(OhirbNodeBase):
    """A leaf node in a OHIRB Tree."""

    def __init__(self):
        super().__init__()
        self.values = []

    def get_lvhash(self, index, hasher):
        return hasher(self.labels[index], self.values[index])

    def get_value(self, index):
        return self.values[index]

    def split(self, index, _): 
        """Creates a new right sibling with all items from given index on.

        idgen is a callback to create a new identifier for the sibling's leftmost child.
        If this is used, the identifier is returned also (otherwise None).
        """
        sibling = OhirbLeaf()
        sibling.labels.extend(self.labels[index:])
        sibling.values.extend(self.values[index:])
        del self.labels[index:]
        del self.values[index:]

        return sibling, None

    def insert(self, label, value, hasher):
        """Inserts the given (label, value) pair."""
        lvhash = hasher(label, value)
        index = self.find(label, lvhash, hasher)
        if index < len(self.labels) and self.labels[index] == label and lvhash == self.get_lvhash(index, hasher):
            raise ValueError("Item ({}, {}) already present in leaf node.".format(label, value))
        self.labels.insert(index, label)
        self.values.insert(index, value)

    def range_search(self, lab1, lvh1, lab2, lvh2, hasher):
        """Returns a list of (label, value) pairs for all items in the given range (inclusive)."""
        start = self.find(lab1, lvh1, hasher)
        end = self.find(lab2, lvh2, hasher)
        while end < len(self) and lab2 == self.labels[end] and lvh2 == self.get_lvhash(end, hasher):
            end += 1
        return list(zip(self.labels[start:end], self.values[start:end]))

    def remove(self, index):
        """Removes the item at the given index and returnes the newly-orphaned
        child id (if any)."""
        del self.labels[index]
        del self.values[index]
        return None

    def merge(self, sibling):
        """Merges this node with the given right sibling node and returns the
        newly-orphaned child id (if any)."""
        assert isinstance(sibling, OhirbLeaf)
        assert len(self.labels) == 0 or len(sibling.labels) == 0 or self.labels[-1] <= sibling.labels[0]
        self.labels.extend(sibling.labels)
        self.values.extend(sibling.values)
        assert len(self.labels) == len(self.values)
        return None


class OhirbInternal(OhirbNodeBase):
    """An internal node in a HIRB Tree."""

    def __init__(self, child):
        """Creates a new internal node with one child, as given."""
        super().__init__()
        self.hashes = []
        assert child
        self.children = [child]

    def get_lvhash(self, index, _):
        return self.hashes[index]

    def get_child(self, index):
        return self.children[index]

    def set_child(self, index, c):
        self.children[index] = c

    def split(self, index, idgen):
        """Creates a new right sibling with all items from given index on.

        idgen is a callback to create a new identifier for the sibling's leftmost child.
        If this is used, the identifier is returned also (otherwise None).
        """
        sibling = OhirbInternal(idgen())
        sibling.labels.extend(self.labels[index:])
        sibling.hashes.extend(self.hashes[index:])
        sibling.children.extend(self.children[index+1:])
        del self.labels[index:]
        del self.hashes[index:]
        del self.children[index+1:]

        return sibling, sibling.children[0]

    def insert(self, index, label, lvhash, right_child):
        """Inserts a new (label, hash) pair in this internal node,
        with new right child as given."""
        if index < len(self.labels) and self.labels[index] == label and self.hashes[index] == lvhash:
            raise ValueError("Duplicate item with label {} in internal node".format(label))
        assert index == 0 or (self.labels[index-1], self.hashes[index-1]) < (label, lvhash)
        assert index == len(self) or (label, lvhash) < (self.labels[index], self.hashes[index])
        self.labels.insert(index, label)
        self.hashes.insert(index, lvhash)
        self.children.insert(index+1, right_child)

    def remove(self, index):
        """Removes the item at the given index and returnes the newly-orphaned
        child id (if any)."""
        del self.labels[index]
        del self.hashes[index]
        return self.children.pop(index+1)
    
    def merge(self, sibling):
        """Merges this node with the given right sibling node and returns the
        newly-orphaned child id (if any)."""
        assert isinstance(sibling, OhirbInternal)
        assert (len(self.labels) == 0 or len(sibling.labels) == 0 or 
                (self.labels[-1],self.hashes[-1]) < (sibling.labels[0],sibling.hashes[0]))
        self.labels.extend(sibling.labels)
        self.hashes.extend(sibling.hashes)
        self.children.extend(sibling.children[1:])
        assert len(self.labels) == len(self.hashes)
        assert len(self.children) == len(self.labels) + 1
        return sibling.children[0]
 

def _traverse(oh, curnode, res, indent, verbose):
    if verbose:
        print(" " * indent, end="")
    if isinstance(curnode, OhirbLeaf):
        if verbose: print(list(zip(curnode.labels, curnode.values)))
        res.extend(zip(curnode.labels, curnode.values))
    else:
        if verbose: print(curnode.labels)
        for i in range(len(curnode) + 1):
            cid = curnode.children[i]
            cref = oh.oram.get_ref(cid)
            cnode = cref.get()
            _traverse(oh, cnode, res, indent + 4, verbose)
            cref.set(cnode)
            curnode.children[i] = cref.ident

def traverse(oh, verbose=True):
    res = []
    _traverse(oh, oh._root, res, 0, verbose)
    oh.oram.finalize()
    return res

if __name__ == '__main__':
    # do some tests
    size = 1000
    nsearches = 300

    from store.nopcipher import NopCipher, PyRand
    from hirb.progbar import ProgressBar
    import sys
    import bisect

    seedstr = sys.argv[1] if len(sys.argv) >= 2 else str(random.randrange(1000000))
    seed = bytes(seedstr, 'utf8')
    print("Using seed", seedstr)
    random.seed(seed)

    # create a bunch of random items
    labelgen = lambda : round(random.normalvariate(1000, 10), 1)
    smallval = ''
    bigval = 'Z'*3

    items = set()
    initial = []
    later = []
    dups = set()
    for _ in range(size):
        label = labelgen() # should give some duplicate labels
        value = ''.join(chr(random.randrange(ord('A'), ord('Z')+1)) for _ in range(2))
        item = (label, value)
        if item in items:
            dups.add(item)
            if item not in initial:
                initial.append(item)
            later.append(item)
        elif random.random() < .5:
            initial.append(item)
            items.add(item)
        else:
            later.append(item)
            items.add(item)
    sitems = sorted(items)
    print("{} initial, {} later, {} dups, {} distinct labels".format(
        len(initial), len(later), len(dups), len(set(l for l,v in items))))

    print("Initializing OHIRB...")
    # create OHIRB with initial data
    oh = create_ohirb(
        items_limit = size,
        label_size = 8,
        value_size = 2,
        bucket_size = 2**10,
        cipher = NopCipher,
        randfile = PyRand(seed),
        verbose = True,
        initial_data = initial
    )

    # insert the rest
    for label, value in later:
        if (label, value) in dups:
            try:
                oh.insert(label, value)
                raise RuntimeError("Duplicate insertion should trigger exception")
            except ValueError:
                pass
        else:
            oh.insert(label, value)

    print("Checking contents...")
    # check contents
    contents = traverse(oh, verbose=(size < 50))
    assert len(contents) == len(oh) == len(items)
    assert set(contents) == items
    h1 = oh.height

    # grow height
    oh.grow()
    assert oh.height == h1 + 1

    print("Doing range searches...")
    # do some range searches
    nfull = 0
    nfail = 0
    percents = []
    with ProgressBar(nsearches) as pbar:
        for _ in range(nsearches):
            left = labelgen()
            right = labelgen()
            if left > right:
                left, right = right, left
            lind = bisect.bisect_left(sitems, (left, smallval))
            rind = bisect.bisect_right(sitems, (right, bigval))
            expected = set(sitems[lind:rind])
            limit = random.randrange(rind - lind + 10) + 1
            actual, gotit = oh.range_search(left, right, limit=limit)
            assert bool(expected) == bool(actual)
            assert expected.issuperset(actual)
            assert gotit == (len(expected) == len(actual))
            if len(actual) < min(limit, len(expected)):
                nfail += 1
            elif gotit:
                nfull += 1
            if len(expected) == 0:
                percents.append(1)
            else:
                percents.append(len(actual) / min(limit, len(expected)))
            pbar += 1

    print("Over {} range searches, {} ({:.1f}%) returned the full range.".format(
        nsearches, nfull, 100*nfull/nsearches))
    print("{} searches ({:.1f}%) returned too few results (\"failures\").".format(
        nfail, 100*nfail/nsearches))
    print("On average, each search returned {:.0f}% of the expected size".format(
        100*sum(percents)/nsearches))

    # do some deletions
    for _ in range(size // 10):
        label, value = random.choice(sitems)
        if (label, value) in items:
            removed = oh.remove(label, value)
            assert removed == (label, value)
            items.remove((label, value))
        label = labelgen()
        ind = bisect.bisect_left(sitems, (label, smallval))
        ind2 = bisect.bisect_right(sitems, (label, bigval))
        if ind < len(sitems) and sitems[ind][0] == label and any(item in items for item in sitems[ind:ind2]):
            rl, rv = oh.remove(label)
            assert rl == label and (rl,rv) in items
            items.remove((rl,rv))
        else:
            try:
                oh.remove(label)
                raise RuntimeError("Extra removal should trigger exception")
            except ValueError:
                pass

    # check contents again
    contents = traverse(oh, verbose=False)
    assert len(contents) == len(oh) == len(items)
    assert set(contents) == items

    print("All tests passed!")
