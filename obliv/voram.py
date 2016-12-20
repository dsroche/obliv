"""Package for ORAM with variable-size blocks and secure deletion

IMPORTANT: The PyCrypto module is required to make this work.
On Debian/*buntu based systems, you will first need to install
python3-dev and libgmp.

The most important class here is (naturally) Voram, which relies crucially
on an encryption scheme and a storage back-end. The default encryption
is AES and the default storage is a Python list.
"""

import collections
from io import BytesIO
import warnings
import pickle
import zlib
import logging
import sys
import itertools
import math

import Crypto
from Crypto.Cipher import AES
from Crypto import Random

from . import stash as stashmod
from .progbar import ProgressBar
from .idstr import idstr

logger = logging.getLogger(__name__) # used to write log messages for this module

def clog(x,b=2):
    """returns the ceil of log base b x, and never returns negative."""
    if x < b:
        return 1
    else:
        return math.ceil(math.log(x,b))

def estimate_idlen(N):
    """Estimates what id length to use based on N, the number of blobs"""
    # make Pr(id collision) < 2^-40
    return math.ceil((39 + clog(N*(N-1)))/8)

def estimate_nodesize(blob_size, nblobs, idlen, keylen):
    """Estimates what node size to have based on
    blob_size, the size in bytes of a typical blob
    nblobs, the number of blobs per bucket
    idlen, the length of ids in bytes
    keylen, the length of keys in bytes,
    """
    nolen = 2*keylen + nblobs*(idlen + blob_size)
    return nolen +  nblobs * clog(2*nolen-2*keylen-2, 8)

def create_voram(blobs_limit, blob_size,
    info_stream=None, simulate=False,
    # ORAM options
    levels=None, nodesize=None,
    stash=None, storage=None, cipher=None, randfile=None,
    keylen=None, idlen=None, IV=None, immediate=True, verbose=False):
    
    """Creates a new Voram to match the expected usage.

    Parameters are automagically chosen for optimal wonderfulness.
    Only two parameters are required here:

    blobs_limit is the most number of blobs that will be stored in
    the Voram. It's probably OK if you go slightly over this limit.

    blob_size is the expected number of bits for blobs that will be
    stored in this Voram.

    If info_stream is not None, the details of Voram construction
    will be printed to it.

    If simulate is True, no Voram is actually created.

    The next set of options (levels, stash, storage, cipher, randfile,
    keylen, idlen, IV, immediate) default to some optimal-ish values,
    or are passed directly to the Voram constructor if specified.
    """
    if levels is None:
        # make 1 Voram node per blob
        levels = clog(blobs_limit) - 1
    if idlen is None:
        idlen = estimate_idlen(blobs_limit)
    if cipher is None:
        cipher = AES
    if keylen is None:
        keylen = cipher.key_size[-1]
    if nodesize is None:
        nblobs = 4 # according to empirical Voram performance, blobs per bucket
        nodesize = estimate_nodesize(blob_size, nblobs, idlen, keylen)
        if nodesize % cipher.block_size:
            cbs = cipher.block_size
            nodesize += cbs - nodesize%cbs
    if info_stream:
        print("""
Creating Voram with:
    bucket size   {:.1f} KB
    # of levels   {}
    total buckets {}
    total size    {:.1f} MB
    id length     {} bits
    key length    {} bits
    # of blobs    {}
""".format(
            nodesize/2**10,
            levels,
            2**(levels+1) - 1,
            (nodesize*(2**levels-1))/2**20,
            idlen*8,
            keylen*8,
            blobs_limit,
        ), file=info_stream)
    
    if not simulate:        
        return Voram(levels=levels, nodesize=nodesize, storage=storage,
            stash=stash, cipher=cipher, randfile=(randfile if randfile else Random.new()),
            keylen=keylen, idlen=idlen, IV=IV, immediate=immediate, verbose=verbose)

def _onegroup(it, n):
    """Return the next size-n group from the given iterable."""
    toret = []
    for _ in range(n):
        try:
            toret.append(next(it))
        except StopIteration:
            break
    if toret:
        return toret
    else:
        raise StopIteration

def groupsof(L, n):
    """Return an iteration of lists, each of which is a size-n group of
    consecutive elements from L.
    """
    it = iter(L)
    while True:
        yield _onegroup(it, n)

class Voram:
    """Class to manage a Path ORAM scheme with variable-size data blocks.

    The storage is a binary tree of bucket nodes.  Each node is encrypted with
    a key stored in its parent.  These keys are randomly re-chosen on every
    access.  In addition to the two children keys, each bucket node stores
    some (partial) data blocks, each of which has a unique identifier. A prefix
    of the identifier string uniquely determines the path containing that data.
    A local stash is also maintained for temporary and overflow storage.
    Every bucket node has exactly the same size.

    Each bucket node is stored in some external storage structure, indexed
    by consecutive integer ids starting from 0. Data is stored as bytes objects,
    but for convenience the [] operator will store and retrieve any Python 
    object using the pickle module (along with zlib compression).

    To store new data in the Voram, call create() which returns a
    Ref object. You can then use the get(), set(), and destroy()
    methods in the Ref object to access that data in the Voram.

    To improve efficiency, you can disable immediate synchronization
    in the Ref object so that each get() or set() does not necessarily
    cause interaction with the backend storage. In this case, you must
    manually call sync() or finalize() in order to write back changes,
    and you accept that there may be information leakage about the 
    memory access pattern if you are not careful.

    Internally, positions in the storage backend _store are integers from 1
    up to len(_store) = 2*_nleaves - 1. The ids assigned to data objects
    (and therefore visible to the Voram user) are bytes objects, the first L
    bits of which determine the actual position.
    """

    """A dictionary of Vorams currently loaded, stored by IV."""
    instances = {}

    @classmethod 
    def fromIV(cls, IV):
        """Returns the Voram with the given IV, if it's loaded in memory."""
        return cls.instances(IV)

    def __init__(self, levels, nodesize, 
            storage=None, stash=None, cipher=AES, randfile=Random.new(),
            keylen=None, idlen=10, IV=None, immediate=True, verbose=False):
        """Creates a new Voram instance.

        levels is the number of levels in the binary tree, i.e.,
        the length of each path in the Path Voram.

        nodesize is the size in bytes of each Voram node, each of which
        will be stored in one position in the given storage back-end.

        storage is the storage back-end class, which must behave like a Python
        list with element access/update using [] syntax and size increases
        using the extend() system call.

        stash is the data structure to use for stash. This must be a
        dictionary like data structure supporting item assignment and
        retrieval with [] syntax, as well as a special method keys_range
        which returns all the keys in the given range. The default is
        to use a Python dictionary-based implementation.

        The cipher will be used to encrypt blocks, and randfile should
        support the read() method to produce cryptographically-random
        bytes on demand. keylen is the length of keys to use with the given
        encryption cipher.

        idlen is the length (in bytes) of identifiers. It's important to avoid
        collisions, so if N is the maximum number of entries in the Voram, you
        probably want 2^(idlen * 8) >= N^2, i.e., idlen >= 4*log_2(N)

        IV is the initialization vector used for encryption. It is also
        used as an identifier for this Voram instance.
        
        immediate determines whether objects are immediately finalized
        with every read/write operation, or if the finilization is
        delayed (and must therefore be done manually).
        """

        self._L = levels
        self._Z = nodesize
        self._cipher = cipher
        self._rf = randfile
        self._idlen = idlen
        self.immediate_default = immediate

        if keylen is not None:
            self._keylen = keylen
        else:
            # Choose largest available keylength by default
            self._keylen = self._cipher.key_size[-1]
        # The length (in bytes) of data length specifications
        self._lenlen = ((self.Z - 2*self._keylen - 2).bit_length() + 7) // 8

        
        if self.Z > 2 * (self.node_data - self.chunk_header):
            raise ValueError("""
More than half your node length will be used by metadata.
Increase the nodesize parameter for this Voram.
nodesize={}, keys take up {}, leaving {} for data.
Chunk headers take up {} = {} for id and {} for length.
                """.strip().format(
                    self._Z,
                    2*self._keylen,
                    self.node_data,
                    self.chunk_header,
                    self._idlen,
                    self._lenlen
                ))

        if self.Z % self._cipher.block_size:
            cbs = self._cipher.block_size
            self._Z += cbs - self.Z%cbs
            warnings.warn("nodesize {} is not a multiple of cipher block size {}.\n" +
                "Adjusting nodesize to {}.".format(nodesize, cbs, self.Z), stacklevel = 2)

        # The initialization vector is static for the lifetime of this Voram.
        self._registerIV(
            self._rf.read(self._cipher.block_size)
            if IV is None else IV)

        # The stash is a map from position to byte array.
        # NOTE: the byte array may be partial!
        # The cache holds data that has been read but not yet written back.
        # All positions that are to be written back are stored in a buffer.
        # The keys of children of all cache nodes are stored until written
        self._stash = stashmod.DictStash() if stash is None else stash
        self._cache = {}
        self._write_buffer = set()
        self._node_keys = {}

        # Initialize the storage with empty nodes
        self.store = [] if storage is None else storage

        nleaves = 1 << (self.L)

        if verbose:
            with ProgressBar(2*nleaves-1) as pbar:
                for count in range(2*nleaves - 1):
                    self._store.append(bytes(self._Z))
                    pbar += 1
        else:
            self._store.extend(bytes(self._Z) for i in range(2*nleaves - 1))

        for pos in reversed(range(nleaves, 2*nleaves)):
            self._writenode(pos, True)
            ppos = pos
            # Some bit magic here to initialize internal nodes using no
            # more than O(L) key storage
            while ppos > 0 and ppos % 2 == 0:
                ppos //= 2
                self._writenode(ppos, True)

        # this keeps track of how many nodes are accessed at a time.
        # used for debugging or to check on the obliviousness.
        self.counts = collections.Counter()

    @property
    def L(self):
        """The number of levels in the Path Voram"""
        return self._L

    @property
    def Z(self):
        """The size of each Voram node, in bytes"""
        return self._Z

    @property
    def IV(self):
        """The initialization vector used with the cipher"""
        return self._IV

    @property
    def store(self):
        """The underlying storage of Voram nodes, perhaps externally"""
        return self._store

    @store.setter
    def store(self, newstore):
        """Change the oram storage device"""
        self._store = newstore
        self._can_prefetch = 'prefetch' in dir(self._store)

    @property
    def idlen(self):
        """The length of identifiers, in bytes"""
        return self._idlen

    @property
    def node_data(self):
        """The amount of data that can be stored in a node, in bytes"""
        return self.Z - 2*self._keylen

    @property
    def chunk_header(self):
        """The length of metadata for a single chunk, in bytes"""
        return self.idlen + self._lenlen

    @property
    def randfile(self):
        return self._rf

    def stash_size(self):
        """Returns the total size of all DATA stored in cache and stash."""
        return sum(len(data) for data in self._stash.values())

    def grow(self):
        """Doubles the storage size by adding one new level."""
        cur_buffer = set(self._write_buffer)
        self._L += 1
        nleaves = 1 << self.L
        self._store.extend(bytes(self._Z) for i in range(nleaves))
        toread = [2**i for i in reversed(range(self.L))]
        for pos in range(nleaves, 2*nleaves):

            self._prefetch(x-1 for x in toread)
            for ppos in reversed(toread):
                self._readnode(ppos)
            toread.clear()
            self._writenode(pos, True)
            ppos = pos + 1
            while ppos > 2 and ppos%2 == 0:
                ppos //= 2
                if (ppos-1) not in cur_buffer:
                    self._writenode(ppos-1)
                toread.append(ppos)
        assert cur_buffer == self._write_buffer

    def new_ident(self):
        """Generates a new random ident for this Voram.

        Normally, you should NOT call this. Instead call create() which will
        create a new Ref instance (which in turn calls this method to generate
        its own ident).
        """
        ident = bytearray(self._rf.read(self._idlen))
        # leading bit of ident must be 1 to distinguish from empty space in nodes.
        ident[0] |= 0x80
        return bytes(ident)

    def create(self, read=True, sync=None, ident=None):
        """Creates a new blob in this Voram and returns a Ref for it.
        
        read indicates whether a random pos should first be read, so that
        this creation looks like any other read/write operation.
        
        sync indicates whether the returned Ref instance will automatically
        sync immediately on every access. The default sync behavior is a
        parameter of the oram itself.

        ident is the ident of the Ref that gets returned. None (the default)
        means generate a new, random ident.
        """

        ref = Ref(self, 
                  syncImmediate=(self.immediate_default if sync is None else sync),
                  nextref = ident)
        if read:
            self._prefetch(x-1 for x in ref.path())
            
            for pos in ref.path():
                self._readnode(pos)
            ref._regen()
        self._cache[ref.ident] = bytes()
        return ref

    def dummy_op(self, sync=None):
        """Performs a do-nothing operation, which appears like any other access.
        
        Just like create(), except nothing is actually added to the cache."""
        rref = Ref(self)
        self._prefetch(x-1 for x in rref.path())
        for pos in rref.path():
            self._readnode(pos)

        if sync or (sync is None and self.immediate_default):
            self.finalize()

    def get_ref(self, ident, sync=None):
        """Gets a Ref object for the given identifier"""
        return Ref(self, ident=ident, 
                   syncImmediate=(self.immediate_default if sync is None else sync))

    def _prefetch(self, indices):
        """Pre-fetches the specified buckets from storage, to the extent
        possible."""
        if self._can_prefetch:
            self.store.prefetch(itertools.islice(indices, self.store.bufsize))

    def fetch(self, ref):
        """Reads a single item from the Voram and updates its reference."""
        if ref.ident in self._cache:
            # do nothing if already fetched
            return

        self._prefetch(x-1 for x in ref.path())
                       
        for pos in ref.path():
            self._readnode(pos)

        if ref.ident in self._stash:
            data = bytes(self._stash[ref.ident])
            del self._stash[ref.ident]
        else:
            raise KeyError("identifier {} not found in Voram".format(ref))

        ref._regen()
        self._cache[ref.ident] = data

    def finalize(self):
        """Writes back all paths that have been read and not yet written."""

        for ident, data in self._cache.items():
            self._stash[ident] = bytearray(data)
        self._cache.clear()

        if self._write_buffer:
            # insert everything from cache (complete items being worked on)
            # into stash (temorarily saved between writes, possibly incomplete)
            num_fetched = len(self._write_buffer)
            self._prefetch(x-1 for x in self._write_buffer)
                    
            for pos in reversed(sorted(self._write_buffer)):
                self._writenode(pos)

            # update counts
            self.counts[num_fetched] += 1

        # At this point, cache is empty and only root node's key is stored.
        assert (len(self._cache) == 0
                and len(self._write_buffer) == 0 
                and 1 in self._node_keys 
                and len(self._node_keys) == 1)

    def get(self, ident):
        """Gets the raw bytes that have been fetched for an identifier."""
        return self._cache[ident]

    def set(self, ident, data):
        """Sets the raw bytes for an identifier"""
        assert type(data) is bytes
        if ident not in self._cache:
            raise KeyError("identifier " + str(ident) + 
                " has not yet been fetched!")
        self._cache[ident] = data

    def remove(self, ident):
        """Removes the named ID from the Voram; still needs finalization."""
        del self._cache[ident]

    def __getitem__(self, ref):
        """Returns the data object associated with a reference."""
        if isinstance(ref, Ref) and ref._oram == self:
            return ref.get()
        else:
            raise KeyError("Invalid reference for this Voram: {}".format(ref))

    def __setitem__(self, ref, value):
        """Assigns the given object to the given reference."""
        if isinstance(ref, Ref) and ref._oram == self:
            ref.set(value)
        else:
            raise KeyError("Invalid reference for this Voram: {}".format(ref))

    def __delitem__(self, ref):
        """Removes the referenced data from the Voram"""
        if isinstance(ref, Ref) and ref._oram == self:
            ref.destroy()
        else:
            raise KeyError("Invalid reference for this Voram: {}".format(ref))

    def _registerIV(self, newiv):
        """Changes the IV to the given one and (re)assigns class instances."""
        if hasattr(self, 'IV') and self.IV in self.instances:
            del self.instances[self.IV]
        self._IV = newiv
        self.instances[self.IV] = self

    def _stash_eligible(self, pos):
        """Iterates over positions that could be written to pos from stash."""
        level = pos.bit_length()
        shift = self._idlen * 8 - level
        startid = (pos << shift).to_bytes(self._idlen, byteorder='big')

        assert self._idlen == len(startid)

        if (pos+1).bit_length() == level:
            endid = ((pos+1) << shift).to_bytes(self._idlen, byteorder='big')
        else:
            endid = b'\xff' * (self._idlen + 1)
        
        return self._stash.keys_range(startid, endid)

    def _rand_enc_key(self):
        """Chooses a new random encryption key"""
        return self._rf.read(self._keylen)

    def _writenode(self, pos, override=False):
        """Writes back a single Voram node to the backend storage.

        Automatically loads data from the stash and keys from _node_keys.
        The storage format of a node is
        enc[ child_key_0 | child_key_1 | (ident | length | data)* ]
        where an ident of 0 indicates the rest of the data is dummy/filler/padding.

        override indicates that the error for writing an unread node
        should be supressed.
        """
        if not override and pos not in self._write_buffer:
            raise ValueError("Can't write node " + str(pos) +
                " that hasn't been read yet.")

        nodeout = BytesIO()
        remaining = self.Z
        
        # Write childrens' decryption keys
        for childind in (pos*2, pos*2+1):
            if childind in self._node_keys:
                nodeout.write(self._node_keys[childind])
                del self._node_keys[childind]
            else:
                # make up a random child key; should only happen for leaves.
                assert pos.bit_length() == self.L+1
                nodeout.write(self._rand_enc_key())
            remaining -= self._keylen

        # Choose data blocks from stash and write their contents
        to_delete = []
        for ident in self._stash_eligible(pos):
            if remaining <= self._idlen + self._lenlen:
                # no more room
                break
            data = self._stash[ident]
            nodeout.write(ident)
            remaining -= self._idlen
            # write as much as possible
            writelen = min(remaining - self._lenlen, len(data))
            nodeout.write(writelen.to_bytes(self._lenlen, byteorder='big'))
            remaining -= self._lenlen
            nodeout.write(data[-writelen:])
            remaining -= writelen
            if len(data) == writelen:
                # remove from stash if all data was written
                to_delete.append(ident)
            else:
                # if only partial data was written, remove that part from stash
                del data[-writelen:]

        for ident in to_delete:
            del self._stash[ident]
       
        # Pad with 0 bytes
        if remaining > 0:
            nodeout.write(b'\0' * remaining)
        else:
            assert remaining == 0

        # Generate a random key and encrypt this node
        self._node_keys[pos] = self._rand_enc_key()
        crypt = self._cipher.new(self._node_keys[pos],
            mode=self._cipher.MODE_CBC,
            IV=self.IV)
        tostore = crypt.encrypt(nodeout.getvalue())
        assert len(tostore) == self.Z

        self._store[pos-1] = tostore
        self._write_buffer.discard(pos)

    def _readnode(self, pos):
        """Reads a single Voram node from the backend storage to the stash.

        Saves data into the stash and updates _node_keys.
        The parent of pos must already be stored in the cache.
        """
        if pos in self._write_buffer:
            # already in the stash; no need to re-read
            return

        if pos not in self._node_keys:
            raise ValueError("No decryption key for position " + str(pos) +
                ".\nDid you forget to read the parent node first?")
        crypt = self._cipher.new(self._node_keys[pos],
            mode=self._cipher.MODE_CBC,
            IV=self.IV)
        rawnode = crypt.decrypt(self._store[pos-1])
        nodein = BytesIO(rawnode)

        # Read and store children's decryption keys
        self._node_keys[2*pos] = nodein.read(self._keylen)
        self._node_keys[2*pos + 1] = nodein.read(self._keylen)

        # Read in data blocks and add to stash
        while True:
            ident = nodein.read(self.idlen)
            if len(ident) < self.idlen or not any(ident):
                break
            self._stash.setdefault(ident, bytearray())
            blen = int.from_bytes(nodein.read(self._lenlen), byteorder='big')
            self._stash[ident].extend(nodein.read(blen))

        self._write_buffer.add(pos)

    def __repr__(self):
        return 'oram.Voram.fromIV({})'.format(repr(self.IV))

    def __getstate__(self):
        """Used by pickle.dump() serialization"""
        state = self.__dict__.copy()
        del state['_cipher']
        del state['_rf']
        state['__cipher_new'] = self._cipher.new
        state['__cipher_MODE_CBC'] = self._cipher.MODE_CBC
        return state

    def __setstate__(self, state):
        """Used by pickle.load() unserialization"""
        self.__dict__.update(state)
        LoadedCipher = collections.namedtuple('LoadedCipher', 'new MODE_CBC')
        self._cipher = LoadedCipher(
                state['__cipher_new'], state['__cipher_MODE_CBC'])
        self._rf = Random.new()
        self.store = self._store

    def debug_levels(self, max_l=None):
        if max_l is None: max_l = self._L

        levels = {}
        dks = {0: self._node_keys[1]}
        for i, data in enumerate(self._store):
            rawdata = self._cipher.new(dks[i],
                                       mode=self._cipher.MODE_CBC,
                                       IV=self.IV).decrypt(self._store[i])
            dks[2*i+1] = rawdata[0:self._keylen]
            dks[2*i+2] = rawdata[self._keylen:2*self._keylen]
            j = 2*self._keylen
            first = True
            
            l=math.floor(math.log(i+1,2))
            if l > max_l : return levels

            while len(rawdata) - j >= self._idlen:
                ident = rawdata[j:j+self._idlen]
                j += self._idlen
                if not any(ident): break
                size = int.from_bytes(rawdata[j:j+self._lenlen], byteorder='big')
                j += self._lenlen + size

                levels.setdefault(l,0)
                levels[l]+=size

        return levels

        # levs = list(levels.keys())
        # levs.sort()
        # for l in levs:
        #     space=2**(l)*self.Z
        #     total_space=(2**(l+1)-1)*self.Z
        #     print("Level {}: {}/{} = {}".format(l,levels[l],space, levels[l]/space))
        # print("Total: {}/{} = {} ".format(sum(levels.values()), total_space, sum(levels.values())/total_space))

    def debug(self, paths=False):
        print("----------------- Voram --------------------")
        print("IV:", self.IV)
        print(self._L, "levels,", self._Z, "node size,", self._idlen, "id length")
        print(self._keylen, "key length,", self._lenlen, "length length")
        posns = set()
        print("CACHE:")
        for k,v in self._cache.items():
            print("   ", idstr(k), len(v), "bytes")
            posns.add(k)
        print("STASH:")
        for k,v in self._stash.items():
            print("   ", idstr(k), len(v), "bytes")
            posns.add(k)
        dks = {0: self._node_keys[1]}
        for i, data in enumerate(self._store):
            rawdata = self._cipher.new(dks[i],mode=self._cipher.MODE_CBC,IV=self.IV).decrypt(self._store[i])
            dks[2*i+1] = rawdata[0:self._keylen]
            dks[2*i+2] = rawdata[self._keylen:2*self._keylen]
            j = 2*self._keylen
            first = True
            while len(rawdata) - j >= self._idlen:
                ident = rawdata[j:j+self._idlen]
                j += self._idlen
                if not any(ident): break
                if first:
                    print("STORE POSN {}:".format(i+1))
                    first = False
                posns.add(ident)
                size = int.from_bytes(rawdata[j:j+self._lenlen], byteorder='big')
                j += self._lenlen + size
                print("    {} - {} bytes".format(idstr(ident), size))
        if paths:
            print("POSN PATHS:")
            tref = Ref(self, b'', False)
            for ident in posns:
                tref._id = bytes(ident)
                print("    {} - {}".format(idstr(ident), tuple(tref.path())))
        print("...........................................")
        print()


class Ref:
    """A reference to a single Voram element
    
    The stored position tag is modified on a fetch() call.
    """

    def __init__(self, oramInstance, ident=None, syncImmediate=False, nextref=None):
        """Generates a new identifier at random.

        Users should not use this.  Call Voram.create() instead to create a
        new object in the Voram and get the reference.
        """
        self._oram = oramInstance
        self._immediate = bool(syncImmediate)
        self._nextref = nextref
        if ident is None:
            self._id = self._oram.new_ident()
        else:
            self._id = bytes(ident)

    @property
    def ident(self):
        """The internal id tag"""
        return self._id

    @property
    def immediate(self):
        """Indicates whether every get() and set() operation immediately syncs.

        If this is false, the user must call sync() or Voram.finalize() to write
        the data back.
        """
        return self._immediate

    @immediate.setter
    def immediate(self, value):
        self._immediate = bool(value)

    def get(self):
        """Returns the data object associated with this reference

        This version uses pickling and compression to return any Python
        object. Use getRaw() to just get the bytes.
        """
        return pickle.loads(zlib.decompress(self.getRaw()))

    def getRaw(self):
        """Like get(), but returns the raw bytes instead."""
        self._oram.fetch(self)
        data = self._oram.get(self.ident)
        if self.immediate:
            self._oram.finalize()
        return data

    def set(self, value):
        """Assigns the given object to the fetched identifier."""
        self.setRaw(zlib.compress(pickle.dumps(value, protocol=3)))

    def setRaw(self, value):
        """Like set(), but sets the raw bytes instead."""
        self._oram.fetch(self)
        self._oram.set(self.ident, value)
        if self.immediate:
            self._oram.finalize()

    def destroy(self):
        """Removes the data from the Voram"""
        self._oram.fetch(self)
        self._oram.remove(self.ident)
        if self.immediate:
            self._oram.finalize()
        self._id = None

    def sync(self):
        """Calls finalize on the containing Voram."""
        if self.immediate:
            warn('sync call is unnecessary on an immediate reference',
                stacklevel=2)
        self._oram.finalize()

    def path(self):
        """Iterates over pos's for the given identifier's path."""
        idint = int.from_bytes(self._id[:(self._oram.L + 8)//8], byteorder='big')
        shift = idint.bit_length() - 1
        for _ in range(self._oram.L+1):
            yield idint >> shift
            shift -= 1

    def set_nextref(self, nextref):
        """Assigns the NEXT ident for this ref.

        See pregen() below.
        """
        if self._nextref is None:
            self._nextref = nextref
        elif self._nextref != nextref:
            raise AttributeError("nextref has already been generated and doesn't match")

    def pregen(self):
        """Pre-generates the NEXT ident for this ref.

        A random ident is generated and returned. The next time _regen() is called,
        that ident is set instead of a newly-created one. This can be useful if
        there is a need to store (finalize) an ident for an as-yet-unchanged
        Voram blob.
        """
        if self._nextref is None:
            self._nextref = self._oram.new_ident()
        return self._nextref

    def _regen(self):
        """Chooses a new random ident."""
        self._id = self.pregen()
        self._nextref = None

    def __repr__(self):
        params = [repr(self._oram), repr(self.ident)]
        if self.immediate:
            params.append(repr(self.immediate))
        return 'oram.Ref({})'.format(', '.join(params))

    def __str__(self):
        return idstr(self.ident)

