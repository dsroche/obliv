==================================================================
  obliv: Variable-Sized Block Oblivious RAM with HIRB dictionary
==================================================================

.. contents::

Overview
========

This is a Python package that supports oblivious storage of data.
The term "oblivious" here means that anyone who sees the 
*access pattern* (the sequence of read/write requests to the backend
server) will provably not learn anything about what data is being
read or written, only some idea of *how much* total data is being
stored and accessed.

In the current implementation, the "backend storage" could be either
a remote server using ``sftp`` or a subdirectory in the filesystem,
and the data being stored could either be variable-sized blocks
with changing pointers (using ``Voram``) or arbirary key/value
storage (using the ``Hirb`` data structure within ``Voram``).

This is an academic project connected with our research paper
from IEEE S&P 2016 (arXiv_, `doi:10.1109/SP.2016.19`__).

.. _arXiv: https://arxiv.org/abs/1505.07391
__ https://doi.org/10.1109/SP.2016.19

While we hope we've done a good job and this software is useful for
someone, note it is **not commercial-grade** and shouldn't be used
for anything important.

Installing
==========

The ``obliv`` package is based on Python3, and requires
version 3.4 or later. You will also need ``pycrypto`` and ``paramiko``,
which are both freely available through PyPI and pip_.
(Actually, the paramiko library is only needed if you want to run ORAMs over
SSH and is not required for the core functionality.)

.. _pip: https://pip.pypa.io/en/latest/installing/

There is a setup.py script that should install the ``obliv`` package,
along with the dependencies as required, if you run:

    pip3 install .

from this directory. Note you can also pass the ``-e`` flag to ``pip3``
if you want to be in "edit mode" so you can edit package files as needed.

If you are hesitant to introduce the ``obliv`` package globally, you
can use Python's virtualenv_ system to make a mini-container for everything
it needs. For example::

    pip3 install virtualenv wheel   # install virtualenv
    virtualenv -p python3 venv      # put package stuff in venv folder
    source venv/bin/activate        # go into the Python virtualenv
    pip3 install -e .               # install obliv package in the virtualenv
    (here you can use the obliv package freely)
    deactivate                      # get out of the virtualenv

Or of course you could install ``pycrypto`` and ``paramiko`` in any
other way, add this directory to your Python path, and just go from there.

.. _virtualenv: https://virtualenv.pypa.io/en/stable/

Test Programs
=============

There are some unit tests in the ``tests/`` directory. To run them all,
just do::

    python3 -m unittest -v

from the top-level directory, which will automagically find the test cases 
and run them. To run just one test case, such as ``test_fstore``, do::

    python3 -m tests.test_fstore -v

SSH Connection
==============

The ``mt_ssh_store`` class supports storage in files on a remote machine,
with communication over persistent SSH connections. In order to support
this capability, you must have the ``paramiko`` package installed (see above),
and you must have public/private key authentication to the server in
question. 

(The ``ssh-keygen`` program is a common command-line tool to create
such a public-private key pair, and the ``ssh-copy-id`` utility can be
used to copy your generated public key to the remote server.)

The ``ssh_info.SSHInfo`` class is used to store information about connecting
to a remote server. See the documentation in that class for details on how
to initialize an ``SSHInfo`` or save (and later load) this information in
JSON format.

There are some tests for the SSH connections that also need some connection
information in order to test the relevant parts of the package. These will
be skipped by default; to enter the server and key information that
will enable them, run::

    python3 -m tests.get_ssh_info

Package Structure
=================

Storage
-------

Oblivious storage is based on a front-end access data structure
(outlined below), and a back-end storage. What "oblivious" means,
roughly, is that anyone who can see everything going on with the
backend storage, will have no idea what is being stored or where.

In practice, the backend storage is a fixed-size list of fixed
size blocks, all of which are encrypted. Internally, these blocks
will represent nodes in a binary tree-based vORAM.

This package provides two classes for backend storage.
``fstore`` can be used to store blocks in a given folder
in the local filesystem, and ``mt_ssh_store`` is used
to store blocks on a remote host using the SSH protocol.

vORAM
-----

The heart of the package is the ``voram.Voram`` class, which represents
an ORAM with variable-size blocks. To create a Voram instance,
use the ``voram.create_voram`` method, which will automatically
choose many settings appropriately.

One of the settings there is the backend storage, which defaults
to a simple Python list, but can be specified to be an instance
of the ``fstore`` or ``mt_ssh_store`` classes as well, as outlined
above.

Note that vORAM is a *non-recursive* ORAM without any position map.
This means that every time an element is accessed, its position
changes in the ORAM. To facilitate usage, the ``create()`` method
of ``voram.Voram`` returns a ``Ref`` object which encapsualates
a single object's (changing) identifier and position in the ORAM,
and handles operations on it. For example::
  
    >>> from obliv.voram import create_voram

    >>> v = create_voram(blobs_limit=10, blob_size=5, nodesize=256)

    >>> r1 = v.create()

    >>> r1.set("value of r1")
    >>> print(r1)
    ident:aceb69cc7e0f

    >>> r1.get()
    'value of r1'
    >>> print(r1)
    ident:e819484f8b4c

    >>> r2 = v.create()
    >>> r2.set('value of r2')
    >>> r1.set('new value of r1')

    >>> r1.get(), r2.get()
    ('new value of r1', 'value of r2')
    >>> print(r1, r2)
    ident:d48a0e996fce ident:b9295f53671e

    >>> r1.get(), r2.get()
    ('new value of r1', 'value of r2')
    >>> print(r1, r2)
    ident:e4dc613ddb2f ident:c8c630544f94

HIRB
----

The direct utility of vORAM is limited by the lack of a position map,
because the (changing) references to every object must somehow be stored or
else that object becomes inaccessible.

The more useful class is ``hirb.Hirb``, which is an oblivious
map data structure built on top of vORAM. Because HIRB is an oblivious
data structure, it takes care of storing all the vORAM positions
within the data structure itself, and only uses O(1) storage for
the root node.

The parameters of a HIRB data structure and the underlying vORAM
are closely connected, so there is a convenience method
``hirb.create_hirb`` to choose most of these for you and create
a HIRB as well as a vORAM to store the HIRB. Again, an underlying
storage object such as ``fstore`` can be specified.

Using a HIRB instance is exactly like using a normal Python
dictionary, except that every lookup, assignment, or deletion
corresponds to the same number of vORAM operations to maintain
obliviousness. For example::

    >>> from obliv.hirb import create_hirb

    >>> h = create_hirb(items_limit=20, value_size=5, bucket_size=512)

    >>> h['k1'] = 'value1'
    >>> print(h['k1'])
    value1

    >>> h['k2'] = 'value2'
    >>> print(h['k2'])
    value2

    >>> len(h)
    2

    >>> h['k1'] = 'new_value1'
    >>> print(h['k1'], h['k2'])
    new_value1 value2

    >>> del h['k1']
    >>> h['k1']
    KeyError: 'k1 is not in the HIRB.'

    >>> print(h['k2'])
    value2

Authors
=======

The research paper introducing the vORAM and HIRB is co-authored by
Daniel S. Roche, Adam Aviv, and Seung Geol Choi at the
U.S. Naval Academy.

This source code was written by Daniel S. Roche and Adam Aviv.

License
=======

The source code is released into the public domain under the
Unlicense_. The original authors are U.S. Government employees
and may not claim copyright. We hope you will use this software
and find some value from it, but we can't make any guarantees!

.. _Unlicense: http://unlicense.org/

Contributing
============

It would be great if you want to make this software better. Just
submit a pull request or send an email to
``roche@usna.edu``.
