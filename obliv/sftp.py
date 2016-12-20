"""Module for a persistent sftp connection"""

import os
from os import path
from collections import MutableSequence
import sys
import threading
from queue import Queue, Empty
from collections import Counter
import weakref
import logging
import getpass

import paramiko

logger = logging.getLogger(__name__)

class SSHInfo():
    """All the information needed to get an SSH connection to use as ORAM storage."""

    def __init__(self, hostname='localhost', username=getpass.getuser(), 
                 privkey=os.path.expanduser('~/.ssh/id_rsa'), keypass='',
                 port=22, fold='voram_store',
                 knownhosts=os.path.expanduser('~/.ssh/known_hosts')):
        """Create a new SSHInfo object as specified.
        
        hostname: the hostname to connect to (default localhost)
        username: the user to login as (default current username)
        privkey: file containing the RSA private key to login with (default ~/.ssh/id_rsa)
        keypass: password for the private key (default none)
        fold: folder under home directory to store ORAM subdirectories (default voram_store)
        knownhosts: knownhosts file containing host public key signature (default ~/.ssh/known_hosts)
        """
        self.hostname = hostname
        self.username = username
        self.port = port
        self.fold = fold

        self.pkfile = privkey
        self.load_pkey(keypass)

        self.khfile = knownhosts
        self.load_hostkey()

    def load_pkey(self, keypass=""):
        """loads the private key from self.pkfile into self.pkey."""
        self.pkey = None
        gotpass = (keypass != '')

        if not os.path.exists(self.pkfile):
           raise ValueError("SSH key file " + privkey + " could not be opened.")

        while True:
            for cls in paramiko.PKey.__subclasses__():
                try:
                    self.pkey = cls.from_private_key_file(self.pkfile, keypass)
                    break
                except paramiko.ssh_exception.SSHException:
                    pass
                except TypeError:
                    pass

            if self.pkey is not None:
               break
            elif gotpass:
               raise ValueError("SSH key file " + self.pkfile + " could not be read.")

            keypass = getpass.getpass("Passphrase for " + self.pkfile + ": ")
            gotpass = True

        self.needpass = gotpass

    def load_hostkey(self):
        """loads the hostkey from self.khfile into self.hostkey."""
        hostkeys = paramiko.util.load_host_keys(self.khfile)
        if self.hostname in hostkeys:
            kt = hostkeys[self.hostname].keys()[0]
            self.hostkey = hostkeys[self.hostname][kt]
        else:
            raise ValueError("No host key for " + self.hostname + " found in " + self.khfile)

    def __getstate__(self):
        """Used to pickle this object."""
        state = self.__dict__.copy()
        del state['pkey']
        del state['hostkey']
        return state
    
    def __setstate__(self, state):
        """Used to unpickle this object."""
        self.__dict__.update(state)
        self.load_pkey()
        self.load_hostkey()


class SFTP:
    """A persistent sftp connection to the specified host folder.
    
    Can be used as a context manager in with: statements."""

    def __init__(self, ssh_info, subdir=None):
        """Sets up a new SFTP connection (without actually opening it).
        
        ssh_info must be an SSHInfo object."""
        
        self._info = ssh_info
        self._subdir = subdir
        self.trans = None

    def open(self):
        """Starts up the connection"""
        if self.trans:
            logger.warning("sftp session already running; can't open")
            return
        try:
            self.trans = paramiko.Transport((self._info.hostname, self._info.port))
            self.trans.connect()
            self.trans.auth_publickey(self._info.username, self._info.pkey)
            self.sftp = paramiko.SFTPClient.from_transport(self.trans)
        except Exception as e:
            raise RuntimeError("ERROR connecting to {}".format(
                self._info.hostname)) from e

        try:
            self.sftp.chdir(self._info.fold)
        except FileNotFoundError:
            self.sftp.mkdir(self._info.fold)
            self.sftp.chdir(self._info.fold)

        if self._subdir:
            try:
                self.sftp.chdir(self._subdir)
            except FileNotFoundError:
                self.sftp.mkdir(self._subdir)
                self.sftp.chdir(self._subdir)

    def close(self):
        """Shuts down the connection"""
        if self.trans:
            self.trans.close()
            del self.sftp
            del self.trans
            self.trans = None

    def read(self, filename):
        """Reads the given filename and returns the bytes"""
        blockf = self.sftp.open(filename, mode='rb')
        res = blockf.read()
        blockf.close()
        return res

    def write(self, filename, contents):
        """Writes the given contents - which should be bytes - to the given file"""
        blockf = self.sftp.open(filename, mode='wb')
        blockf.write(contents)
        blockf.close()

    def delete(self, filename):
        """Deletes the given filename."""
        self.sftp.remove(filename)

    def __enter__(self):
        """magic method called at the beginning of a with statement"""
        self.open()
        return self

    def __exit__(self, extype, exval, tb):
        """magic method called at the end of a with block"""
        self.close()


