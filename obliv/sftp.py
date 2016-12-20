"""Module for a persistent sftp connection"""

import logging

import paramiko

logger = logging.getLogger(__name__)

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


