"""Handles inputting, storing, and loading info for a remote SSH connection."""

import os
import json
import logging
import getpass

from . import sftp

import paramiko

logger = logging.getLogger(__name__)

def input_ssh_info():
    """Prompts the user for the relevant fields and returns an SSHInfo instance.
    
    In case of error or user abort, returns None."""

    print("Please enter the following info on an SSH server to connect to.")
    print("The default values are in [brackets] if you enter nothing.")
    print("Entering a capital Q for any entry quits.")

    data = [
        ('hostname', 'remote server', 'localhost', str),
        ('port', 'remote port', 22, int),
        ('username', 'remote username', getpass.getuser(), str),
        ('fold', 'folder on remote server', 'obliv_store', str),
        ('privkey', 'private key file',
         os.path.expanduser('~/.ssh/id_rsa'), 'file'),
        ('keypass', 'passphrase for private key', '', "password"),
        ('knownhosts', 'known hosts file',
         os.path.expanduser('~/.ssh/known_hosts'), 'file'),
    ]

    d = {}
    for kname, pstr, default, typ in data:
        prompt = kname + ' [' + str(default) + ']: '
        if typ == "password":
            res = getpass.getpass(prompt)
            typ = str
        else:
            res = input(prompt)

        res = res.strip()
        if res == '':
            res = default
        elif res == 'Q':
            del d
            return None

        if typ == 'file':
            if not os.path.exists(res):
                print("ERROR: file not found. Deleting info and aborting.")
                del d
                return None
        else:
            res = typ(res)

        d[kname] = res
        del res

    d['needpass'] = (d['keypass'] != '')

    info = dict_to_ssh_info(d, d['keypass'])
    del d['keypass']

    tryit = input('Try connecting now? [Y]: ').strip()
    if not tryit or tryit.lower().startswith('y'):
        try:
            with sftp.SFTP(info):
                print("Connection successful")
        except:
            print("ERROR: The connection did not go through. Deleting info and aborting.")
            del info
            return None

    return info


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


def file_to_ssh_info(filename):
    """Reads in JSON data from file and returns SSHInfo instance"""

    with open(filename) as fin:
        d = json.load(fin)
        return dict_to_ssh_info(d)


def dict_to_ssh_info(d, pw=None):
    """Converts a dictionary to SSHInfo instance"""

    if pw is None:
        if d['needpass']:
            pw = getpass.getpass(
                prompt="Enter password for " + d['privkey'] + ": ")
        else:
            pw = ''

    info = SSHInfo(
        hostname=d['hostname'],
        username=d['username'],
        privkey=d['privkey'],
        keypass=pw,
        port=d['port'],
        fold=d['fold'],
        knownhosts=d['knownhosts'],
    )

    del pw
    return info



def ssh_info_to_file(filename, info):
    """Writes JSON data for given SSHInfo object to the given file"""

    d = {
        'needpass': info.needpass,
        'hostname': info.hostname,
        'username': info.username,
        'privkey': info.pkfile,
        'port': info.port,
        'fold': info.fold,
        'knownhosts': info.khfile,
    }

    with open(filename, 'w') as fout:
        json.dump(d, fout, indent=4)

