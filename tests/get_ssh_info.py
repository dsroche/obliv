#!/usr/bin/env python3

"""Handles inputting, storing, and loading info for a remote SSH connection.

In order to test the SFTP-based backend storage systems, you need
to have some remote server you can connect to.
That has to be different depending on the user and whatever
machines they might have access to.
(Actually I think you can set up a mock SSH server in paramiko,
but I didn't feel like figuring that out.)
"""

import getpass
import json
import os

from obliv import sftp

save_file = os.path.join(os.path.dirname(__file__), 'ssh_info.json')
saved_info = None

def get_info():
    """Opportunistically gets SSHInfo instance.

    It first tries to read it from the save_file, and if that fails
    it reads from the user and saves it to save_file for next time.
    """
    global saved_info
    if saved_info is None:
        try:
            saved_info = file_to_ssh_info(save_file)
        except (FileNotFoundError, ValueError, KeyError) as e:
            if type(e) is not FileNotFoundError:
                print("WARNING: saved info could not be read")
            saved_info = input_ssh_info()
            try:
                ssh_info_to_file(save_file, saved_info)
            except (OSError, TypeError):
                print("WARNING: could not save ssh info to file", save_file)
                pass
    return saved_info

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

    info = sftp.SSHInfo(
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


def input_ssh_info():
    """Prompts the user for the relevant fields"""
    print("Please enter the following info on an SSH server to connect to.")
    print("The default values are in [brackets] if you enter nothing.")

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

        if typ == 'file':
            if not os.path.exists(res):
                raise ValueError("file not found: " + res)
        else:
            res = typ(res)

        d[kname] = res
        del res

    d['needpass'] = (d['keypass'] != '')

    info = dict_to_ssh_info(d, d['keypass'])
    
    del d['keypass']
    return info

# will read in the info when the module is loaded
saved_info = get_info()

if __name__ == '__main__':
    from obliv import sftp
    info = get_info()

    with sftp.SFTP(info):
        print("SSH info successfully saved.")
