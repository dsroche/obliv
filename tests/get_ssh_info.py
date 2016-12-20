#!/usr/bin/env python3

"""Opportunistically gets/saves info on an SSH connection to use in testing."""

import getpass
import os
import warnings

from obliv import ssh_info

save_file = os.path.join(os.path.dirname(__file__), 'ssh_info.json')
saved_info = False

def load_info():
    """Loads saved SSH info from save_file.
    
    On failure, returns None."""
    global saved_info, save_file, FAIL
    if saved_info is False:
        try:
            saved_info = ssh_info.file_to_ssh_info(save_file)
        except FileNotFoundError:
            saved_info = None
        except ValueError:
            warnings.warn("could not read ssh info. Maybe incorrect passphrase for private key?")
            saved_info = None
        except KeyError:
            warnings.warn("saved ssh info in {} is malformed".format(save_file))
            saved_info = None
    return saved_info

if __name__ == '__main__':
    # if run as a script, prompt user for info and save it
    print(
"""In order to do testing of the SSH connections, we need a server to
connect to using password-less SSH. That is, you need a public/private
key pair to connect. (It's OK if the private key requires a passphrase.)
You will now be prompted for that information.
""")
    saved_info = ssh_info.input_ssh_info()
    if saved_info is not None:
        print("Saving info (without any passphrases) to", save_file)
        ssh_info.ssh_info_to_file(save_file, saved_info)

else:
    # not run as a script, so try reading in when the module is loaded
    load_info()
