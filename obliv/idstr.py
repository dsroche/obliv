"""Tiny module to pretty-print an ident string."""

# has to be its own module to avoid circular imports

def idstr(ident):
    """Returns a string for the given identifier."""
    return 'ident:' + hex(int.from_bytes(ident,'big')).lstrip('0x')

