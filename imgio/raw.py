"""
Utility functions for decoding sensor raw files.
"""

import numpy as np


######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################


def decode(raw: np.ndarray, bpp: int, packing: str) -> np.ndarray:
    if bpp == 10:
        if packing == "mipi":
            frame = decode_mipi10(raw)
        if packing == "plain":
            frame = decode_plain10(raw)
        if packing == "unpacked":
            frame = raw.view(np.uint16)
    elif bpp == 12:
        if packing == "mipi":
            frame = decode_mipi12(raw)
        if packing == "plain":
            frame = decode_plain12(raw)
        if packing == "unpacked":
            frame = raw.view(np.uint16)
    return frame


def write(filespec, image, _maxval, _pack=False, _verbose=False):
    # Warning: hardcoded endianness (x86), unpacked raw only
    with open(filespec, "wb") as outfile:
        image = image.copy(order="C")  # ensure x86 byte order
        outfile.write(image)


def decode_plain10(raw):
    # LSB-first contiguous packing of 4 x 10-bit pixels into 5 bytes:
    #   byte0: a7 a6 a5 a4 a3 a2 a1 a0
    #   byte1: b5 b4 b3 b2 b1 b0 a9 a8
    #   byte2: c3 c2 c1 c0 b9 b8 b7 b6
    #   byte3: d1 d0 c9 c8 c7 c6 c5 c4
    #   byte4: d9 d8 d7 d6 d5 d4 d3 d2
    nbytes = raw.size - raw.size % 5
    b = raw[:nbytes].reshape(-1, 5).astype(np.uint16)
    p = np.empty((len(b), 4), dtype=np.uint16)
    p[:, 0] = b[:, 0] | (b[:, 1] & 0x03) << 8
    p[:, 1] = b[:, 1] >> 2 | (b[:, 2] & 0x0F) << 6
    p[:, 2] = b[:, 2] >> 4 | (b[:, 3] & 0x3F) << 4
    p[:, 3] = b[:, 3] >> 6 | b[:, 4] << 2
    return p.ravel()


def decode_plain12(raw):
    # LSB-first contiguous packing of 2 x 12-bit pixels into 3 bytes:
    #   byte0: a7 a6 a5 a4 a3 a2 a1 a0
    #   byte1: b3 b2 b1 b0 aB aA a9 a8
    #   byte2: cB cA c9 c8 b7 b6 b5 b4
    nbytes = raw.size - raw.size % 3
    b = raw[:nbytes].reshape(-1, 3).astype(np.uint16)
    p = np.empty((len(b), 2), dtype=np.uint16)
    p[:, 0] = b[:, 0] | (b[:, 1] & 0x0F) << 8
    p[:, 1] = b[:, 1] >> 4 | b[:, 2] << 4
    return p.ravel()


def decode_mipi10(raw):
    # MSB-first disjoint packing of 4 x 10-bit pixels into 5 bytes:
    #   byte0: a9 a8 a7 a6 a5 a4 a3 a2
    #   byte1: b9 b8 b7 b6 b5 b4 b3 b2
    #   byte2: c9 c8 c7 c6 c5 c4 c3 c2
    #   byte3: d9 d8 d7 d6 d5 d4 d3 d2
    #   byte4: d1 d0 c1 c0 b1 b0 a1 a0
    nbytes = raw.size - raw.size % 5
    b = raw[:nbytes].reshape(-1, 5).astype(np.uint16)
    p = np.empty((len(b), 4), dtype=np.uint16)
    p[:, 0] = (b[:, 0] << 2) | ((b[:, 4] >> 0) & 3)
    p[:, 1] = (b[:, 1] << 2) | ((b[:, 4] >> 2) & 3)
    p[:, 2] = (b[:, 2] << 2) | ((b[:, 4] >> 4) & 3)
    p[:, 3] = (b[:, 3] << 2) | ((b[:, 4] >> 6) & 3)
    return p.ravel()


def decode_mipi12(raw):
    # MSB-first disjoint packing of 2 x 12-bit pixels into 3 bytes:
    #   byte0: aB aA a9 a8 a7 a6 a5 a4
    #   byte1: bB bB b9 b8 b7 b6 b5 b4
    #   byte2: b3 b2 b1 b0 a3 a2 a1 a0
    nbytes = raw.size - raw.size % 3
    b = raw[:nbytes].reshape(-1, 3)
    p = np.empty((len(b), 2), dtype=np.uint16)
    p[:, 0] = (b[:, 0] << 4) | ((b[:, 2] >> 0) & 0xF)
    p[:, 1] = (b[:, 1] << 4) | ((b[:, 2] >> 4) & 0xF)
    return p.ravel()
