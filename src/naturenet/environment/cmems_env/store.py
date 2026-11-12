"""Zarr helpers: v2-format arrays (readable by zarr 2 and 3), zstd compression, bit rounding."""
import os
import shutil

import numpy as np
import zarr
from numcodecs import Blosc

ZARR3 = int(zarr.__version__.split(".")[0]) >= 3


def compressor(clevel=5):
    return Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE)


def create_array(path, shape, chunks, dtype="f4", fill_value=np.nan, clevel=5, attrs=None, overwrite=True):
    """Create a zarr v2-format array (SIT-FUSE reads these with zarr.load / dask.from_zarr)."""
    if overwrite and os.path.exists(path):
        shutil.rmtree(path)
    kw = dict(shape=shape, chunks=chunks, dtype=dtype, fill_value=fill_value,
              compressor=compressor(clevel), overwrite=overwrite)
    if ZARR3:
        arr = zarr.create(store=path, zarr_format=2, **kw)
    else:
        arr = zarr.create(store=path, **kw)
    if attrs:
        arr.attrs.update(attrs)
    return arr


def open_array(path, mode="r"):
    return zarr.open_array(path, mode=mode) if ZARR3 else zarr.open_array(path, mode=mode)


def save_scene(path, data, attrs=None, clevel=5):
    """One scene = one (C, H, W) float32 array in a single chunk (fast whole-scene reads)."""
    arr = create_array(path, data.shape, data.shape, dtype="f4", fill_value=None, clevel=clevel, attrs=attrs)
    arr[...] = data
    return path


def bitround(a, keepbits):
    """Round float32 mantissas to `keepbits` bits (round-half-even); NaN/inf preserved.

    Lossy but much more compressible; relative error <= 2**-(keepbits+1).
    Never apply to arrays holding a finite sentinel such as -999999.
    """
    if keepbits is None or keepbits >= 23:
        return a
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = a.view(np.uint32)
    drop = 23 - int(keepbits)
    half_m1 = np.uint32((1 << (drop - 1)) - 1)
    mask = np.uint32((0xFFFFFFFF << drop) & 0xFFFFFFFF)
    lsb = (b >> np.uint32(drop)) & np.uint32(1)
    out = (b + half_m1 + lsb) & mask
    special = (b & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    out = np.where(special, b, out)
    return out.view(np.float32)
