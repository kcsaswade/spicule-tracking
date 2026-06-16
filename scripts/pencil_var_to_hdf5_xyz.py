#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pencil as pc


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def discover_var_numbers(datadir: Path, pattern: str = "VAR*") -> List[int]:
    ivars = []
    for p in sorted(datadir.glob(pattern), key=natural_key):
        m = re.fullmatch(r"VAR(\d+)", p.name)
        if m:
            ivars.append(int(m.group(1)))
    return ivars


def parse_var_selection(selection: str, datadir: Path) -> List[int]:
    selection = selection.strip()
    if selection.lower() == "all":
        ivars = discover_var_numbers(datadir)
        if not ivars:
            raise ValueError(f"No VAR files found in {datadir}")
        return ivars

    ivars: List[int] = []
    for chunk in selection.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            start, end = int(a), int(b)
            step = 1 if end >= start else -1
            ivars.extend(list(range(start, end + step, step)))
        else:
            ivars.append(int(chunk))
    if not ivars:
        raise ValueError("Empty VAR selection")
    return ivars


def get_attr_or_item(obj, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    raise AttributeError(f"Missing field: {name}")


def as_1d_axis(arr: np.ndarray, expected: int, name: str) -> np.ndarray:
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Axis {name} is not 1D after squeeze: shape={arr.shape}")
    if arr.size != expected:
        raise ValueError(f"Axis {name} length mismatch: expected {expected}, got {arr.size}")
    return arr.astype(np.float64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Pencil Code VAR snapshots to HDF5 with lnTT ordered as [x, y, z], using a mid-plane y slice and writing temperature[frame, z, x]."
    )
    parser.add_argument("--datadir", required=True, help="Pencil Code data directory")
    parser.add_argument(
        "--vars",
        default="all",
        help='VAR selection: "all", "55,56,60", or range "55-120"',
    )
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Storage dtype for temperature/density datasets",
    )
    parser.add_argument(
        "--include-density",
        action="store_true",
        help="Also convert lnrho to rho and store density[frame,z,x]",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression filter",
    )
    args = parser.parse_args()

    datadir = Path(args.datadir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    ivars = parse_var_selection(args.vars, datadir)
    fields = ["lnTT"] + (["lnrho"] if args.include_density else [])

    first = pc.read.varraw(ivar=ivars[0], var_list=fields, datadir=str(datadir))
    lnTT = np.asarray(get_attr_or_item(first, "lnTT"))
    if lnTT.ndim != 3:
        raise ValueError(f"Expected lnTT to be 3D, got shape {lnTT.shape}")

    nx, ny, nz = lnTT.shape
    y_mid = ny // 2

    x = as_1d_axis(np.asarray(get_attr_or_item(first, "x")), nx, "x")
    z = as_1d_axis(np.asarray(get_attr_or_item(first, "z")), nz, "z")

    compression = None if args.compression == "none" else args.compression
    dtype = np.float32 if args.dtype == "float32" else np.float64
    nframes = len(ivars)
    chunk_frames = min(16, nframes)

    with h5py.File(output, "w") as h5:
        dset_T = h5.create_dataset(
            "temperature",
            shape=(nframes, nz, nx),
            dtype=dtype,
            chunks=(chunk_frames, nz, nx),
            compression=compression,
            shuffle=True if compression else False,
        )
        dset_time = h5.create_dataset("time", shape=(nframes,), dtype=np.float64)
        dset_ivar = h5.create_dataset("ivar", data=np.asarray(ivars, dtype=np.int32))
        dset_x = h5.create_dataset("x", data=x)
        dset_z = h5.create_dataset("z", data=z)

        dset_rho = None
        if args.include_density:
            dset_rho = h5.create_dataset(
                "density",
                shape=(nframes, nz, nx),
                dtype=dtype,
                chunks=(chunk_frames, nz, nx),
                compression=compression,
                shuffle=True if compression else False,
            )

        h5.attrs["source"] = "Pencil Code VAR snapshots"
        h5.attrs["input_field_order"] = "[x, y, z]"
        h5.attrs["temperature_source_field"] = "lnTT"
        h5.attrs["temperature_transform"] = "T = exp(lnTT)"
        h5.attrs["slice_method"] = "mid-plane"
        h5.attrs["y_mid_index"] = y_mid
        h5.attrs["stored_shape"] = "temperature[frame, z, x]"
        h5.attrs["x_dataset"] = "x"
        h5.attrs["z_dataset"] = "z"
        h5.attrs["time_dataset"] = "time"
        h5.attrs["ivar_dataset"] = "ivar"
        h5.attrs["temperature_units"] = "K (assumed if lnTT is ln(T))"
        if args.include_density:
            h5.attrs["density_source_field"] = "lnrho"
            h5.attrs["density_transform"] = "rho = exp(lnrho)"

        for i, ivar in enumerate(ivars):
            var = pc.read.varraw(ivar=ivar, var_list=fields, datadir=str(datadir))
            lnTT_i = np.asarray(get_attr_or_item(var, "lnTT"))
            if lnTT_i.shape != (nx, ny, nz):
                raise ValueError(
                    f"Shape mismatch for VAR{ivar}: expected {(nx, ny, nz)}, got {lnTT_i.shape}"
                )
            T2d = np.exp(lnTT_i[:, y_mid, :], dtype=np.float64).T.astype(dtype, copy=False)
            dset_T[i, :, :] = T2d

            sim_time = float(np.asarray(get_attr_or_item(var, "t")).squeeze())
            dset_time[i] = sim_time

            if dset_rho is not None:
                lnrho_i = np.asarray(get_attr_or_item(var, "lnrho"))
                if lnrho_i.shape != (nx, ny, nz):
                    raise ValueError(
                        f"Shape mismatch for lnrho in VAR{ivar}: expected {(nx, ny, nz)}, got {lnrho_i.shape}"
                    )
                rho2d = np.exp(lnrho_i[:, y_mid, :], dtype=np.float64).T.astype(dtype, copy=False)
                dset_rho[i, :, :] = rho2d

    print(f"Wrote {output}")
    print(f"Frames: {nframes}")
    print(f"temperature shape: ({nframes}, {nz}, {nx})")
    print(f"mid-plane y index: {y_mid}")
    print(f"ivar range: {ivars[0]} .. {ivars[-1]}")


if __name__ == "__main__":
    main()
