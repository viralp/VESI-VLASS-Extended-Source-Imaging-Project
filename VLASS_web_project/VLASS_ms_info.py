#!/usr/bin/env python3
"""
Description:
  Given an object name and target RA/Dec (decimal degrees),
  find relevant VLASS quicklook tiles, extract measurement sets
  from casa_pipescript.py, and optionally download quicklook FITS/logs.

Notes:
  - Requires Python 3.x
  - Depends on: requests, numpy, pandas, astropy
  - Script will fail if NRAO quicklook archive is down.
"""

import argparse
import csv
import glob
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u


# -----------------------------
# Helpers
# -----------------------------

JNAME_REGEX = r"J\d{6}[+|-]\d{6}\.\d\d\.\d{4}\S{3}"
MS_REGEX = r"(VLASS[^'\"]+?)\.ms"


def unique_list(seq: List[str]) -> List[str]:
    """Return unique items preserving order."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def check_url_exists(url: str, timeout: int = 8) -> bool:
    """HEAD-check a URL."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return (r.status_code // 100) == 2
    except requests.RequestException:
        return False


def download_file(url: str, outpath: Path, overwrite: bool = False, timeout: int = 30) -> bool:
    """Download URL to outpath. Returns True if downloaded."""
    if outpath.exists() and not overwrite:
        return True

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"[download] failed: {url}\n  -> {e}")
        return False


# -----------------------------
# Tile boundary / selection
# -----------------------------

def read_tile_boundaries(csv_path: Path) -> pd.DataFrame:
    """
    Read Tile_Boundaries.csv into a DataFrame.

    Expected columns by index (per your original file):
      0 tile_id
      1 dec_start
      2 dec_end
      3 ra_start_hours
      4 ra_end_hours
      5 vlass_id
    """
    rows = []
    with open(csv_path, newline="") as f:
        data = list(csv.reader(f))

    for tile in data:
        try:
            tile_id = tile[0]
            dec_start = float(tile[1])
            dec_end = float(tile[2])
            ra_start_deg = float(tile[3]) * 15.0
            ra_end_deg = float(tile[4]) * 15.0
            vlass_id = tile[5]
            rows.append((tile_id, dec_start, dec_end, ra_start_deg, ra_end_deg, vlass_id))
        except Exception:
            continue

    df = pd.DataFrame(
        rows,
        columns=["tile_id", "dec_start", "dec_end", "ra_start_deg", "ra_end_deg", "vlass_id"]
    )
    return df


def find_candidate_tiles(df: pd.DataFrame, ra_deg: float, dec_deg: float, im_size_arcsec: float = 1.0) -> pd.DataFrame:
    """
    Find tiles intersecting a small box around (ra_deg, dec_deg).

    Uses your original 9-point check:
      corners, center, mid-edges.
    """
    im_size_deg = im_size_arcsec / 3600.0
    ra_left, ra_right = ra_deg - im_size_deg, ra_deg + im_size_deg
    dec_down, dec_up = dec_deg - im_size_deg, dec_deg + im_size_deg

    ra_pts = np.array([ra_left, ra_right, ra_left, ra_right, ra_deg, ra_deg, ra_deg, ra_left, ra_right])
    dec_pts = np.array([dec_up, dec_down, dec_down, dec_up, dec_deg, dec_up, dec_down, dec_deg, dec_deg])

    # Vectorized containment: a tile is a candidate if ANY of the points is inside it.
    candidates = []
    for _, row in df.iterrows():
        inside_ra = (ra_pts > row.ra_start_deg) & (ra_pts < row.ra_end_deg)
        inside_dec = (dec_pts > row.dec_start) & (dec_pts < row.dec_end)
        if np.any(inside_ra & inside_dec):
            candidates.append(row)

    if not candidates:
        return df.iloc[0:0]

    return pd.DataFrame(candidates)


# -----------------------------
# VLASS quicklook parsing
# -----------------------------

def list_quicklook_dirs(vlass_id: str, tile_id: str) -> Tuple[str, List[str]]:
    """
    List quicklook directory names for a (vlass_id, tile_id).
    Returns (base_url, list_of_dirnames).

    For v2 epochs (e.g. VLASS1.1v2, VLASS1.2v2) the directory prefix
    on disk is "VLASS1.1.ql..." / "VLASS1.2.ql...", i.e. we strip 'v2'
    before ".ql". This matches the original manual script.
    """
    base_url = f"https://archive-new.nrao.edu/vlass/quicklook/{vlass_id}/{tile_id}/"
    r = requests.get(base_url, timeout=15)
    r.raise_for_status()
    html = r.text

    matches = re.findall(JNAME_REGEX, html)
    if not matches:
        return base_url, []

    # Directory prefix used on the archive for quicklook products:
    #   VLASS1.1v2 -> VLASS1.1.ql...
    #   VLASS1.2v2 -> VLASS1.2.ql...
    #   VLASS2.1   -> VLASS2.1.ql...
    #   VLASS2.2   -> VLASS2.2.ql...
    #   VLASS3.1   -> VLASS3.1.ql...
    #   VLASS3.2   -> VLASS3.2.ql...
    epoch_label = vlass_id.replace("v2", "")

    dirs = [f"{epoch_label}.ql.{tile_id}.{m}" for m in matches]
    return base_url, unique_list(dirs)


def jname_to_coord(jname: str) -> SkyCoord:
    """
    Convert a VLASS J-name (JHHMMSS+DDMMSS.xx.xxxx...) to SkyCoord.

    We parse only the HHMMSS and DDMMSS part.
    """
    # Example: J123456+654321.12.3456xyz
    hh = int(jname[1:3])
    mm = int(jname[3:5])
    ss = int(jname[5:7])
    sign_dd = jname[7]  # + or -
    dd = int(jname[8:10])
    dm = int(jname[10:12])
    ds = int(jname[12:14])

    ra_str = f"{hh}h{mm}m{ss}s"
    dec_str = f"{sign_dd}{dd}d{dm}m{ds}s"
    return SkyCoord(ra_str, dec_str, frame="icrs")


def pick_closest_quicklook_dir(
    vlass_ids_to_try: List[str],
    tile_id: str,
    ra_deg: float,
    dec_deg: float
) -> Optional[Tuple[str, str, float, float, float, str]]:
    """
    Try multiple VLASS epochs for a tile; pick the closest quicklook directory to target.

    Returns:
      (full_directory_name, base_url, ra2, dec2, min_sep_deg, used_vlass_id)
    """
    target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")

    for vid in vlass_ids_to_try:
        try:
            base_url, dirs = list_quicklook_dirs(vid, tile_id)
            if not dirs:
                continue

            seps = []
            coords = []
            for d in dirs:
                # Extract the J-name from "...tile.Jname"
                jmatch = re.search(JNAME_REGEX, d)
                if not jmatch:
                    continue
                c = jname_to_coord(jmatch.group(0))
                coords.append(c)
                seps.append(c.separation(target).deg)

            if not seps:
                continue

            min_pos = int(np.argmin(seps))
            closest_dir = dirs[min_pos]
            closest_coord = coords[min_pos]
            return (
                closest_dir,
                base_url,
                closest_coord.ra.deg,
                closest_coord.dec.deg,
                float(seps[min_pos]),
                vid
            )
        except Exception:
            continue

    return None


def extract_ms_from_pipescript(vlass_id: str, tile_id: str, full_dir: str) -> List[str]:
    """
    Read casa_pipescript.py and extract VLASS*.ms names.

    Note: URL uses vlass_id (may include 'v2') in the path, while
    'full_dir' uses the stripped epoch label for v2 cases.

    Also normalizes any trailing '_split' so that
    'VLASS3.1...._split.ms' -> 'VLASS3.1....ms'.
    """
    url = f"https://archive-new.nrao.edu/vlass/quicklook/{vlass_id}/{tile_id}/{full_dir}/casa_pipescript.py"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        text = r.text
        bases = re.findall(MS_REGEX, text)

        ms_clean = []
        for base in bases:
            # strip trailing '_split' if present
            if base.endswith("_split"):
                base = base[:-6]
            name = base + ".ms"
            ms_clean.append(name)

        return unique_list(ms_clean)
    except Exception:
        return []


def download_epoch_products(
    object_name: str,
    vlass_id: str,
    tile_id: str,
    full_dir: str,
    outdir: Path,
    overwrite: bool = False
) -> None:
    """
    Download the pbcor tt0 FITS subimage + casa_commands.log for one epoch.

    For v2 epochs, the quicklook directory uses the stripped label
    ("VLASS1.1.ql..." etc.), but the path still includes the full
    vlass_id ("VLASS1.1v2/...").
    """
    base = f"https://archive-new.nrao.edu/vlass/quicklook/{vlass_id}/{tile_id}/{full_dir}/"
    fits_name = f"{full_dir}.I.iter1.image.pbcor.tt0.subim.fits"
    log_name = "casa_commands.log"

    fits_url = base + fits_name
    log_url = base + log_name

    fits_out = outdir / f"{object_name}_VLASS_{vlass_id}_image.fits"
    log_out = outdir / f"{object_name}_{vlass_id}_casa_commands.log"

    ok_fits = download_file(fits_url, fits_out, overwrite=overwrite)
    ok_log = download_file(log_url, log_out, overwrite=overwrite)

    if ok_fits:
        print(f"[download] {fits_out.name}")
    if ok_log:
        print(f"[download] {log_out.name}")


def parse_logs_for_ms(object_name: str, outdir: Path) -> None:
    """
    Scan downloaded casa_commands logs for imported vis MS.
    """
    log_files = sorted(glob.glob(str(outdir / f"{object_name}*.log")))
    if not log_files:
        print("No .log files found.")
        return

    print("Found log files:")
    for lf in log_files:
        print("  ", lf)

    print("\nExtracted MS entries:")
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                if "hifv_importdata" in line and "vis=[" in line:
                    vis_match = re.search(r"vis=\[(?:'|\")(.+?)(?:_split)?\.ms(?:'|\")\]", line)
                    if vis_match:
                        vis_base = vis_match.group(1)
                        ms_name = vis_base + ".ms"
                        vmatch = re.search(r"(VLASS\d(?:\.\d)?)", vis_base, re.IGNORECASE)
                        vver = vmatch.group(1) if vmatch else "Unknown"
                        print(f"{vver} --> {ms_name}")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Find VLASS quicklook tiles/MS for a given RA/Dec."
    )
    ap.add_argument("object_name", help="Name/label for the target")
    ap.add_argument("ra_deg", type=float, help="Right Ascension in decimal degrees")
    ap.add_argument("dec_deg", type=float, help="Declination in decimal degrees")
    ap.add_argument("--tile-csv", default="Tile_Boundaries.csv", help="Path to Tile_Boundaries.csv")
    ap.add_argument("--imsize-arcsec", type=float, default=1.0, help="Half-size of search box (arcsec)")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--no-download", action="store_true", help="Do not download quicklook products")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads")
    ap.add_argument(
        "--epochs",
        default=None,
        help="Comma-separated VLASS epochs to try (e.g., VLASS1.2v2,VLASS2.2,VLASS3.2). "
             "Default: try the tile's epoch, then common fallbacks."
    )
    args = ap.parse_args()

    object_name = args.object_name
    ra_deg = float(args.ra_deg)
    dec_deg = float(args.dec_deg)
    outdir = Path(args.outdir)

    # Read tiles
    tiles_df = read_tile_boundaries(Path(args.tile_csv))
    if tiles_df.empty:
        raise RuntimeError("Tile_Boundaries.csv appears empty or unreadable.")

    # Candidate tiles
    cand = find_candidate_tiles(tiles_df, ra_deg, dec_deg, args.imsize_arcsec)
    if cand.empty:
        print("No tiles found covering the target.")
        return

    measurement_sets = []
    chosen_tile = None
    chosen_epoch = None
    chosen_full_dir = None
    chosen_base_url = None
    chosen_ra2 = None
    chosen_dec2 = None
    chosen_min_sep = None

    # For each tile that covers the box, grab at least one directory and MS list
    print("Candidate tiles:")
    for _, tile in cand.iterrows():
        print(
            f"  {tile.tile_id}  RA range=({tile.ra_start_deg:.5f},{tile.ra_end_deg:.5f}) "
            f"Dec range=({tile.dec_start:.5f},{tile.dec_end:.5f})"
        )

    for _, tile in cand.iterrows():
        tile_id = tile.tile_id
        tile_epoch = tile.vlass_id

        # Determine epochs to try
        if args.epochs:
            epochs_to_try = [e.strip() for e in args.epochs.split(",") if e.strip()]
        else:
            # try the tile epoch first, then likely alternates
            epochs_to_try = unique_list([
                tile_epoch,
                "VLASS1.1v2", "VLASS1.2v2",
                "VLASS2.1", "VLASS2.2",
                "VLASS3.1", "VLASS3.2",
            ])

        print(f"\nSearching epochs for tile {tile_id}: {', '.join(epochs_to_try)}")

        closest = pick_closest_quicklook_dir(epochs_to_try, tile_id, ra_deg, dec_deg)
        if closest is None:
            print("  No suitable quicklook directories found for this tile.")
            continue

        full_dir, base_url, ra2, dec2, min_sep, used_epoch = closest
        print(
            f"  Best match: {used_epoch}/{tile_id}/{full_dir} "
            f"(sep={min_sep:.4f} deg, center=({ra2:.5f},{dec2:.5f}))"
        )

        ms_list = extract_ms_from_pipescript(used_epoch, tile_id, full_dir)
        if not ms_list:
            print("  No MS names found in casa_pipescript.py for this directory.")
        else:
            print("  MS names from casa_pipescript.py:")
            for m in ms_list:
                print("   ", m)

        measurement_sets.extend(ms_list)

        # keep the first successful tile/epoch for downloads / further processing
        if chosen_tile is None:
            chosen_tile = tile_id
            chosen_epoch = used_epoch
            chosen_full_dir = full_dir
            chosen_base_url = base_url
            chosen_ra2, chosen_dec2, chosen_min_sep = ra2, dec2, min_sep

    measurement_sets = unique_list(measurement_sets)
    print(f"\nUnique measurement sets required for cluster {object_name}:")
    if not measurement_sets:
        print("  (none found)")
    else:
        for ms in measurement_sets:
            print(" ", ms)

    if chosen_tile is None:
        print("\nNo quicklook directories could be matched; nothing more to do.")
        print("\nDone.\n")
        return

    print(
        f"\nClosest quicklook dir:\n"
        f"  epoch: {chosen_epoch}\n"
        f"  tile : {chosen_tile}\n"
        f"  dir  : {chosen_full_dir}\n"
        f"  url  : {chosen_base_url}\n"
        f"  sep  : {chosen_min_sep:.4f} deg\n"
        f"  dir RA/Dec: {chosen_ra2:.6f}, {chosen_dec2:.6f}"
    )

    # ---------------------------------------------------------
    # Extract MS entries for all related epochs (1,2,3)
    # using casa_pipescript.py only, no FITS/log download.
    # ---------------------------------------------------------
    if args.epochs:
        related_epochs = [e.strip() for e in args.epochs.split(",") if e.strip()]
    else:
        if chosen_epoch in ("VLASS1.1v2", "VLASS2.1", "VLASS3.1"):
            related_epochs = ["VLASS1.1v2", "VLASS2.1", "VLASS3.1"]
        elif chosen_epoch in ("VLASS1.2v2", "VLASS2.2", "VLASS3.2"):
            related_epochs = ["VLASS1.2v2", "VLASS2.2", "VLASS3.2"]
        else:
            related_epochs = [chosen_epoch]

    # Extract the J-name part so we can rebuild directory names
    j_part = None
    split_tag = f".ql.{chosen_tile}."
    if split_tag in chosen_full_dir:
        j_part = chosen_full_dir.split(split_tag, 1)[1]

    print("\nExtracted MS entries:")
    for ep in related_epochs:
        if j_part is not None:
            epoch_label = ep.replace("v2", "")
            ep_dir = f"{epoch_label}.ql.{chosen_tile}.{j_part}"
        else:
            # Fallback: reuse chosen_full_dir if pattern not found
            ep_dir = chosen_full_dir

        ms_list_ep = extract_ms_from_pipescript(ep, chosen_tile, ep_dir)
        for ms in ms_list_ep:
            vmatch = re.search(r"(VLASS\d(?:\.\d)?)", ms, re.IGNORECASE)
            vver = vmatch.group(1) if vmatch else ep
            print(f"{vver} --> {ms}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

