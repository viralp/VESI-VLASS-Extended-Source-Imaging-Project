#!/usr/bin/env python3
"""
Flask web front-end for the VLASS joint-AWP pipeline.

Section 1:
  - Object name + RA/Dec (any format astropy understands).
  - Optional SIMBAD/NED name lookup (Simbad/NED search button).
  - Optional CSV upload with a list of objects:
       * either one column of object names
       * or object, RA, Dec columns
    For each row, find VLASS MS files and display per-object results.
  - Calls VLASS_Tile_Puller_Cleaned.py (the one in the same folder)
    and shows VLASS *.ms names parsed from its stdout.

Section 2:
  - Lets you set *all* parameters from VLASS_awp_joint_parameters.py
    in the same order, including:
      vis_list (via VLASS1/2/3 MS dropdowns)
      processing_flags (target_split, init_imaging, mask, selfcal,
                       delay_selfcal, pol_cleaning per epoch)
      joint_* flags, SPX flags, split parameters, imaging parameters,
      weighting parameters.
  - Writes VLASS_web_config.json, then calls:
      sbatch --mail-user=<email> run_casa.sh
  - If no MS is selected for VLASS1/2/3, it writes report_only=True so
    the CASA script can just generate HTML reports from existing FITS.

Section 3:
  - Per-epoch standard VLASS pipeline:
    * Choose MS for VLASS1/2/3 independently
    * Supply imagename, phasecenter RA/Dec (any reasonable format),
      imaging_mode, pblimit, search_radius_arcsec
    * For each selected MS:
         - Create folder named after MS (without .ms)
         - Create subfolder 'pipeline_run' inside it
         - Write SEIP_parameter.list in pipeline_run
         - Write casa_imaging_vlass.py in pipeline_run, with 'vis'
           set to the full path of the chosen MS
         - Write run_casa_pipeline.sh in pipeline_run and submit via sbatch
"""

import os
import json
import glob
import re
import socket
import subprocess
import shutil
import csv
import io
from pathlib import Path

from flask import Flask, request, render_template_string

from astropy.coordinates import SkyCoord
from astropy import units as u

app = Flask(__name__)

# Path to THIS directory (where VLASS_web_interface.py lives)
SCRIPT_DIR = Path(__file__).resolve().parent
TILE_PULLER_SCRIPT = SCRIPT_DIR / "VLASS_Tile_Puller_Cleaned.py"

# ----------------------------
# Helpers
# ----------------------------

# liberal .ms matcher for anything the tile puller prints
MS_REGEX = re.compile(r"([A-Za-z0-9._\-]+\.ms)")


def _extract_ms_from_stdout(stdout_text: str):
    """
    Extract measurement-set names ending in '.ms' from tile-puller stdout,
    without dumping the whole stdout back to the user.

    Strategy:
      - Look only at lines containing '.ms'
      - Split on whitespace
      - Strip simple punctuation around tokens
      - Keep unique tokens ending with '.ms'
    """
    ms_names = []
    for line in stdout_text.splitlines():
        if ".ms" not in line:
            continue
        for tok in re.split(r"\s+", line.strip()):
            tok = tok.strip(",:;()")
            if tok.endswith(".ms") and tok not in ms_names:
                ms_names.append(tok)
    return ms_names


def run_cmd(cmd, cwd=None):
    """
    Python 3.6 compatible subprocess wrapper.
    Returns CompletedProcess with .stdout/.stderr as text.
    """
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=cwd,
    )


def _fix_dotted_dms(dec_str: str) -> str:
    """
    Fix Dec strings like '+67.25.48.780' -> '+67 25 48.780' so Astropy can parse them.
    If it doesn’t look like that pattern, return unchanged.
    """
    s = (dec_str or "").strip()
    if not s:
        return s

    sign = ""
    if s[0] in "+-":
        sign = s[0]
        s = s[1:]

    parts = s.split(".")
    # e.g. ["67","25","48","780"] -> "67 25 48.780"
    if len(parts) >= 3:
        deg = parts[0]
        arcmin = parts[1]
        sec = ".".join(parts[2:])
        return f"{sign}{deg} {arcmin} {sec}"
    else:
        return sign + s


def parse_ra_dec_to_deg(ra_str: str, dec_str: str):
    """
    Parse RA/Dec strings into decimal degrees using astropy.

    Accepts, for example:
      RA:
        - "266.4194"
        - "17 45 40.6"
        - "17:45:40.6"
      Dec:
        - "-29.0042"
        - "-29 00 15"
        - "-29:00:15"
        - "+67.25.48.780" (normalized to +67 25 48.780)

    Strategy:
      1) Try both as simple decimal degrees.
      2) Try RA hourangle, Dec degrees (hms + dms).
      3) Let SkyCoord guess.
    """
    ra_s = (ra_str or "").strip()
    dec_s_raw = (dec_str or "").strip()
    if not ra_s or not dec_s_raw:
        raise ValueError("RA and Dec must not be empty.")

    # Normalize dotted DMS formats and stray commas
    dec_s = _fix_dotted_dms(dec_s_raw.replace(",", " "))

    # 1) simple decimal degrees (RA, Dec)
    try:
        return float(ra_s), float(dec_s)
    except ValueError:
        pass

    # 2) hourangle for RA, degrees for Dec (hms, dms)
    try:
        c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        pass

    # 3) let astropy guess
    c = SkyCoord(ra_s, dec_s, frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def format_phasecenter_string(ra_str: str, dec_str: str) -> str:
    """
    Build CASA-ready phasecenter string for SEIP_parameter.list like:
        'J2000 03:05:06.51 +67.25.48.780'

    RA:  h:m:s with colons
    Dec: d.m.s with dots, e.g. +DD.MM.SS.sss
    """
    ra_deg, dec_deg = parse_ra_dec_to_deg(ra_str, dec_str)
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")

    # RA in hms with colons, 2 decimal places in seconds
    ra_hms = c.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True)

    # Dec in dms with colons, 3 decimal places in seconds, then replace ':' with '.'
    dec_dms = c.dec.to_string(
        unit=u.deg, sep=":", alwayssign=True, precision=3, pad=True
    )
    dec_dotted = dec_dms.replace(":", ".")

    return f"J2000 {ra_hms} {dec_dotted}"


def call_tile_puller(object_name: str, ra_str: str, dec_str: str):
    """
    Run VLASS_Tile_Puller_Cleaned.py (from the same folder as this file)
    and parse the MS names from stdout. Returns (ms_list, message).

    On success:
        (["foo.ms", "bar.ms"], None)

    On failure:
        ([], "short error message ...")
    """
    if not TILE_PULLER_SCRIPT.exists():
        msg = (
            "VLASS_Tile_Puller_Cleaned.py not found at:\n"
            f"  {TILE_PULLER_SCRIPT}\n"
            "Make sure the working tile-puller script is in the same folder as VLASS_web_interface.py."
        )
        return [], msg

    try:
        ra_deg, dec_deg = parse_ra_dec_to_deg(ra_str, dec_str)
    except Exception as e:
        return [], f"Could not parse RA/Dec into degrees: {e}"

    cmd = [
        "python3",
        str(TILE_PULLER_SCRIPT),
        str(object_name),
        str(ra_deg),
        str(dec_deg),
    ]

    try:
        result = run_cmd(cmd)
    except FileNotFoundError:
        return [], f"Could not run {TILE_PULLER_SCRIPT} (python3 not found or path issue)."

    stdout = result.stdout or ""
    stderr = (result.stderr or "").strip()

    # Robust, but quiet, MS extraction
    ms_names = _extract_ms_from_stdout(stdout)

    # Error / success handling
    if result.returncode != 0:
        if stderr:
            tail = "\n".join(stderr.splitlines()[-8:])
            msg = (
                "Tile puller failed (non-zero exit).\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr tail:\n{tail}"
            )
        else:
            msg = (
                "Tile puller failed (non-zero exit, no stderr).\n"
                f"Command: {' '.join(cmd)}"
            )
        return [], msg

    if not ms_names:
        # Success exit code but we couldn't see any .ms in the output
        msg = (
            "Tile puller ran but no VLASS *.ms names were parsed from its output. "
            "Please run the tile puller manually on the command line to inspect details."
        )
        return [], msg

    # Success: just return the extracted MS list, no huge stdout blob
    return ms_names, None


# ---- SIMBAD and NED query helpers for STEP 1 ----

def query_simbad_coords(object_name: str):
    """
    Query SIMBAD for RA/Dec of the given object name.
    Returns ((RA_str, Dec_str), error_message) where one of them is None.
    """
    try:
        from astroquery.simbad import Simbad
    except ImportError:
        return None, "astroquery.simbad not installed in this environment."

    try:
        result = Simbad.query_object(object_name)
    except Exception as e:
        return None, f"SIMBAD query error: {e}"

    if result is None or len(result) == 0:
        return None, "Object not found in SIMBAD."

    ra = str(result["RA"][0])   # sexagesimal, e.g. '03 05 06.51'
    dec = str(result["DEC"][0]) # sexagesimal, e.g. '+67 25 48.8'
    return (ra, dec), None


def query_ned_coords(object_name: str):
    """
    Query NED for RA/Dec of the given object name.
    Returns ((RA_deg_str, Dec_deg_str), error_message) where one of them is None.
    """
    try:
        from astroquery.ned import Ned
    except ImportError:
        return None, "astroquery.ned not installed in this environment."

    try:
        result = Ned.query_object(object_name)
    except Exception as e:
        return None, f"NED query error: {e}"

    if result is None or len(result) == 0:
        return None, "Object not found in NED."

    ra_deg = float(result["RA"][0])
    dec_deg = float(result["DEC"][0])
    return (f"{ra_deg:.6f}", f"{dec_deg:.6f}"), None


def parse_bool_form(form, name, default):
    """Parse a True/False select element."""
    val = form.get(name, "True" if default else "False")
    return str(val).lower() in ("true", "1", "yes", "on")


def parse_pointingoff(s: str, default):
    """Parse pointingoff from a text input like '300,30' or '300 30'."""
    if not s.strip():
        return default
    try:
        parts = re.split(r"[,\s]+", s.strip())
        vals = [float(p) for p in parts if p]
        if len(vals) >= 2:
            return [vals[0], vals[1]]
        elif len(vals) == 1:
            return [vals[0], default[1]]
    except Exception:
        pass
    return default


def find_ms_in_cwd():
    """
    Return a sorted list of *.ms entries in the current working directory.
    Includes both directories and regular files ending with '.ms'.
    """
    ms_list = []
    for name in sorted(os.listdir(".")):
        path = os.path.join(".", name)
        if name.endswith(".ms") and (os.path.isdir(path) or os.path.isfile(path)):
            ms_list.append(name)
    return ms_list


def pick_epoch_ms(ms_list):
    """
    From a list of MS names, pick at most one MS per epoch (VLASS1/2/3).
    Returns (ms1, ms2, ms3).
    """
    ms1 = ms2 = ms3 = None
    others = []

    for m in ms_list:
        if "VLASS1" in m and ms1 is None:
            ms1 = m
        elif "VLASS2" in m and ms2 is None:
            ms2 = m
        elif "VLASS3" in m and ms3 is None:
            ms3 = m
        else:
            others.append(m)

    # If nothing matched VLASS1/2/3 explicitly, just fill from the front of the list
    remaining = [x for x in ms_list if x not in (ms1, ms2, ms3)]
    if ms1 is None and remaining:
        ms1 = remaining.pop(0)
    if ms2 is None and remaining:
        ms2 = remaining.pop(0)
    if ms3 is None and remaining:
        ms3 = remaining.pop(0)

    return ms1, ms2, ms3


# ----------------------------
# HTML template
# ----------------------------

TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>VESI – VLASS Extended Source Imaging</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 20px;
        background-color: #ffffff;
        color: #003366;
      }
      a { color: #0055aa; text-decoration: none; }
      a:hover { text-decoration: underline; }
      fieldset {
        margin-bottom: 1.5em;
        border: 1px solid #88aacc;
        border-radius: 6px;
      }
      legend {
        font-weight: bold;
        font-size: 1.05em;
        padding: 0 6px;
        color: #002244;
      }
      label { display: inline-block; width: 190px; vertical-align: top; }
      input[type=text] { width: 260px; }
      .ms-list { max-height: 260px; overflow-y: auto; border: 1px solid #ccc; padding: 6px; }
      .ms-block { border: 1px solid #ddd; padding: 8px; margin-bottom: 8px; border-radius: 4px; }
      .section-title { font-size: 1.05em; font-weight: bold; margin-top: 0.8em; }
      select { min-width: 80px; }
      textarea { width: 260px; height: 4em; }
      pre {
        font-size: 0.9em;
        white-space: pre-wrap;
        border-left: 4px solid #88aacc;
        padding-left: 8px;
      }
      button {
        background-color: #0055aa;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 6px 14px;
        cursor: pointer;
        font-weight: bold;
      }
      button:hover {
        background-color: #003f7d;
      }
      .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
      }
      .header-logo img {
        max-height: 120px;
      }
      .header-title {
        text-align: center;
        flex: 1;
        margin: 0 20px;
      }
      .header-title img {
        max-height: 120px;
      }
      .header-title h2 {
        font-size: 1.0em;
        margin: 4px 0 8px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }
      .header-title p {
        margin: 4px 0 0;
        font-size: 0.95em;
      }
    </style>
  </head>
  <body>
    <div class="header">
      <div class="header-logo">
        <!-- Replace with actual VLA logo path if different -->
        <img src="{{ url_for('static', filename='vla_logo.png') }}" alt="VLA logo">
      </div>
      <div class="header-title">
        <!-- Center VESI logo image instead of text -->
        <img src="{{ url_for('static', filename='VESI.png') }}" alt="VESI logo">
        <h2>VLASS Extended Source Imaging</h2>
        <p>
          Download data from
          <a href="https://data.nrao.edu/" target="_blank">VLA Archive</a>
        </p>
      </div>
      <div class="header-logo">
        <!-- Replace with actual VLASS logo path if different -->
        <img src="{{ url_for('static', filename='vlass_logo.png') }}" alt="VLASS logo">
      </div>
    </div>

    {% if message %}
      <pre><b>{{ message }}</b></pre>
    {% endif %}

    <form method="post" action="{{ url_for('index') }}" enctype="multipart/form-data">
      <!-- SECTION 1: FIND MS -->
      <fieldset>
        <legend>STEP 1 – Find VLASS Measurement Sets</legend>
        <p>
          <label for="object_name">Object name:</label>
          <input type="text" id="object_name" name="object_name"
                 value="{{ object_name or '' }}">
          <button type="submit" name="action" value="query_name">Simbad/NED search</button>
        </p>
        <p>
          <label for="ra_tile">R.A.:</label>
          <input type="text" id="ra_tile" name="ra_tile"
                 value="{{ ra_tile or '' }}">
        </p>
        <p>
          <label for="dec_tile">Dec.:</label>
          <input type="text" id="dec_tile" name="dec_tile"
                 value="{{ dec_tile or '' }}">
        </p>

        {% if simbad_coords or simbad_error or ned_coords or ned_error %}
          <div style="margin-top: 0.5em; margin-bottom: 0.5em;">
            {% if simbad_coords %}
              <p><b>SIMBAD:</b> RA {{ simbad_coords[0] }}, Dec {{ simbad_coords[1] }}</p>
            {% elif simbad_error %}
              <p><b>SIMBAD:</b> {{ simbad_error }}</p>
            {% endif %}

            {% if ned_coords %}
              <p><b>NED:</b> RA {{ ned_coords[0] }} deg, Dec {{ ned_coords[1] }} deg</p>
            {% elif ned_error %}
              <p><b>NED:</b> {{ ned_error }}</p>
            {% endif %}
          </div>
        {% endif %}

        <p>
          <label for="csv_file">Object list CSV:</label>
          <input type="file" id="csv_file" name="csv_file" accept=".csv">
          <button type="submit" name="action" value="process_csv">Process CSV</button>
        </p>

        <p>
          <button type="submit" name="action" value="find_ms">Find VLASS MS</button>
        </p>

        {% if found_ms %}
          <p><b>VLASS MSes:</b></p>
          <div class="ms-list">
            <ul>
            {% for m in found_ms %}
              <li>{{ m }}</li>
            {% endfor %}
            </ul>
          </div>
        {% endif %}

        {% if csv_results %}
          <p><b>VLASS MSes from CSV:</b></p>
          <div class="ms-list">
            {% for row in csv_results %}
              <div class="ms-block">
                <p><b>{{ row['object'] }}</b>
                   {% if row['ra'] or row['dec'] %}
                     (RA {{ row['ra'] or '?' }}, Dec {{ row['dec'] or '?' }})
                   {% endif %}
                </p>
                {% if row['error'] %}
                  <p>{{ row['error'] }}</p>
                {% else %}
                  <ul>
                    <li>VLASS1: {{ row['ms1'] or 'None found' }}</li>
                    <li>VLASS2: {{ row['ms2'] or 'None found' }}</li>
                    <li>VLASS3: {{ row['ms3'] or 'None found' }}</li>
                  </ul>
                {% endif %}
              </div>
            {% endfor %}
          </div>
        {% endif %}
      </fieldset>

      <!-- SECTION 2: PIPELINE OPTIONS & JOB SUBMISSION -->
      <fieldset>
        <legend>STEP 2 – Configure Imaging Pipeline & Submit Job</legend>

        <!-- 2.1 vis_list + processing_flags -->
        <p class="section-title">Observation Selection & Per-Epoch Options</p>
        <div class="ms-list">
          {% for i in vlass_indices %}
          <div class="ms-block">
            <p><b>VLASS {{ i }}</b></p>
            <p>
              <label>Measurement set:</label>
              <select name="vis{{i}}">
                <option value="">-- none --</option>
                {% for m in local_ms %}
                  <option value="{{m}}"
                    {% if selected_vis[i] == m %}selected{% endif %}>{{m}}</option>
                {% endfor %}
              </select>
            </p>
            <p>
              <label>Target_split:</label>
              <select name="vis{{i}}_target_split">
                <option value="True"  {% if per_flags[i]['target_split'] %}selected{% endif %}>True</option>
                <option value="False" {% if not per_flags[i]['target_split'] %}selected{% endif %}>False</option>
              </select>
            </p>
            <p>
              <label>Init_imaging:</label>
              <select name="vis{{i}}_init_imaging">
                <option value="True"  {% if per_flags[i]['init_imaging'] %}selected{% endif %}>True</option>
                <option value="False" {% if not per_flags[i]['init_imaging'] %}selected{% endif %}>False</option>
              </select>
            </p>
            <p>
              <label>Mask:</label>
              <select name="vis{{i}}_mask">
                <option value="True"  {% if per_flags[i]['mask'] %}selected{% endif %}>True</option>
                <option value="False" {% if not per_flags[i]['mask'] %}selected{% endif %}>False</option>
              </select>
            </p>
            <p>
              <label>Selfcal:</label>
              <select name="vis{{i}}_selfcal">
                <option value="True"  {% if per_flags[i]['selfcal'] %}selected{% endif %}>True</option>
                <option value="False" {% if not per_flags[i]['selfcal'] %}selected{% endif %}>False</option>
              </select>
            </p>
            <p>
              <label>Delay_selfcal:</label>
              <select name="vis{{i}}_delay_selfcal">
                <option value="True"  {% if per_flags[i]['delay_selfcal'] %}selected{% endif %}>True</option>
                <option value="False" {% if not per_flags[i]['delay_selfcal'] %}selected{% endif %}>False</option>
              </select>
            </p>
            <p>
              <label>Pol_cleaning:</label>
              <select name="vis{{i}}_pol_cleaning">
                <option value="True"  {% if per_flags[i]['pol_cleaning'] %}selected{% endif %}>True</option>
                <option value="False" {% if not per_flags[i]['pol_cleaning'] %}selected{% endif %}>False</option>
              </select>
            </p>
          </div>
          {% endfor %}
        </div>

        <!-- 2.2 joint_deconvolution / joint_selfcal / joint_pol_deconvolution / awp_image_cube -->
        <p class="section-title">Joint Imaging & A-Projection Options</p>
        <p>
          <label for="joint_deconvolution">Joint_deconvolution:</label>
          <select id="joint_deconvolution" name="joint_deconvolution">
            <option value="True"  {% if joint_deconvolution %}selected{% endif %}>True</option>
            <option value="False" {% if not joint_deconvolution %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="joint_selfcal">Joint_selfcal:</label>
          <select id="joint_selfcal" name="joint_selfcal">
            <option value="True"  {% if joint_selfcal %}selected{% endif %}>True</option>
            <option value="False" {% if not joint_selfcal %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="joint_pol_deconvolution">Joint_pol_deconvolution:</label>
          <select id="joint_pol_deconvolution" name="joint_pol_deconvolution">
            <option value="True"  {% if joint_pol_deconvolution %}selected{% endif %}>True</option>
            <option value="False" {% if not joint_pol_deconvolution %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="awp_image_cube">Awp_image_cube:</label>
          <select id="awp_image_cube" name="awp_image_cube">
            <option value="True"  {% if awp_image_cube %}selected{% endif %}>True</option>
            <option value="False" {% if not awp_image_cube %}selected{% endif %}>False</option>
          </select>
        </p>

        <!-- 2.3 SPX flags -->
        <p class="section-title">Spectral Index Mapping</p>
        <p>
          <label for="spx_map_VLASS1">Spx_map_VLASS1:</label>
          <select id="spx_map_VLASS1" name="spx_map_VLASS1">
            <option value="True"  {% if spx_map_VLASS1 %}selected{% endif %}>True</option>
            <option value="False" {% if not spx_map_VLASS1 %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="spx_map_VLASS2">Spx_map_VLASS2:</label>
          <select id="spx_map_VLASS2" name="spx_map_VLASS2">
            <option value="True"  {% if spx_map_VLASS2 %}selected{% endif %}>True</option>
            <option value="False" {% if not spx_map_VLASS2 %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="spx_map_VLASS3">Spx_map_VLASS3:</label>
          <select id="spx_map_VLASS3" name="spx_map_VLASS3">
            <option value="True"  {% if spx_map_VLASS3 %}selected{% endif %}>True</option>
            <option value="False" {% if not spx_map_VLASS3 %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="spx_map_VLASS_combine">Spx_map_VLASS_combine:</label>
          <select id="spx_map_VLASS_combine" name="spx_map_VLASS_combine">
            <option value="True"  {% if spx_map_VLASS_combine %}selected{% endif %}>True</option>
            <option value="False" {% if not spx_map_VLASS_combine %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="weighted_combine_maps">Weighted_combine_maps:</label>
          <select id="weighted_combine_maps" name="weighted_combine_maps">
            <option value="True"  {% if weighted_combine_maps %}selected{% endif %}>True</option>
            <option value="False" {% if not weighted_combine_maps %}selected{% endif %}>False</option>
          </select>
        </p>

        <!-- 2.4 split parameters -->
        <p class="section-title">Data Selection & Split Parameters</p>
        <p>
          <label for="colname">Colname:</label>
          <input type="text" id="colname" name="colname"
                 value="{{ colname or 'corrected' }}">
        </p>
        <p>
          <label for="channelaverage">Channelaverage:</label>
          <select id="channelaverage" name="channelaverage">
            <option value="True"  {% if channelaverage %}selected{% endif %}>True</option>
            <option value="False" {% if not channelaverage %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="timeaverage">Timeaverage:</label>
          <select id="timeaverage" name="timeaverage">
            <option value="True"  {% if timeaverage %}selected{% endif %}>True</option>
            <option value="False" {% if not timeaverage %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="Ra_1">Ra_1:</label>
          <input type="text" id="Ra_1" name="Ra_1"
                 value="{{ Ra_1 or '266.4194' }}">
        </p>
        <p>
          <label for="Dec_1">Dec_1:</label>
          <input type="text" id="Dec_1" name="Dec_1"
                 value="{{ Dec_1 or '-29.0042' }}">
        </p>
        <p>
          <label for="R">R (deg):</label>
          <input type="text" id="R" name="R"
                 value="{{ R or '0.1' }}">
        </p>

        <!-- 2.5 masking parameter -->
        <p class="section-title">Masking Threshold</p>
        <p>
          <label for="mask_threshold">Mask_threshold (sigma):</label>
          <input type="text" id="mask_threshold" name="mask_threshold"
                 value="{{ mask_threshold or '3' }}">
        </p>

        <!-- 2.6 imaging parameters -->
        <p class="section-title">Imaging Configuration</p>
        <p>
          <label for="cell">Cell (e.g. 0.6arcsec):</label>
          <input type="text" id="cell" name="cell"
                 value="{{ cell or '0.6arcsec' }}">
        </p>
        <p>
          <label for="niter">Niter:</label>
          <input type="text" id="niter" name="niter"
                 value="{{ niter or '20000' }}">
        </p>
        <p>
          <label for="gridder">Gridder:</label>
          <input type="text" id="gridder" name="gridder"
                 value="{{ gridder or 'awproject' }}">
        </p>
        <p>
          <label for="specmode">Specmode:</label>
          <input type="text" id="specmode" name="specmode"
                 value="{{ specmode or 'mfs' }}">
        </p>
        <p>
          <label for="parallel">Parallel:</label>
          <select id="parallel" name="parallel">
            <option value="True"  {% if parallel %}selected{% endif %}>True</option>
            <option value="False" {% if not parallel %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="imagename_base">Imagename_base:</label>
          <input type="text" id="imagename_base" name="imagename_base"
                 value="{{ imagename_base or 'Sgr_pol' }}">
        </p>
        <p>
          <label for="field">Field:</label>
          <input type="text" id="field" name="field"
                 value="{{ field or '' }}">
        </p>
        <p>
          <label for="spw">Spw:</label>
          <input type="text" id="spw" name="spw"
                 value="{{ spw or '' }}">
        </p>
        <p>
          <label for="uvrange">Uvrange:</label>
          <input type="text" id="uvrange" name="uvrange"
                 value="{{ uvrange or '' }}">
        </p>
        <p>
          <label for="mask">Mask:</label>
          <input type="text" id="mask" name="mask"
                 value="{{ mask or '' }}">
        </p>
        <p>
          <label for="usepointing">Usepointing:</label>
          <select id="usepointing" name="usepointing">
            <option value="True"  {% if usepointing %}selected{% endif %}>True</option>
            <option value="False" {% if not usepointing %}selected{% endif %}>False</option>
          </select>
        </p>
        <p>
          <label for="pointingoff">Pointingoff (e.g. 300,30):</label>
          <input type="text" id="pointingoff" name="pointingoff"
                 value="{{ pointingoff_text or '300,30' }}">
        </p>

        <!-- 2.7 weighted combined maps -->
        <p class="section-title">Weighted Combination of Epochs</p>
        <p>
          <label for="array_size">Array_size:</label>
          <input type="text" id="array_size" name="array_size"
                 value="{{ array_size or '5' }}">
        </p>
        <p>
          <label for="model_combine_method">Model combine method:</label>
          <select id="model_combine_method" name="model_combine_method">
            <option value="mean" {% if model_combine_method == 'mean' %}selected{% endif %}>Simple mean</option>
            <option value="rms" {% if model_combine_method == 'rms' %}selected{% endif %}>RMS weighted</option>
          </select>
        </p>
        <p>
          <label for="model_target_beam">Target beam (bmaj,bmin,bpa):</label>
          <input type="text" id="model_target_beam" name="model_target_beam"
                 value="{{ model_target_beam or '' }}"
                 placeholder="e.g. 2.5,2.0,45.0">
        </p>

        <!-- 2.8 SLURM email -->
        <p class="section-title">Job Submission & Email Notification (Joint AWP)</p>
        <p>
          <label for="email2">Notification email:</label>
          <input type="text" id="email2" name="email2"
                 value="{{ email or '' }}">
        </p>

        <p>
          <button type="submit" name="action" value="submit_job">
            Write VLASS_web_config.json & Submit SBATCH
          </button>
        </p>
      </fieldset>

      <!-- SECTION 3: STANDARD VLASS PIPELINE -->
      <fieldset>
        <legend>STEP 3 – Run Standard VLASS Pipeline</legend>

        <p class="section-title">Per-Epoch Inputs</p>
        <div class="ms-list">
          {% for i in vlass_indices %}
          <div class="ms-block">
            <p><b>VLASS {{ i }}</b></p>
            <p>
              <label>Measurement set:</label>
              <select name="std_vis{{i}}">
                <option value="">-- none --</option>
                {% for m in local_ms %}
                  <option value="{{m}}"
                    {% if std_params[i]['vis'] == m %}selected{% endif %}>{{m}}</option>
                {% endfor %}
              </select>
            </p>
            <p>
              <label>Imagename:</label>
              <input type="text" name="std_imagename{{i}}"
                     value="{{ std_params[i]['imagename'] }}">
            </p>
            <p>
              <label>Phasecenter RA:</label>
              <input type="text" name="std_phase_ra{{i}}"
                     value="{{ std_params[i]['phase_ra'] }}"
                     placeholder="e.g. 03:05:06.51 or 46.2771">
            </p>
            <p>
              <label>Phasecenter Dec:</label>
              <input type="text" name="std_phase_dec{{i}}"
                     value="{{ std_params[i]['phase_dec'] }}"
                     placeholder="e.g. +67:25:48.78 or 67.4302">
            </p>
            <p>
              <label>Imaging_mode:</label>
              <input type="text" name="std_imaging_mode{{i}}"
                     value="{{ std_params[i]['imaging_mode'] }}">
            </p>
            <p>
              <label>Pblimit:</label>
              <input type="text" name="std_pblimit{{i}}"
                     value="{{ std_params[i]['pblimit'] }}">
            </p>
            <p>
              <label>Search_radius_arcsec:</label>
              <input type="text" name="std_search_radius{{i}}"
                     value="{{ std_params[i]['search_radius'] }}"
                     placeholder="e.g. 360">
            </p>
          </div>
          {% endfor %}
        </div>

        <p class="section-title">Job Submission & Email Notification (Standard VLASS)</p>
        <p>
          <label for="email3">Notification email:</label>
          <input type="text" id="email3" name="email3"
                 value="{{ email or '' }}">
        </p>

        <p>
          <button type="submit" name="action" value="submit_std_pipeline">
            Create SEIP_parameter.list & Submit Standard VLASS Jobs
          </button>
        </p>
      </fieldset>
    </form>
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    # Defaults mirroring VLASS_awp_joint_parameters.py
    object_name = ""
    ra_tile = ""
    dec_tile = ""

    # Parameter defaults
    joint_deconvolution = True
    joint_selfcal = True
    joint_pol_deconvolution = True
    awp_image_cube = True

    spx_map_VLASS1 = True
    spx_map_VLASS2 = True
    spx_map_VLASS3 = True
    spx_map_VLASS_combine = True
    weighted_combine_maps = True

    colname = "corrected"
    channelaverage = False
    timeaverage = False
    Ra_1 = "266.4194"
    Dec_1 = "-29.0042"
    R = "0.1"

    mask_threshold = "3"

    cell = "0.6arcsec"
    niter = "20000"
    gridder = "awproject"
    specmode = "mfs"
    parallel = False
    imagename_base = "Sgr_pol"
    field = ""
    spw = ""
    uvrange = ""
    mask = ""
    usepointing = True
    pointingoff = [300.0, 30.0]
    pointingoff_text = "300,30"

    array_size = "5"

    model_combine_method = "mean"  # "mean" or "rms"
    model_target_beam = ""      # e.g. "2.5,2.0,45.0" (bmaj,bmin arcsec, bpa deg)

    email = ""

    found_ms = []
    message = None

    # SIMBAD / NED outputs for template
    simbad_coords = None
    simbad_error = None
    ned_coords = None
    ned_error = None

    # CSV results structure
    csv_results = []

    # Local MS entries for dropdown – *.ms dirs/files only
    try:
        local_ms = find_ms_in_cwd()
    except Exception:
        local_ms = []

    vlass_indices = [1, 2, 3]

    # Per-epoch default flags
    per_flags = {
        1: {
            "target_split": False,
            "init_imaging": True,
            "mask": False,
            "selfcal": True,
            "delay_selfcal": False,
            "pol_cleaning": True,
        },
        2: {
            "target_split": False,
            "init_imaging": True,
            "mask": False,
            "selfcal": True,
            "delay_selfcal": False,
            "pol_cleaning": True,
        },
        3: {
            "target_split": False,
            "init_imaging": True,
            "mask": False,
            "selfcal": True,
            "delay_selfcal": False,
            "pol_cleaning": True,
        },
    }
    selected_vis = {1: "", 2: "", 3: ""}

    # Section 3: per-epoch standard pipeline options
    std_params = {}
    for i in vlass_indices:
        std_params[i] = {
            "vis": "",
            "imagename": "",
            "phase_ra": "",
            "phase_dec": "",
            "imaging_mode": "VLASS-SE-CONT-MOSAIC",
            "pblimit": "0.15",
            "search_radius": "",
        }

    if request.method == "POST":
        action = request.form.get("action")

        if action == "query_name":
            # Only SIMBAD/NED lookup; no tile puller
            object_name = request.form.get("object_name", "").strip()
            ra_tile = request.form.get("ra_tile", "").strip()
            dec_tile = request.form.get("dec_tile", "").strip()

            if not object_name:
                message = "Please enter an object name for SIMBAD/NED search."
            else:
                simbad_coords, simbad_error = query_simbad_coords(object_name)
                ned_coords, ned_error = query_ned_coords(object_name)

        elif action == "find_ms":
            # SECTION 1 – tile puller only, RA/Dec required
            object_name = request.form.get("object_name", "").strip()
            ra_tile = request.form.get("ra_tile", "").strip()
            dec_tile = request.form.get("dec_tile", "").strip()

            if not object_name or not ra_tile or not dec_tile:
                message = "Please fill object name, RA and Dec for tile puller."
            else:
                found_ms, msg = call_tile_puller(object_name, ra_tile, dec_tile)
                if msg:
                    message = msg

        elif action == "process_csv":
            # Process an uploaded CSV of objects
            object_name = request.form.get("object_name", "").strip()
            ra_tile = request.form.get("ra_tile", "").strip()
            dec_tile = request.form.get("dec_tile", "").strip()

            file = request.files.get("csv_file")
            if not file or file.filename == "":
                message = "Please choose a CSV file to process."
            else:
                try:
                    content = file.read().decode("utf-8")
                    reader = csv.reader(io.StringIO(content))
                    rows = [r for r in reader if any(c.strip() for c in r)]
                except Exception as exc:
                    message = f"Could not read CSV file: {exc}"
                    rows = []

                if rows:
                    # Detect simple header with RA/DEC column names
                    start_index = 0
                    first = rows[0]
                    lower = [c.strip().lower() for c in first]
                    if any("ra" == c or c.startswith("ra ") for c in lower) and \
                       any(c.startswith("dec") for c in lower):
                        start_index = 1

                    for row in rows[start_index:]:
                        if not row:
                            continue
                        obj = row[0].strip()
                        if not obj:
                            continue

                        ra_val = None
                        dec_val = None
                        tile_msg = None
                        ms_list = []

                        # Try to use RA/Dec from columns if present
                        if len(row) >= 3:
                            cand_ra = row[1].strip()
                            cand_dec = row[2].strip()
                            if cand_ra and cand_dec:
                                try:
                                    # Just test parsing; use original strings if OK
                                    _ = parse_ra_dec_to_deg(cand_ra, cand_dec)
                                    ra_val = cand_ra
                                    dec_val = cand_dec
                                except Exception:
                                    ra_val = None
                                    dec_val = None

                        # If RA/Dec not usable, fall back to SIMBAD
                        if ra_val is None or dec_val is None:
                            coords, err = query_simbad_coords(obj)
                            if coords:
                                ra_val, dec_val = coords
                            else:
                                # Can't proceed without coordinates
                                csv_results.append(
                                    {
                                        "object": obj,
                                        "ra": None,
                                        "dec": None,
                                        "ms1": None,
                                        "ms2": None,
                                        "ms3": None,
                                        "error": f"Could not get coordinates: {err}",
                                    }
                                )
                                continue

                        # Now we have RA/Dec strings; call tile puller
                        ms_list, tile_msg = call_tile_puller(obj, ra_val, dec_val)
                        ms1, ms2, ms3 = pick_epoch_ms(ms_list)

                        csv_results.append(
                            {
                                "object": obj,
                                "ra": ra_val,
                                "dec": dec_val,
                                "ms1": ms1,
                                "ms2": ms2,
                                "ms3": ms3,
                                "error": tile_msg,
                            }
                        )
                else:
                    message = "CSV file appears to be empty or unreadable."

        elif action == "submit_job":
            # SECTION 2 ONLY (joint AWP)
            # per-epoch MS and flags
            for i in vlass_indices:
                vis_val = request.form.get(f"vis{i}", "").strip()
                selected_vis[i] = vis_val
                per_flags[i]["target_split"] = parse_bool_form(
                    request.form, f"vis{i}_target_split", per_flags[i]["target_split"]
                )
                per_flags[i]["init_imaging"] = parse_bool_form(
                    request.form, f"vis{i}_init_imaging", per_flags[i]["init_imaging"]
                )
                per_flags[i]["mask"] = parse_bool_form(
                    request.form, f"vis{i}_mask", per_flags[i]["mask"]
                )
                per_flags[i]["selfcal"] = parse_bool_form(
                    request.form, f"vis{i}_selfcal", per_flags[i]["selfcal"]
                )
                per_flags[i]["delay_selfcal"] = parse_bool_form(
                    request.form, f"vis{i}_delay_selfcal", per_flags[i]["delay_selfcal"]
                )
                per_flags[i]["pol_cleaning"] = parse_bool_form(
                    request.form, f"vis{i}_pol_cleaning", per_flags[i]["pol_cleaning"]
                )

            joint_deconvolution = parse_bool_form(
                request.form, "joint_deconvolution", joint_deconvolution
            )
            joint_selfcal = parse_bool_form(
                request.form, "joint_selfcal", joint_selfcal
            )
            joint_pol_deconvolution = parse_bool_form(
                request.form, "joint_pol_deconvolution", joint_pol_deconvolution
            )
            awp_image_cube = parse_bool_form(
                request.form, "awp_image_cube", awp_image_cube
            )

            spx_map_VLASS1 = parse_bool_form(
                request.form, "spx_map_VLASS1", spx_map_VLASS1
            )
            spx_map_VLASS2 = parse_bool_form(
                request.form, "spx_map_VLASS2", spx_map_VLASS2
            )
            spx_map_VLASS3 = parse_bool_form(
                request.form, "spx_map_VLASS3", spx_map_VLASS3
            )
            spx_map_VLASS_combine = parse_bool_form(
                request.form, "spx_map_VLASS_combine", spx_map_VLASS_combine
            )
            weighted_combine_maps = parse_bool_form(
                request.form, "weighted_combine_maps", weighted_combine_maps
            )

            colname = request.form.get("colname", colname).strip()
            channelaverage = parse_bool_form(
                request.form, "channelaverage", channelaverage
            )
            timeaverage = parse_bool_form(
                request.form, "timeaverage", timeaverage
            )
            Ra_1 = request.form.get("Ra_1", Ra_1).strip()
            Dec_1 = request.form.get("Dec_1", Dec_1).strip()
            R = request.form.get("R", R).strip()

            mask_threshold = request.form.get(
                "mask_threshold", mask_threshold
            ).strip()

            cell = request.form.get("cell", cell).strip()
            niter = request.form.get("niter", niter).strip()
            gridder = request.form.get("gridder", gridder).strip()
            specmode = request.form.get("specmode", specmode).strip()
            parallel = parse_bool_form(request.form, "parallel", parallel)
            imagename_base = request.form.get(
                "imagename_base", imagename_base
            ).strip()
            field = request.form.get("field", field).strip()
            spw = request.form.get("spw", spw).strip()
            uvrange = request.form.get("uvrange", uvrange).strip()
            mask = request.form.get("mask", mask).strip()
            usepointing = parse_bool_form(
                request.form, "usepointing", usepointing
            )
            pointingoff_text = request.form.get(
                "pointingoff", pointingoff_text
            ).strip()
            pointingoff = parse_pointingoff(pointingoff_text, pointingoff)

            array_size = request.form.get("array_size", array_size).strip()
            model_combine_method = request.form.get("model_combine_method", model_combine_method).strip() or "mean"
            model_target_beam = request.form.get("model_target_beam", model_target_beam).strip()

            email = request.form.get("email2", "").strip()

            # Build vis_list and processing_flags for JSON
            vis_list = [selected_vis[i] for i in vlass_indices]
            processing_flags = [per_flags[i] for i in vlass_indices]

            # Report-only mode if no vis selected
            report_only = all((v.strip() == "" for v in vis_list))

            if not email:
                message = "Please enter an email address for job notification (Section 2)."
            else:
                # Convert numeric + RA/Dec for CASA
                try:
                    R_val = float(R)
                except ValueError:
                    R_val = 0.1
                try:
                    mask_val = float(mask_threshold)
                except ValueError:
                    mask_val = 3.0
                try:
                    niter_val = int(float(niter))
                except ValueError:
                    niter_val = 20000

                try:
                    Ra_1_deg, Dec_1_deg = parse_ra_dec_to_deg(Ra_1, Dec_1)
                except Exception as e:
                    message = f"Could not parse Ra_1/Dec_1: {e}"
                else:
                    cfg = {
                        "vis_list": vis_list,
                        "processing_flags": processing_flags,
                        "joint_deconvolution": joint_deconvolution,
                        "joint_selfcal": joint_selfcal,
                        "joint_pol_deconvolution": joint_pol_deconvolution,
                        "awp_image_cube": awp_image_cube,
                        "spx_map_VLASS1": spx_map_VLASS1,
                        "spx_map_VLASS2": spx_map_VLASS2,
                        "spx_map_VLASS3": spx_map_VLASS3,
                        "spx_map_VLASS_combine": spx_map_VLASS_combine,
                        "weighted_combine_maps": weighted_combine_maps,
                        "colname": colname,
                        "channelaverage": channelaverage,
                        "timeaverage": timeaverage,
                        "Ra_1": Ra_1_deg,
                        "Dec_1": Dec_1_deg,
                        "R": R_val,
                        "mask_threshold": mask_val,
                        "cell": cell,
                        "niter": niter_val,
                        "gridder": gridder,
                        "specmode": specmode,
                        "parallel": parallel,
                        "imagename_base": imagename_base,
                        "field": field,
                        "spw": spw,
                        "uvrange": uvrange,
                        "mask": mask,
                        "usepointing": usepointing,
                        "pointingoff": pointingoff,
                        "model_combine_method": model_combine_method,
                        "model_target_beam": model_target_beam,
                        "array_size": int(float(array_size)) if array_size else 5,
                        "report_only": report_only,
                    }
                    with open("VLASS_web_config.json", "w") as f:
                        json.dump(cfg, f, indent=2)

                    # Submit job
                    cmd = ["sbatch", f"--mail-user={email}", "run_casa.sh"]
                    try:
                        result = run_cmd(cmd)
                        if result.returncode == 0:
                            message = (
                                "Submitted job via sbatch. Slurm says: "
                                + (result.stdout or "").strip()
                            )
                        else:
                            message = (
                                f"sbatch returned code {result.returncode}: "
                                f"{(result.stderr or '').strip()}"
                            )
                    except FileNotFoundError:
                        message = (
                            "Could not find sbatch command. "
                            "Are you on the cluster head node?"
                        )

        elif action == "submit_std_pipeline":
            # SECTION 3 ONLY – standard VLASS pipeline
            email = request.form.get("email3", "").strip()

            # Fill std_params from form
            for i in vlass_indices:
                std_params[i]["vis"] = request.form.get(f"std_vis{i}", "").strip()
                std_params[i]["imagename"] = request.form.get(f"std_imagename{i}", "").strip()
                std_params[i]["phase_ra"] = request.form.get(f"std_phase_ra{i}", "").strip()
                std_params[i]["phase_dec"] = request.form.get(f"std_phase_dec{i}", "").strip()
                std_params[i]["imaging_mode"] = (
                    request.form.get(f"std_imaging_mode{i}", std_params[i]["imaging_mode"]).strip()
                    or "VLASS-SE-CONT-MOSAIC"
                )
                std_params[i]["pblimit"] = (
                    request.form.get(f"std_pblimit{i}", std_params[i]["pblimit"]).strip()
                    or "0.15"
                )
                std_params[i]["search_radius"] = request.form.get(
                    f"std_search_radius{i}", std_params[i]["search_radius"]
                ).strip()

            if not email:
                message = "Please enter an email address for job notification (Section 3)."
            else:
                jobs = []
                casa_template = SCRIPT_DIR / "casa_imaging_vlass.py"
                if not casa_template.exists():
                    message = (
                        f"Cannot find casa_imaging_vlass.py at {casa_template}. "
                        "Place the standard VLASS pipeline script next to VLASS_web_interface.py."
                    )
                else:
                    template_text = casa_template.read_text()

                    for i in vlass_indices:
                        ms_name = std_params[i]["vis"]
                        if not ms_name:
                            continue  # skip empty epoch

                        ms_path = Path(ms_name)
                        if not ms_path.exists():
                            jobs.append(
                                f"VLASS {i}: MS '{ms_name}' not found in working directory; skipped."
                            )
                            continue

                        ms_base = ms_path.name
                        ms_root = ms_base[:-3] if ms_base.endswith(".ms") else ms_base
                        job_dir = Path(ms_root)
                        pipeline_dir = job_dir / "pipeline_run"

                        # Create ms_root and pipeline_run directories
                        job_dir.mkdir(exist_ok=True)
                        pipeline_dir.mkdir(exist_ok=True)

                        # Build phasecenter string from RA/Dec (robust parser, dotted Dec)
                        try:
                            phasecenter = format_phasecenter_string(
                                std_params[i]["phase_ra"], std_params[i]["phase_dec"]
                            )
                        except Exception as exc:
                            jobs.append(
                                f"VLASS {i}: could not parse phasecenter RA/Dec: {exc}"
                            )
                            continue

                        # Numeric parameters
                        try:
                            pblimit_val = float(std_params[i]["pblimit"] or "0.15")
                        except ValueError:
                            pblimit_val = 0.15

                        search_radius_val = None
                        if std_params[i]["search_radius"]:
                            try:
                                search_radius_val = float(std_params[i]["search_radius"])
                            except ValueError:
                                search_radius_val = None

                        imagename = std_params[i]["imagename"] or ms_root

                        # --- Write SEIP_parameter.list in pipeline_run ---
                        seip_path = pipeline_dir / "SEIP_parameter.list"
                        with open(seip_path, "w") as f:
                            f.write(f"imagename='{imagename}'\n")
                            f.write(f"phasecenter='{phasecenter}'\n")
                            f.write(f"imaging_mode='{std_params[i]['imaging_mode']}'\n")
                            f.write(f"pblimit={pblimit_val}\n")
                            if search_radius_val is not None:
                                f.write(f"search_radius_arcsec={search_radius_val}\n")

                        # --- Write casa_imaging_vlass.py in pipeline_run, injecting vis path ---
                        full_vis_path = ms_path.resolve()
                        injected_line = f"vis = r'{full_vis_path}'\n"

                        import re

                        # Prefer to replace any existing "vis = ..." line in the template
                        m = re.search(r"^vis\s*=.*$", template_text, flags=re.MULTILINE)
                        if m:
                           # Replace the whole vis=... line with our injected one
                           new_text = template_text[:m.start()] + injected_line + template_text[m.end():]
                        elif "__rethrow_casa_exceptions" in template_text:
                           # Fallback: insert just before __rethrow_casa_exceptions (older templates)
                           new_text = template_text.replace(
                           "__rethrow_casa_exceptions",
                           injected_line + "__rethrow_casa_exceptions",
                           1,
                           )
                        else:
                            # Last resort: just prepend
                            new_text = injected_line + "\n" + template_text

                        imaging_script_path = pipeline_dir / "casa_imaging_vlass.py"
                        imaging_script_path.write_text(new_text)


                        # --- Write run_casa_pipeline.sh in pipeline_run ---
                        run_sh = pipeline_dir / "run_casa_pipeline.sh"
                        run_sh_text = f"""#!/bin/sh
#Don't put any commands before the #SBATCH options or they will not work
#SBATCH --mem=200G                        # Amount of memory needed by the whole job.
#SBATCH --time=14-00:00                   # Expected runtime
#SBATCH --mail-type=END,FAIL              # Send email when Jobs end or fail
#SBATCH --mail-user={email}
#SBATCH -n 1

# casa's python requires a DISPLAY for matplotlib, so create a virtual X server
xvfb-run -d /lustre/aoc/users/vparekh/CASA/casa-6.6.6-17-pipeline-2025.1.0.19-py3.10.el8/bin/casa --pipeline --nogui -c casa_imaging_vlass.py
"""
                        with open(run_sh, "w") as f:
                            f.write(run_sh_text)
                        os.chmod(run_sh, 0o755)

                        # --- Submit job from pipeline_run (one sbatch per epoch) ---
                        try:
                            result = run_cmd(["sbatch", run_sh.name], cwd=str(pipeline_dir))
                            if result.returncode == 0:
                                jobs.append(
                                    f"VLASS {i}: submitted from {pipeline_dir} – {(result.stdout or '').strip()}"
                                )
                            else:
                                jobs.append(
                                    f"VLASS {i}: sbatch returned {result.returncode} – {(result.stderr or '').strip()}"
                                )
                        except FileNotFoundError:
                            jobs.append(
                                f"VLASS {i}: sbatch command not found on this host."
                            )

                if jobs:
                    message = "Section 3 results:\n" + "\n".join(jobs)
                else:
                    message = "Section 3: no VLASS epochs had MS selected; nothing to do."

    return render_template_string(
        TEMPLATE,
        object_name=object_name,
        ra_tile=ra_tile,
        dec_tile=dec_tile,
        found_ms=found_ms,
        local_ms=local_ms,
        vlass_indices=vlass_indices,
        per_flags=per_flags,
        selected_vis=selected_vis,
        joint_deconvolution=joint_deconvolution,
        joint_selfcal=joint_selfcal,
        joint_pol_deconvolution=joint_pol_deconvolution,
        awp_image_cube=awp_image_cube,
        spx_map_VLASS1=spx_map_VLASS1,
        spx_map_VLASS2=spx_map_VLASS2,
        spx_map_VLASS3=spx_map_VLASS3,
        spx_map_VLASS_combine=spx_map_VLASS_combine,
        weighted_combine_maps=weighted_combine_maps,
        colname=colname,
        channelaverage=channelaverage,
        timeaverage=timeaverage,
        Ra_1=Ra_1,
        Dec_1=Dec_1,
        R=R,
        mask_threshold=mask_threshold,
        cell=cell,
        niter=niter,
        gridder=gridder,
        specmode=specmode,
        parallel=parallel,
        imagename_base=imagename_base,
        field=field,
        spw=spw,
        uvrange=uvrange,
        mask=mask,
        usepointing=usepointing,
        pointingoff_text=pointingoff_text,
        array_size=array_size,
        model_combine_method=model_combine_method,
        model_target_beam=model_target_beam,
        email=email,
        message=message,
        std_params=std_params,
        simbad_coords=simbad_coords,
        simbad_error=simbad_error,
        ned_coords=ned_coords,
        ned_error=ned_error,
        csv_results=csv_results,
    )


if __name__ == "__main__":
    # Start from this port (can override with env var VLASS_WEB_PORT)
    base_port = int(os.environ.get("VLASS_WEB_PORT", 5000))
    max_tries = 20  # try base_port, base_port+1, ..., base_port+19

    port = base_port
    for _ in range(max_tries):
        # Try to bind the port ourselves first.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                # Port is busy, try the next one
                print(f"Port {port} in use, trying {port + 1}...")
                port += 1
                continue

        # If we got here, the port is free (we just bound it).
        print(f"Starting VLASS web interface on http://127.0.0.1:{port}")
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
        break
    else:
        print(f"No free port found in range {base_port}–{base_port + max_tries - 1}")

