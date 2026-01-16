#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimator Streamlit 2 — clean repatch
- STEP-driven ops (holes & pockets)
- Auto-populated costing (sync with STEP)
- Setup breakdown cards (Tool changes, Face, Profile, Blind/Through holes)
- Preview G-code + 3D toolpath (optional)
NOTE: Preview G-code is for visualization/costing only.
"""

import math
import json
import os
import hashlib
import tempfile
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import pandas as pd
import streamlit as st

# Optional: Plotly for toolpath visualization
try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

# --- Streamlit compatibility helper for rerun() across versions ---
def _st_rerun():
    try:
        st.rerun()  # Streamlit >=1.26
    except Exception:
        try:
            st.experimental_rerun()  # older versions
        except Exception:
            pass

# --- Optional CAD backends ----------------------------------------------------
HAVE_OCC = False
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods_Face
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cone, GeomAbs_BSplineSurface, GeomAbs_BezierSurface
    HAVE_OCC = True
except Exception:
    HAVE_OCC = False

HAVE_CQ = False
try:
    import cadquery as cq
    HAVE_CQ = True
except Exception:
    HAVE_CQ = False

# ==============================================================================
# Data structures & core timing/costing helpers
# ==============================================================================
@dataclass
class Machine:
    max_feed_xy: float  # mm/min
    max_feed_z: float   # mm/min
    rapid_xy: float     # mm/min
    rapid_z: float      # mm/min
    accel: float        # mm/s^2 (same for cut/rapid here)
    tool_change_s: float

# trapezoidal motion time estimator (do not simulate real CNC controllers)
def trapezoidal_time(L_mm: float, v_target_mm_min: float, a_mm_s2: float) -> float:
    if L_mm <= 0 or v_target_mm_min <= 0 or a_mm_s2 <= 0:
        return 0.0
    v = v_target_mm_min / 60.0  # mm/s
    t_acc = v / a_mm_s2
    x_acc = 0.5 * a_mm_s2 * t_acc**2
    if 2 * x_acc >= L_mm:
        # triangular profile
        return 2 * math.sqrt(max(L_mm, 0.0) / a_mm_s2)
    else:
        x_cruise = L_mm - 2 * x_acc
        return 2 * t_acc + (x_cruise / v)


def mill_time(
    machine: Machine,
    cut_length_mm: float,
    cut_feed_mm_min: float,
    rapid_length_mm: float,
    rapid_axis: str = 'xy',
    toolchanges: int = 0,
    scale_factor: float = 1.10
) -> float:
    t_cut = trapezoidal_time(
        cut_length_mm,
        min(max(cut_feed_mm_min, 0.0), machine.max_feed_xy),
        machine.accel
    )
    v_rapid = machine.rapid_xy if rapid_axis.lower() != 'z' else machine.rapid_z
    t_rapid = trapezoidal_time(rapid_length_mm, v_rapid, machine.accel)
    t_tc = max(toolchanges, 0) * machine.tool_change_s
    return max((t_cut + t_rapid + t_tc) * max(scale_factor, 0.01), 0.0)


def drill_time(
    machine: Machine,
    holes: int,
    depth_mm: float,
    drill_feed_mm_min: float,
    approach_mm: float = 2.0,
    retract_mm: float = 2.0,
    scale_factor: float = 1.05
) -> float:
    t_one = (
        trapezoidal_time(approach_mm, machine.rapid_z, machine.accel)
        + trapezoidal_time(depth_mm, min(drill_feed_mm_min, machine.max_feed_z), machine.accel)
        + trapezoidal_time(retract_mm, machine.rapid_z, machine.accel)
    )
    return max(holes, 0) * t_one * max(scale_factor, 0.01)


def cost_breakdown(
    cycle_time_min: float,
    setup_min: float,
    batch_qty: int,
    machine_rate_per_hour: float,
    labor_rate_per_hour: float,
    material_cost_per_part: float,
    tooling_cost_per_part: float,
    overhead_pct: float = 0.15,
    margin_pct: float = 0.10
) -> Dict[str, float]:
    machine_cost = (max(cycle_time_min, 0.0) / 60.0) * max(machine_rate_per_hour, 0.0)
    labor_cost = (max(setup_min, 0.0) / 60.0) * max(labor_rate_per_hour, 0.0) / max(batch_qty, 1)
    subtotal = (
        machine_cost
        + labor_cost
        + max(material_cost_per_part, 0.0)
        + max(tooling_cost_per_part, 0.0)
    )
    overhead = subtotal * max(overhead_pct, 0.0)
    total = subtotal + overhead
    price = total * (1 + max(margin_pct, 0.0))
    return {
        'machine_cost': machine_cost,
        'labor_cost': labor_cost,
        'material_cost': material_cost_per_part,
        'tooling_cost': tooling_cost_per_part,
        'overhead': overhead,
        'price_per_part': price,
    }

# --- Auto-fill CNC costing defaults from STEP + ops (helpers) ---

def _machine_rates_from_preset(preset: str) -> tuple[float, float]:
    if preset == "Desktop router":
        return 50.0, 30.0
    if preset == "VMC midsize":
        return 80.0, 35.0
    if preset == "Haas MiniMill":
        return 70.0, 35.0
    return 75.0, 35.0


def _estimate_setup_minutes(mill_df: pd.DataFrame | None,
                            drill_df: pd.DataFrame | None,
                            hole_info: dict | None) -> float:
    base = 20.0
    toolchanges = 0
    if isinstance(mill_df, pd.DataFrame) and not mill_df.empty:
        try:
            toolchanges = int(pd.to_numeric(mill_df.get("toolchanges", 0)).fillna(0).sum())
        except Exception:
            pass
    groups = 0
    if isinstance(hole_info, dict):
        groups = len(hole_info.get("hole_groups", []) or [])
    return float(base + 5.0 * toolchanges + (10.0 if groups >= 5 else (5.0 if groups >= 2 else 0.0)))


def _estimate_tooling_cost_per_part(mill_df: pd.DataFrame | None,
                                    drill_df: pd.DataFrame | None,
                                    batch_qty: int = 10) -> float:
    m = len(mill_df) if isinstance(mill_df, pd.DataFrame) else 0
    d = len(drill_df) if isinstance(drill_df, pd.DataFrame) else 0
    ops = max(m + d, 1)
    return round(0.15 * ops, 2)


def _material_cost_from_mass_session() -> float | None:
    mass_g = st.session_state.get("step_mass_g")
    if mass_g is None:
        return None
    mat_cost_per_kg = st.session_state.get("last_material_cost_per_kg")
    if mat_cost_per_kg is None:
        mat_cost_per_kg = 12.0
    return float((mass_g / 1000.0) * float(mat_cost_per_kg))


def autofill_cnc_costing_defaults(preset: str):
    mach_rate, labor_rate = _machine_rates_from_preset(preset)
    mill_df = st.session_state.get("mill_df")
    drill_df = st.session_state.get("drill_df")
    hole_info = st.session_state.get("hole_info")
    setup_min = _estimate_setup_minutes(mill_df, drill_df, hole_info)
    batch_qty = int(st.session_state.get("autofill_batch_qty", 10))
    material_cost = _material_cost_from_mass_session()
    if material_cost is None:
        material_cost = 12.0
    tooling_cost = _estimate_tooling_cost_per_part(mill_df, drill_df, batch_qty=batch_qty)
    overhead_pct = 0.15
    margin_pct = 0.10
    shift_hours = 8.0
    queue_days = 1.0
    st.session_state["cost_setup_min_default"] = float(setup_min)
    st.session_state["cost_batch_qty_default"] = int(batch_qty)
    st.session_state["cost_machine_rate_default"] = float(mach_rate)
    st.session_state["cost_labor_rate_default"] = float(labor_rate)
    st.session_state["cost_material_per_part_default"] = float(round(material_cost, 2))
    st.session_state["cost_tooling_per_part_default"] = float(tooling_cost)
    st.session_state["cost_overhead_pct_default"] = float(overhead_pct * 100.0)
    st.session_state["cost_margin_pct_default"] = float(margin_pct * 100.0)
    st.session_state["cost_shift_hours_default"] = float(shift_hours)
    st.session_state["cost_queue_days_default"] = float(queue_days)

# ==============================================================================
# Unit helpers
# ==============================================================================

def _unit_to_mm_factor(unit: str) -> float:
    u = (unit or "").strip().upper()
    if u in ("MM", "MILLIMETER", "MILLIMETRE"):
        return 1.0
    if u in ("CM", "CENTIMETER", "CENTIMETRE"):
        return 10.0
    if u in ("M", "METER", "METRE"):
        return 1000.0
    if u in ("INCH", "IN", "INCHES"):
        return 25.4
    if u in ("FT", "FOOT", "FEET"):
        return 304.8
    return 1.0


def _scale_bbox_to_mm(bbox: Tuple[float, float, float, float, float, float], factor: float):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return (xmin * factor, ymin * factor, zmin * factor, xmax * factor, ymax * factor, zmax * factor)

# ==============================================================================
# CadQuery tessellation for approx area (fallback)
# ==============================================================================

def _approx_surface_area_from_tris(shape) -> float:
    area = 0.0
    try:
        tess = shape.tessellate(angular_tolerance=0.1, linear_tolerance=0.1)
        verts = tess[0]
        tris = tess[1]
        def tri_area(a, b, c):
            ax, ay, az = a
            bx, by, bz = b
            cx, cy, cz = c
            ux, uy, uz = (bx - ax, by - ay, bz - az)
            vx, vy, vz = (cx - ax, cy - ay, cz - az)
            cx_ = uy * vz - uz * vy
            cy_ = uz * vx - ux * vz
            cz_ = ux * vy - uy * vx
            return 0.5 * math.sqrt(cx_ * cx_ + cy_ * cy_ + cz_ * cz_)
        for i, j, k in tris:
            area += tri_area(verts[i], verts[j], verts[k])
    except Exception:
        area = 0.0
    return float(area)

# ==============================================================================
# STEP property readers
# ==============================================================================

def read_step_properties(file_path: str) -> Optional[Tuple[float, float, Tuple[float, float, float, float, float, float], str, str]]:
    """
    Return (volume_mm3, area_mm2, (xmin,ymin,zmin,xmax,ymax,zmax), display_units, backend) or None.
    All numeric outputs are converted/scaled to **mm** and **mm^2/mm^3**.
    """
    # Prefer pythonocc-core
    if HAVE_OCC:
        try:
            reader = STEPControl_Reader()
            status = reader.ReadFile(file_path)
            if status != IFSelect_RetDone:
                return None
            reader.TransferRoots()
            shape = reader.OneShape()
            # Volume & surface area (in file units)
            gprops = GProp_GProps()
            brepgprop.VolumeProperties(shape, gprops)
            volume = float(gprops.Mass())
            gprops = GProp_GProps()
            brepgprop.SurfaceProperties(shape, gprops)
            area = float(gprops.Mass())
            # Bounding box (in file units)
            box = Bnd_Box()
            brepbndlib_Add(shape, box)
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
            # Units
            units = "MM"
            try:
                units_list = reader.FileUnits()
                if isinstance(units_list, (list, tuple)) and len(units_list) > 0:
                    units = str(units_list[0]).upper()
            except Exception:
                pass
            f = _unit_to_mm_factor(units)
            bbox_mm = _scale_bbox_to_mm((xmin, ymin, zmin, xmax, ymax, zmax), f)
            volume_mm3 = volume * (f ** 3)
            area_mm2 = area * (f ** 2)
            return volume_mm3, area_mm2, bbox_mm, "mm", "pythonocc-core"
        except Exception:
            pass
    # CadQuery fallback (units=mm)
    if HAVE_CQ:
        try:
            wp = cq.importers.importStep(file_path)
            solids = wp.vals()
            if not solids:
                try:
                    s = wp.val()
                    solids = [s] if s is not None else []
                except Exception:
                    solids = []
            if not solids:
                return None
            total_volume = 0.0
            xmin = ymin = zmin = float('inf')
            xmax = ymax = zmax = float('-inf')
            total_area = 0.0
            for s in solids:
                total_volume += float(s.Volume())
                bb = s.BoundingBox()
                xmin = min(xmin, bb.xmin)
                ymin = min(ymin, bb.ymin)
                zmin = min(zmin, bb.zmin)
                xmax = max(xmax, bb.xmax)
                ymax = max(ymax, bb.ymax)
                zmax = max(zmax, bb.zmax)
                total_area += _approx_surface_area_from_tris(s)
            return total_volume, total_area, (xmin, ymin, zmin, xmax, ymax, zmax), "mm", "cadquery"
        except Exception:
            return None
    return None

# ==============================================================================
# OCC helpers for feature heuristics
# ==============================================================================

def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _norm(v):
    l = math.sqrt(max(_dot(v, v), 1e-12))
    return (v[0]/l, v[1]/l, v[2]/l)

def _angle_deg(u, v):
    ux, uy, uz = _norm(u); vx, vy, vz = _norm(v)
    d = max(min(ux*vx + uy*vy + uz*vz, 1.0), -1.0)
    return math.degrees(math.acos(d))

def _is_parallel(u, v, tol=1e-3):
    u = _norm(u); v = _norm(v)
    return abs(_dot(u, v)) > (1 - tol)


def _bbox_of_face(face):
    box = Bnd_Box()
    brepbndlib_Add(face, box)
    return box.Get()  # (xmin, ymin, zmin, xmax, ymax, zmax)


def _load_occ_shape_and_unit_factor(file_path: str):
    if not HAVE_OCC:
        return None, 1.0
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != IFSelect_RetDone:
        return None, 1.0
    reader.TransferRoots()
    shape = reader.OneShape()
    units = "MM"
    try:
        ulist = reader.FileUnits()
        if isinstance(ulist, (list, tuple)) and ulist:
            units = str(ulist[0]).upper()
    except Exception:
        pass
    return shape, _unit_to_mm_factor(units)

# Tiny tap-drill table (for labels)
_METRIC_TAP_DRILL_MM = {3.0: 2.5, 4.0: 3.3, 5.0: 4.2, 6.0: 5.0, 8.0: 6.8, 10.0: 8.5, 12.0: 10.2, 16.0: 14.0}

def _is_tap_drill_like(d_mm: float, tol=0.15):
    for nominal, drill in _METRIC_TAP_DRILL_MM.items():
        if abs(d_mm - drill) <= tol:
            return f"M{int(nominal)} (drill ~{drill:.1f})"
    return ""


def _detect_cylindrical_holes_occ_with_centers(shape, unit_factor_to_mm: float):
    holes = []
    try:
        cyl_faces = []
        cone_faces = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = topods_Face(exp.Current())
            try:
                surf = BRepAdaptor_Surface(face, True)
                stype = surf.GetType()
                if stype == GeomAbs_Cylinder:
                    c = surf.Cylinder()
                    ax = c.Axis().Direction()
                    cyl_faces.append({
                        "face": face,
                        "r": c.Radius(),
                        "dir": _norm((ax.X(), ax.Y(), ax.Z()))
                    })
                elif stype == GeomAbs_Cone:
                    cn = surf.Cone()
                    ax = cn.Axis().Direction()
                    cone_faces.append({
                        "face": face,
                        "semi": cn.SemiAngle(),
                        "dir": _norm((ax.X(), ax.Y(), ax.Z()))
                    })
            except Exception:
                pass
            exp.Next()

        def _length_along_axis(face, axis):
            (xmin, ymin, zmin, xmax, ymax, zmax) = _bbox_of_face(face)
            pts = [(xmin,ymin,zmin),(xmin,ymin,zmax),(xmin,ymax,zmin),(xmin,ymax,zmax),
                   (xmax,ymin,zmin),(xmax,ymin,zmax),(xmax,ymax,zmin),(xmax,ymax,zmax)]
            dots = [p[0]*axis[0] + p[1]*axis[1] + p[2]*axis[2] for p in pts]
            return (max(dots) - min(dots)) * unit_factor_to_mm

        for cf in cyl_faces:
            face = cf["face"]
            r_mm = cf["r"] * unit_factor_to_mm
            axis = cf["dir"]
            d_mm = 2.0 * r_mm
            depth_mm = max(_length_along_axis(face, axis), 0.0)
            is_cbore = any(_is_parallel(axis, other["dir"]) and (other["r"] > cf["r"] * 1.05)
                           for other in cyl_faces if other is not cf)
            is_csk = (not is_cbore) and any(_is_parallel(axis, con["dir"]) for con in cone_faces)
            kind = "counterbore" if is_cbore else ("countersink" if is_csk else "simple")
            # center from face bbox mid-point (file units -> mm)
            bxmin, bymin, bzmin, bxmax, bymax, bzmax = _bbox_of_face(face)
            cx = (bxmin + bxmax) * 0.5 * unit_factor_to_mm
            cy = (bymin + bymax) * 0.5 * unit_factor_to_mm
            cz = (bzmin + bzmax) * 0.5 * unit_factor_to_mm
            if r_mm > 0.25 and depth_mm > 0.2 * r_mm:
                holes.append({
                    "diameter_mm": d_mm,
                    "depth_mm": depth_mm,
                    "axis_dir": axis,
                    "type": kind,
                    "tapped_guess": _is_tap_drill_like(d_mm),
                    "center_mm": (cx, cy, cz)
                })
    except Exception:
        pass
    return holes


def _cluster_holes(holes, d_tol_mm: float = 0.2, depth_tol_mm: float = 0.5):
    groups = []
    for h in holes:
        placed = False
        for g in groups:
            if (
                g["type"] == h.get("type")
                and g.get("tapped_guess") == h.get("tapped_guess")
                and abs(g["diameter_mm"] - h["diameter_mm"]) <= d_tol_mm
                and abs(g["depth_mm"] - h["depth_mm"]) <= depth_tol_mm
            ):
                g["depth_mm"] = (g["depth_mm"] * g["holes"] + h["depth_mm"]) / (g["holes"] + 1)
                g["holes"] += 1
                placed = True
                break
        if not placed:
            groups.append({
                "diameter_mm": h["diameter_mm"],
                "depth_mm": h["depth_mm"],
                "holes": 1,
                "type": h.get("type", "simple"),
                "tapped_guess": h.get("tapped_guess", "")
            })
    groups.sort(key=lambda x: (x["diameter_mm"], x["depth_mm"]))
    return groups


def _detect_planar_pockets_occ(shape, unit_factor_to_mm: float, bbox_mm):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox_mm
    z_top = zmax
    pockets = []
    try:
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = topods_Face(exp.Current())
            try:
                surf = BRepAdaptor_Surface(face, True)
                if surf.GetType() != GeomAbs_Plane:
                    exp.Next(); continue
                pln = surf.Plane()
                n = pln.Axis().Direction()
                n_vec = _norm((n.X(), n.Y(), n.Z()))
                if abs(n_vec[2] - 1.0) > 0.02:
                    exp.Next(); continue
                fxmin, fymin, fzmin, fxmax, fymax, fzmax = _bbox_of_face(face)
                z_face = (fzmin + fzmax) / 2.0 * unit_factor_to_mm
                depth_mm = z_top - z_face
                if depth_mm < 0.5:
                    exp.Next(); continue
                gp = GProp_GProps()
                brepgprop.SurfaceProperties(face, gp)
                area_mm2 = gp.Mass() * (unit_factor_to_mm ** 2)
                if area_mm2 < 50.0:
                    exp.Next(); continue
                pockets.append({"area_mm2": float(area_mm2), "depth_mm": float(depth_mm)})
            except Exception:
                pass
            exp.Next()
    except Exception:
        pass
    return pockets, None

# Pocket-based cut-length estimate ------------------------------------------------

def _estimate_mill_length_from_pockets(pockets, tool_d_mm: float, stepover_ratio: float) -> float:
    if tool_d_mm <= 0:
        return 0.0
    stepover = max(tool_d_mm * max(stepover_ratio, 0.05), 0.05)
    total_len = 0.0
    for p in pockets:
        area = max(float(p.get("area_mm2", 0.0)), 0.0)
        if area <= 0:
            continue
        tracks = area / max(stepover, 0.05)
        eff_len = max(math.sqrt(area), tool_d_mm)
        total_len += (tracks / max(eff_len, 0.1))
    return total_len * 1.0

# ==============================================================================
# Auto-generation of ops from STEP
# ==============================================================================

def autogen_ops_from_step(
    file_path: str,
    bbox_mm: Tuple[float, float, float, float, float, float],
    machine: Machine,
    rough_tool_d_mm: float = 10.0,
    profile_tool_d_mm: float = 8.0,
    rough_feed_mm_min: float = 2400.0,
    profile_feed_mm_min: float = 2200.0,
    stepover_ratio: float = 0.6,
    stepdown_ratio: float = 0.5,
    default_drill_feed_small: float = 500.0,
    default_drill_feed_large: float = 800.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox_mm
    dx = max(xmax - xmin, 0.0)
    dy = max(ymax - ymin, 0.0)
    dz = max(zmax - zmin, 0.0)

    hole_groups = []
    holes_xy = []
    pockets = []
    if HAVE_OCC:
        shape, f = _load_occ_shape_and_unit_factor(file_path)
        if shape is not None:
            holes = _detect_cylindrical_holes_occ_with_centers(shape, f)
            hole_groups = _cluster_holes(holes)
            holes_xy = holes[:]  # keep raw list for coordinates
            pockets, _ = _detect_planar_pockets_occ(shape, f, bbox_mm)

    # Facing (bbox-based raster)
    long_dim = max(dx, dy)
    short_dim = min(dx, dy)
    stepover = max(rough_tool_d_mm * stepover_ratio, 0.1)
    n_passes = max(int(math.ceil(short_dim / stepover)), 1)
    facing_cut_len = n_passes * long_dim
    facing_rapid_len = max(n_passes - 1, 0) * stepover

    # Profile + pockets
    pocket_len = _estimate_mill_length_from_pockets(pockets, profile_tool_d_mm, stepover_ratio)
    perimeter = 2.0 * (dx + dy)
    stepdown = max(profile_tool_d_mm * stepdown_ratio, 0.1)
    n_depth = max(int(math.ceil(dz / stepdown)), 1) if dz > 0 else 1
    profile_cut_len = perimeter * n_depth + pocket_len
    profile_rapid_len = n_depth * profile_tool_d_mm

    mill_rows = [
        {
            "operation": "Facing (auto)",
            "cut_length_mm": round(facing_cut_len, 1),
            "cut_feed_mm_min": rough_feed_mm_min,
            "rapid_length_mm": round(facing_rapid_len, 1),
            "rapid_axis": "xy",
            "toolchanges": 1,
            "scale_factor": 1.10,
        },
        {
            "operation": "Perimeter+Pockets (auto)",
            "cut_length_mm": round(profile_cut_len, 1),
            "cut_feed_mm_min": profile_feed_mm_min,
            "rapid_length_mm": round(profile_rapid_len, 1),
            "rapid_axis": "xy",
            "toolchanges": 1,
            "scale_factor": 1.12,
        },
    ]
    mill_df = pd.DataFrame(mill_rows)
    mill_df = mill_df[["cut_length_mm", "cut_feed_mm_min", "rapid_length_mm", "rapid_axis", "toolchanges", "scale_factor"]].assign(operation=mill_df.get("operation", ""))

    drill_rows = []
    if hole_groups:
        for g in hole_groups:
            d = g["diameter_mm"]
            depth = g["depth_mm"]
            holes_ct = g["holes"]
            drill_feed = default_drill_feed_small if d < 5.0 else default_drill_feed_large
            note = f"⌀{d:.1f} mm {g.get('type','').strip() or 'simple'}"
            if g.get("tapped_guess"):
                note += f" | {g['tapped_guess']} (guess)"
            drill_rows.append({
                "holes": holes_ct,
                "depth_mm": round(depth, 2),
                "drill_feed_mm_min": drill_feed,
                "approach_mm": 2.0,
                "retract_mm": 2.0,
                "scale_factor": 1.05,
                "note": note,
            })
    drill_df = (pd.DataFrame(drill_rows)[["holes", "depth_mm", "drill_feed_mm_min", "approach_mm", "retract_mm", "scale_factor", "note"]]
                if drill_rows else pd.DataFrame(columns=["holes", "depth_mm", "drill_feed_mm_min", "approach_mm", "retract_mm", "scale_factor", "note"]))

    summary = {"pockets": pockets, "hole_groups": hole_groups, "holes_xy": holes_xy}
    return mill_df, drill_df, summary

# ==============================================================================
# G-code generation & preview
# ==============================================================================

def _fmt(v):
    return f"{v:.3f}"


def gcode_header(safe_z=5.0, spindle=8000):
    return [
        "(Preview G-code generated by Streamlit app)",
        "G90 G17 G21",
        f"G0 Z{_fmt(safe_z)}",
        f"S{int(spindle)} M3"
    ]


def gcode_footer():
    return ["M5", "G0 Z5.000", "G0 X0 Y0", "M30"]


def gcode_facing(bbox_mm, stepdown=0.5, stepover=0.6, tool_d=10.0, feed=2400.0, safe_z=5.0):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox_mm
    longX = (xmax - xmin) >= (ymax - ymin)
    width = (ymax - ymin) if longX else (xmax - xmin)
    pitch = max(tool_d * stepover, 0.1)
    n = max(int(math.ceil(width / pitch)), 1)
    lines = []
    z = zmax - min(stepdown, max(zmax - zmin, 0.5))
    for i in range(n):
        if longX:
            y = ymin + i * pitch
            p1, p2 = ((xmin, y), (xmax, y)) if i % 2 == 0 else ((xmax, y), (xmin, y))
        else:
            x = xmin + i * pitch
            p1, p2 = ((x, ymin), (x, ymax)) if i % 2 == 0 else ((x, ymax), (x, ymin))
        lines.append(f"G0 Z{_fmt(safe_z)}")
        lines.append(f"G0 X{_fmt(p1[0])} Y{_fmt(p1[1])}")
        lines.append(f"G1 Z{_fmt(z)} F{_fmt(feed)}")
        lines.append(f"G1 X{_fmt(p2[0])} Y{_fmt(p2[1])} F{_fmt(feed)}")
    lines.append(f"G0 Z{_fmt(safe_z)}")
    return lines


def gcode_perimeter(bbox_mm, depth, stepdown=2.0, feed=2200.0, safe_z=5.0):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox_mm
    target_z = zmax - max(depth, 0.0)
    z_steps = []
    z_now = zmax
    while z_now - stepdown > target_z:
        z_now -= stepdown
        z_steps.append(z_now)
    z_steps.append(target_z)

    lines = []
    path = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    for z in z_steps:
        lines.append(f"G0 Z{_fmt(safe_z)}")
        lines.append(f"G0 X{_fmt(path[0][0])} Y{_fmt(path[0][1])}")
        lines.append(f"G1 Z{_fmt(z)} F{_fmt(feed)}")
        for (x, y) in path[1:]:
            lines.append(f"G1 X{_fmt(x)} Y{_fmt(y)} F{_fmt(feed)}")
    lines.append(f"G0 Z{_fmt(safe_z)}")
    return lines


def gcode_drilling(holes_with_centers, top_z, safe_z=5.0, peck=False, feed=600.0, depth_override=None):
    lines = []
    for h in holes_with_centers:
        cx, cy, cz = h.get("center_mm", (None, None, None))
        if cx is None:
            continue
        depth = depth_override if depth_override is not None else h.get("depth_mm", 0.0)
        final_z = top_z - max(depth, 0.0)
        lines.append(f"G0 Z{_fmt(safe_z)}")
        lines.append(f"G0 X{_fmt(cx)} Y{_fmt(cy)}")
        if peck:
            mid = (top_z + final_z) * 0.5
            lines.append(f"G1 Z{_fmt(mid)} F{_fmt(feed)}")
            lines.append(f"G0 Z{_fmt(safe_z)}")
            lines.append(f"G0 X{_fmt(cx)} Y{_fmt(cy)}")
            lines.append(f"G1 Z{_fmt(final_z)} F{_fmt(feed)}")
            lines.append(f"G0 Z{_fmt(safe_z)}")
        else:
            lines.append(f"G1 Z{_fmt(final_z)} F{_fmt(feed)}")
            lines.append(f"G0 Z{_fmt(safe_z)}")
    return lines

# Parse G-code to 3D line segments ------------------------------------------------
_CMD = re.compile(r'([GMTFXYZS])\s*(-?\d+(?:\.\d+)?)', re.I)

def _parse_gcode_to_lines(gcode_lines):
    x = y = z = None
    last = None
    rapid = True
    segments = []
    for ln in gcode_lines:
        tokens = {m.group(1).upper(): float(m.group(2)) for m in _CMD.finditer(ln)}
        if 'G' in tokens:
            g = int(tokens['G'])
            if g == 0:
                rapid = True
            elif g == 1:
                rapid = False
        nx, ny, nz = x, y, z
        if 'X' in tokens: nx = tokens['X']
        if 'Y' in tokens: ny = tokens['Y']
        if 'Z' in tokens: nz = tokens['Z']
        if nx is not None and ny is not None and nz is not None:
            if last is not None and (nx != x or ny != y or nz != z):
                segments.append((x, y, z, nx, ny, nz, rapid))
            last = (nx, ny, nz)
        x, y, z = nx, ny, nz
    return segments


def plot_gcode_segments(segments, bbox_mm=None):
    data = []
    for is_rapid in (False, True):
        xs, ys, zs = [], [], []
        for (x0, y0, z0, x1, y1, z1, rapid) in segments:
            if rapid != is_rapid:
                continue
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            zs += [z0, z1, None]
        if xs and HAVE_PLOTLY:
            data.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(width=3, color='#ff7f0e' if is_rapid else '#1f77b4'),
                name='Rapid' if is_rapid else 'Cut'
            ))
    if bbox_mm and HAVE_PLOTLY:
        xmin, ymin, zmin, xmax, ymax, zmax = bbox_mm
        bx = [xmin,xmax,xmax,xmin,xmin, xmin,xmax,xmax,xmin,xmin, xmin,xmin,xmax,xmax,xmax,xmax]
        by = [ymin,ymin,ymax,ymax,ymin, ymin,ymin,ymax,ymax,ymin, ymin,ymax,ymax,ymin,ymin,ymax]
        bz = [zmin,zmin,zmin,zmin,zmin, zmax,zmax,zmax,zmax,zmax, zmin,zmax,zmax,zmax,zmin,zmin]
        data.append(go.Scatter3d(x=bx, y=by, z=bz, mode='lines',
                                 line=dict(width=2, color='gray'), name='BBox'))
    if HAVE_PLOTLY:
        fig = go.Figure(data=data)
        fig.update_layout(scene=dict(aspectmode='data'), height=500, margin=dict(l=0,r=0,t=0,b=0))
        return fig
    return None

# ==============================================================================
# Setup breakdown helpers (defined BEFORE UI)
# ==============================================================================

def _fmt_hms(seconds: float) -> str:
    s = int(round(max(seconds, 0)))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"


def _closest_principal_axis(vec, tol_deg=12.0):
    if not vec:
        return "Unknown"
    axes = {"X+": (1,0,0), "X-": (-1,0,0), "Y+": (0,1,0), "Y-": (0,-1,0), "Z+": (0,0,1), "Z-": (0,0,-1)}
    best_name, best_ang = None, 999.0
    for name, a in axes.items():
        ang = _angle_deg(vec, a)
        if ang < best_ang:
            best_name, best_ang = name, ang
    return best_name if best_ang <= tol_deg else "Tilted"


def _setup_display_name(axis_name: str) -> str:
    return {
        "Z+": "Top (Z+)", "Z-": "Bottom (Z-)",
        "X+": "Side (X+)", "X-": "Side (X-)",
        "Y+": "Side (Y+)", "Y-": "Side (Y-)",
        "Tilted": "Tilted", "Unknown": "Unknown"
    }.get(axis_name, axis_name)


def _hole_is_through_along_axis(hole: dict, bbox_mm, axis_name: str, tol=0.9) -> bool:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox_mm
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    depth = float(hole.get("depth_mm", 0.0))
    axis_thickness = {"X+": dx, "X-": dx, "Y+": dy, "Y-": dy, "Z+": dz, "Z-": dz}.get(axis_name, dz)
    return depth >= tol * max(axis_thickness, 1e-6)


def _aggregate_milling_time(machine: Machine, mill_df: pd.DataFrame) -> dict:
    face_s = profile_s = toolchange_s = 0.0
    df = mill_df if isinstance(mill_df, pd.DataFrame) else pd.DataFrame()
    for _, row in df.iterrows():
        try:
            op = str(row.get('operation', '')).lower()
            t = mill_time(
                machine,
                float(row.get('cut_length_mm', 0.0)),
                float(row.get('cut_feed_mm_min', 0.0)),
                float(row.get('rapid_length_mm', 0.0)),
                str(row.get('rapid_axis', 'xy')),
                int(row.get('toolchanges', 0)),
                float(row.get('scale_factor', 1.0))
            )
            if "face" in op:
                face_s += t
            elif "perimeter" in op or "profile" in op:
                profile_s += t
            else:
                profile_s += t
            toolchange_s += int(row.get('toolchanges', 0)) * float(machine.tool_change_s)
        except Exception:
            pass
    return {"FACE": face_s, "PROFILE": profile_s, "TOOL CHANGES": toolchange_s}


def _aggregate_drilling_by_setup(machine: Machine, holes_xy: list, bbox_mm, feed_small=500.0, feed_large=800.0):
    setups_drill = {}
    for h in holes_xy or []:
        axis = _closest_principal_axis(h.get("axis_dir"))
        setup_name = _setup_display_name(axis)
        is_through = _hole_is_through_along_axis(h, bbox_mm, axis)
        d = float(h.get("diameter_mm", 0.0))
        feed = feed_small if d < 5.0 else feed_large
        t_one = trapezoidal_time(2.0, machine.rapid_z, machine.accel) \
                + trapezoidal_time(float(h.get("depth_mm", 0.0)), min(feed, machine.max_feed_z), machine.accel) \
                + trapezoidal_time(2.0, machine.rapid_z, machine.accel)
        t_one *= 1.05
        bucket = 'THROUGH HOLES' if is_through else 'BLIND HOLES'
        setups_drill.setdefault(setup_name, {'BLIND HOLES': 0.0, 'THROUGH HOLES': 0.0})
        setups_drill[setup_name][bucket] += t_one
    return setups_drill


def _sum_cost(seconds: float, rate_per_hr: float) -> float:
    return float(seconds) / 3600.0 * float(rate_per_hr)


def build_setup_breakdown(machine: Machine,
                          mill_df: pd.DataFrame,
                          hole_info: dict,
                          bbox_mm,
                          machine_rate_per_hour: float,
                          setup_charge_sgd: float = 30.0,
                          max_setups: int = 2):
    from collections import OrderedDict
    mill_df = mill_df if isinstance(mill_df, pd.DataFrame) else pd.DataFrame()
    setups = OrderedDict()
    milling = _aggregate_milling_time(machine, mill_df)
    top_name = _setup_display_name("Z+")
    setups[top_name] = {'items': [], 'machining_time_s': 0.0, 'setup_charge_sgd': setup_charge_sgd, 'total_cost_sgd': 0.0}
    for label in ("TOOL CHANGES", "FACE", "PROFILE"):
        t = milling.get(label, 0.0)
        if t > 0:
            c = _sum_cost(t, machine_rate_per_hour)
            setups[top_name]['items'].append({'label': label, 'time_s': t, 'cost_sgd': c})
            setups[top_name]['machining_time_s'] += t
            setups[top_name]['total_cost_sgd']   += c
    holes_xy = (hole_info or {}).get("holes_xy", [])
    drill_grouped = _aggregate_drilling_by_setup(machine, holes_xy, bbox_mm)
    for setup_name, buckets in drill_grouped.items():
        if setup_name not in setups:
            setups[setup_name] = {'items': [], 'machining_time_s': 0.0, 'setup_charge_sgd': setup_charge_sgd, 'total_cost_sgd': 0.0}
        for label in ("BLIND HOLES", "THROUGH HOLES"):
            t = buckets.get(label, 0.0)
            if t > 0:
                c = _sum_cost(t, machine_rate_per_hour)
                setups[setup_name]['items'].append({'label': label, 'time_s': t, 'cost_sgd': c})
                setups[setup_name]['machining_time_s'] += t
                setups[setup_name]['total_cost_sgd']   += c
    for s in setups.values():
        s['total_cost_sgd'] += float(setup_charge_sgd)
    if len(setups) > max_setups:
        first_key = next(iter(setups))
        merged = {'items': [], 'machining_time_s': 0.0, 'setup_charge_sgd': setup_charge_sgd, 'total_cost_sgd': 0.0}
        for k in list(setups.keys())[1:]:
            if k == first_key:
                continue
            for it in setups[k]['items']:
                merged['items'].append(it)
                merged['machining_time_s'] += it['time_s']
                merged['total_cost_sgd']   += it['cost_sgd']
            merged['total_cost_sgd'] += setup_charge_sgd
            del setups[k]
        setups["Secondary (indexed)"] = merged
    return setups

# ==============================================================================
# Streamlit UI
# ==============================================================================

st.set_page_config(page_title='Estimator Streamlit 2', layout='wide')
st.title('⚙️ Machining Cost Estimator — STEP-driven')
st.caption('Auto-ops from STEP, auto-costing, setup breakdown, and G-code preview (beta).')

with st.sidebar:
    st.markdown("### Backends & Libraries")
    st.write(f"pythonocc-core available: **{HAVE_OCC}**")
    st.write(f"CadQuery available: **{HAVE_CQ}**")
    st.write(f"Plotly available: **{HAVE_PLOTLY}**")
    st.caption("App prefers pythonocc-core; falls back to CadQuery for properties.")

st.header('Import STEP (.step/.stp) to extract properties')
uploaded_step = st.file_uploader('Upload STEP file', type=['step', 'stp'])
step_props = None
step_tmp_path = None

if uploaded_step is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.stp') as tmp:
        tmp.write(uploaded_step.read())
        step_tmp_path = tmp.name
    st.session_state["step_tmp_path"] = step_tmp_path
    step_props = read_step_properties(step_tmp_path)
    if step_props is None:
        if not HAVE_OCC and not HAVE_CQ:
            st.error('Neither **pythonocc-core** nor **CadQuery** is available.\nInstall: `pip install pythonocc-core` (recommended) or `pip install cadquery`.')
        else:
            st.error('Failed to parse STEP file. If complex, try the other backend or a simplified file.')
    else:
        vol_mm3, area_mm2, (xmin, ymin, zmin, xmax, ymax, zmax), units, backend = step_props
        st.success(
            f'Backend: **{backend}**  \n'
            f'Units (display): {units}  \n'
            f'Volume: {vol_mm3:,.0f} mm³  \n'
            f'Surface area: {area_mm2:,.0f} mm²'
        )
        st.info(
            f'Bounding box (mm): X [{xmin:.2f}, {xmax:.2f}]  '\
            f'Y [{ymin:.2f}, {ymax:.2f}]  '\
            f'Z [{zmin:.2f}, {zmax:.2f}]'
        )
        st.subheader('Material mass helper (from STEP volume)')
        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            density_g_cm3 = st.number_input(
                'Density (g/cm³)', value=2.70, min_value=0.0,
                help='e.g., 6061-T6 Aluminum ≈ 2.70 g/cm³; SS304 ≈ 8.00 g/cm³'
            )
        with colm2:
            material_cost_per_kg = st.number_input('Material cost (SGD/kg)', value=12.0, min_value=0.0)
        with colm3:
            use_for_am = st.checkbox('Use computed mass in AM section', value=False)
        mass_g = (vol_mm3 / 1000.0) * density_g_cm3
        st.write(f'Estimated mass: **{mass_g:,.1f} g**')
        st.write(f'Estimated material cost per part: **{(mass_g/1000.0)*material_cost_per_kg:,.2f} SGD**')
        st.session_state['step_mass_g'] = mass_g if use_for_am else st.session_state.get('step_mass_g', None)
        st.session_state['step_area_mm2'] = area_mm2
        st.session_state['step_bbox_mm'] = (xmin, ymin, zmin, xmax, ymax, zmax)
        # Remember last material cost/kg for costing auto-fill
        st.session_state['last_material_cost_per_kg'] = material_cost_per_kg
        # Auto-fill costing defaults if sync is enabled
        st.session_state.setdefault('autofill_costing_sync', True)
        if st.session_state['autofill_costing_sync']:
            autofill_cnc_costing_defaults(st.session_state.get('cnc_machine_preset', '-- none --'))

# Process picker (CNC only for now)
process_mode = st.radio('Select process method', ['CNC (Subtractive)'], horizontal=True)

if process_mode == 'CNC (Subtractive)':
    with st.sidebar:
        st.header('CNC Machine')
        preset = st.selectbox(
            'Machine preset (optional)',
            ['-- none --', 'Desktop router', 'VMC midsize', 'Haas MiniMill'],
            index=0,
            help='Prefill realistic limits (you can still edit numbers below).'
        )
        st.session_state['cnc_machine_preset'] = preset
        default_vals = {
            "Desktop router": dict(max_feed_xy=12000.0, max_feed_z=6000.0, rapid_xy=18000.0, rapid_z=12000.0, accel=1500.0, tool_change_s=8.0),
            "VMC midsize":   dict(max_feed_xy=24000.0, max_feed_z=15000.0, rapid_xy=30000.0, rapid_z=20000.0, accel=2500.0, tool_change_s=5.0),
            "Haas MiniMill": dict(max_feed_xy=15000.0, max_feed_z=10000.0, rapid_xy=24000.0, rapid_z=20000.0, accel=2000.0, tool_change_s=6.0),
        }
        vals = default_vals.get(preset, {})
        max_feed_xy = st.number_input('Max feed XY (mm/min)', value=float(vals.get("max_feed_xy", 15000.0)), min_value=0.0)
        max_feed_z  = st.number_input('Max feed Z (mm/min)',  value=float(vals.get("max_feed_z", 10000.0)),  min_value=0.0)
        rapid_xy    = st.number_input('Rapid XY (mm/min)',     value=float(vals.get("rapid_xy", 24000.0)),   min_value=0.0)
        rapid_z     = st.number_input('Rapid Z (mm/min)',      value=float(vals.get("rapid_z", 20000.0)),   min_value=0.0)
        accel       = st.number_input('Accel (mm/s²)',         value=float(vals.get("accel", 2000.0)),      min_value=1.0)
        tool_change_s = st.number_input('Tool-change time (s)', value=float(vals.get("tool_change_s", 6.0)), min_value=0.0)
        machine = Machine(max_feed_xy=max_feed_xy, max_feed_z=max_feed_z, rapid_xy=rapid_xy, rapid_z=rapid_z, accel=accel, tool_change_s=tool_change_s)

    st.header('CNC Operations')
    # Editors with session state
    default_mill_cols = {'cut_length_mm': 3200.0,'cut_feed_mm_min': 2400.0,'rapid_length_mm': 600.0,'rapid_axis': 'xy','toolchanges': 1,'scale_factor': 1.12}
    default_drill_cols = {'holes': 12,'depth_mm': 12.0,'drill_feed_mm_min': 600.0,'approach_mm': 2.0,'retract_mm': 2.0,'scale_factor': 1.05}
    mill_df_state = st.session_state.get('mill_df', pd.DataFrame([default_mill_cols]))
    drill_df_state = st.session_state.get('drill_df', pd.DataFrame([default_drill_cols]))

    # Auto-generate section (optional)
    autogen_enabled = (st.session_state.get("step_tmp_path")) and st.session_state.get('step_bbox_mm') is not None
    with st.expander("Auto-generate machining steps from STEP (beta)"):
        st.write("Creates **Facing**, **Perimeter+Pockets**, and **Drilling** rows from the STEP geometry. You can edit them afterwards.")
        colg1, colg2, colg3, colg4 = st.columns(4)
        with colg1:
            rough_tool_d = st.number_input("Facing tool ⌀ (mm)", value=10.0, min_value=0.1)
        with colg2:
            profile_tool_d = st.number_input("Profile tool ⌀ (mm)", value=8.0, min_value=0.1)
        with colg3:
            rough_feed = st.number_input("Facing feed (mm/min)", value=2400.0, min_value=0.0)
        with colg4:
            profile_feed = st.number_input("Profile feed (mm/min)", value=2200.0, min_value=0.0)
        sync_on_change = st.checkbox("🔁 Sync tables when STEP changes", value=True)

        if autogen_enabled:
            step_path_use = st.session_state.get("step_tmp_path")
            fingerprint = hashlib.sha256(open(step_path_use,'rb').read()).hexdigest() if step_path_use and os.path.exists(step_path_use) else None
            if sync_on_change and (st.session_state.get("step_fingerprint") != fingerprint):
                gen_mill_df, gen_drill_df, hole_info = autogen_ops_from_step(
                    file_path=step_path_use,
                    bbox_mm=st.session_state['step_bbox_mm'],
                    machine=machine,
                    rough_tool_d_mm=rough_tool_d,
                    profile_tool_d_mm=profile_tool_d,
                    rough_feed_mm_min=rough_feed,
                    profile_feed_mm_min=profile_feed
                )
                st.session_state['mill_df'] = gen_mill_df
                st.session_state['drill_df'] = gen_drill_df
                st.session_state['hole_info'] = hole_info
                st.session_state['step_fingerprint'] = fingerprint
                _st_rerun()

        if st.button("Generate once", disabled=not autogen_enabled):
            if not HAVE_OCC:
                st.warning("Auto-generation needs pythonocc-core. Install it with `pip install pythonocc-core`.")
            else:
                step_path_use = st.session_state.get("step_tmp_path")
                gen_mill_df, gen_drill_df, hole_info = autogen_ops_from_step(
                    file_path=step_path_use,
                    bbox_mm=st.session_state['step_bbox_mm'],
                    machine=machine,
                    rough_tool_d_mm=rough_tool_d,
                    profile_tool_d_mm=profile_tool_d,
                    rough_feed_mm_min=rough_feed,
                    profile_feed_mm_min=profile_feed
                )
                st.session_state['mill_df'] = gen_mill_df
                st.session_state['drill_df'] = gen_drill_df
                st.session_state['hole_info'] = hole_info
                _st_rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Milling ops')
        mill_df = st.data_editor(
            st.session_state.get('mill_df', mill_df_state),
            num_rows='dynamic', use_container_width=True, key='mill_editor'
        )
        st.session_state['mill_df'] = mill_df
    with col2:
        st.subheader('Drilling ops')
        drill_df = st.data_editor(
            st.session_state.get('drill_df', drill_df_state),
            num_rows='dynamic', use_container_width=True, key='drill_editor'
        )
        st.session_state['drill_df'] = drill_df

    st.divider()
    st.header('CNC Costing, Batch & Lead-time')
    col_sync_a, col_sync_b = st.columns([1,1])
    with col_sync_a:
        sync_costing = st.checkbox("🔁 Keep costing in sync with STEP", value=st.session_state.get('autofill_costing_sync', True))
        st.session_state['autofill_costing_sync'] = sync_costing
    with col_sync_b:
        if st.button("Apply STEP-based costing defaults now"):
            autofill_cnc_costing_defaults(st.session_state.get('cnc_machine_preset', '-- none --'))
            _st_rerun()

    colA, colB, colC, colD = st.columns(4)
    with colA:
        setup_min = st.number_input('Setup (min)', value=float(st.session_state.get('cost_setup_min_default', 30.0)), min_value=0.0)
        batch_qty = st.number_input('Batch quantity', value=int(st.session_state.get('cost_batch_qty_default', 10)), min_value=1, step=1)
    with colB:
        machine_rate_per_hour = st.number_input('Machine rate (SGD/hr)', value=float(st.session_state.get('cost_machine_rate_default', 80.0)), min_value=0.0)
        labor_rate_per_hour = st.number_input('Labor rate (SGD/hr)', value=float(st.session_state.get('cost_labor_rate_default', 35.0)), min_value=0.0)
    with colC:
        material_cost_per_part = st.number_input('Material per part (SGD)', value=float(st.session_state.get('cost_material_per_part_default', 12.0)), min_value=0.0)
        tooling_cost_per_part = st.number_input('Tooling per part (SGD)', value=float(st.session_state.get('cost_tooling_per_part_default', 2.5)), min_value=0.0)
    with colD:
        overhead_pct = st.number_input('Overhead %', value=float(st.session_state.get('cost_overhead_pct_default', 15.0)), min_value=0.0, max_value=100.0) / 100.0
        margin_pct = st.number_input('Margin %', value=float(st.session_state.get('cost_margin_pct_default', 10.0)), min_value=0.0, max_value=100.0) / 100.0

    colE, colF = st.columns(2)
    with colE:
        shift_hours_per_day = st.number_input('Shift hours / day', value=float(st.session_state.get('cost_shift_hours_default', 8.0)), min_value=1.0)
    with colF:
        queue_buffer_days = st.number_input('Queue/Wait allowance (days)', value=float(st.session_state.get('cost_queue_days_default', 1.0)), min_value=0.0)

    if st.button('Calculate CNC time & cost'):
        total_seconds = 0.0
        # Milling
        for _, row in st.session_state['mill_df'].iterrows():
            try:
                total_seconds += mill_time(
                    machine,
                    float(row.get('cut_length_mm', 0.0)),
                    float(row.get('cut_feed_mm_min', 0.0)),
                    float(row.get('rapid_length_mm', 0.0)),
                    str(row.get('rapid_axis', 'xy')),
                    int(row.get('toolchanges', 0)),
                    float(row.get('scale_factor', 1.0))
                )
            except Exception as e:
                st.warning(f'Skipped a milling row due to error: {e}')
        # Drilling
        for _, row in st.session_state['drill_df'].iterrows():
            try:
                total_seconds += drill_time(
                    machine,
                    int(row.get('holes', 0)),
                    float(row.get('depth_mm', 0.0)),
                    float(row.get('drill_feed_mm_min', 0.0)),
                    float(row.get('approach_mm', 0.0)),
                    float(row.get('retract_mm', 0.0)),
                    float(row.get('scale_factor', 1.0))
                )
            except Exception as e:
                st.warning(f'Skipped a drilling row due to error: {e}')
        cycle_time_min = total_seconds / 60.0
        cb = cost_breakdown(
            cycle_time_min, setup_min, int(batch_qty),
            machine_rate_per_hour, labor_rate_per_hour,
            material_cost_per_part, tooling_cost_per_part,
            overhead_pct, margin_pct
        )
        st.success(f'Estimated CNC cycle time: **{cycle_time_min:.2f} min** per part')
        # Lead-time proxy
        total_runtime_min = setup_min + batch_qty * cycle_time_min
        total_runtime_hr = total_runtime_min / 60.0
        runtime_days = total_runtime_hr / max(shift_hours_per_day, 0.001)
        lead_time_days = queue_buffer_days + runtime_days
        st.info(
            f'**Total runtime** (setup + batch): **{total_runtime_min:.1f} min** '
            f'(~{total_runtime_hr:.2f} hr) for {batch_qty} parts  \n'
            f'**Lead-time proxy**: ~**{lead_time_days:.2f} days** '
            f'(assuming {shift_hours_per_day:g} hr/day + {queue_buffer_days:g} day buffer)'
        )
        breakdown_df = pd.DataFrame([
            {'Category': 'Machine time', 'SGD': cb['machine_cost']},
            {'Category': 'Labor (setup amortized)', 'SGD': cb['labor_cost']},
            {'Category': 'Material', 'SGD': cb['material_cost']},
            {'Category': 'Tooling', 'SGD': cb['tooling_cost']},
            {'Category': 'Overhead', 'SGD': cb['overhead']},
            {'Category': 'Price per part', 'SGD': cb['price_per_part']},
        ])
        st.subheader('CNC Cost breakdown (SGD)')
        st.dataframe(breakdown_df, use_container_width=True)
        st.download_button('Download CNC breakdown CSV', data=breakdown_df.to_csv(index=False).encode('utf-8'), file_name='cnc_breakdown.csv', mime='text/csv')
 
        # --- Final amount including setup charges (batch) ---
        total_setup_charges = 0.0
        setup_charge_for_final = float(st.session_state.get('setup_charge_per_setup', 30.0))
        if st.session_state.get('step_bbox_mm') is not None:
            try:
                setups_for_final = build_setup_breakdown(
                    machine=machine,
                    mill_df=st.session_state.get('mill_df', pd.DataFrame()),
                    hole_info=st.session_state.get('hole_info', {}),
                    bbox_mm=st.session_state['step_bbox_mm'],
                    machine_rate_per_hour=float(machine_rate_per_hour),
                    setup_charge_sgd=setup_charge_for_final,
                    max_setups=2
                )
                total_setup_charges = sum(float(s.get('setup_charge_sgd', 0.0)) for s in setups_for_final.values())
            except Exception as e:
                st.warning(f"Could not compute setup charges for final amount: {e}")
                total_setup_charges = 0.0
        try:
            bq = int(batch_qty)
        except NameError:
            bq = int(st.session_state.get('cost_batch_qty_default', 10))
        final_amount_sgd = float(cb['price_per_part']) * bq + float(total_setup_charges)

        st.subheader('Final amount (incl. setup charges)')
        st.success(f"Final amount for batch: **${final_amount_sgd:.2f}**")
        st.caption(f"Price per part (incl. overhead & margin): ${cb['price_per_part']:.2f} × {bq} parts + setup charges ${total_setup_charges:.2f}")
        st.caption(f"Lead-time proxy: ~{lead_time_days:.2f} days")

        # Optional: order summary CSV
        try:
            summary_rows = [
                {'Metric': 'Price per part (incl. OH & margin)', 'Value': round(float(cb['price_per_part']), 2)},
                {'Metric': 'Batch quantity', 'Value': int(bq)},
                {'Metric': 'Setup charges (total)', 'Value': round(float(total_setup_charges), 2)},
                {'Metric': 'Final amount (batch incl. setup)', 'Value': round(float(final_amount_sgd), 2)},
                {'Metric': 'Lead-time (days, proxy)', 'Value': round(float(lead_time_days), 2)},
            ]
            order_df = pd.DataFrame(summary_rows)
            st.download_button('Download order summary CSV', data=order_df.to_csv(index=False).encode('utf-8'), file_name='order_summary.csv', mime='text/csv')
        except Exception as _e:
            pass


    # ---- Setup breakdown (beta) ----
    st.divider()
    st.subheader("Setup breakdown (beta)")
    col_sb_a, col_sb_b = st.columns([1,1])
    with col_sb_a:
        show_breakdown = st.checkbox("Show per-setup breakdown", value=True)
    with col_sb_b:
        setup_charge_sgd = st.number_input("Setup charge per setup (SGD)", value=30.0, min_value=0.0, key="setup_charge_per_setup")

    if show_breakdown and st.session_state.get('step_bbox_mm') is not None:
        try:
            setups = build_setup_breakdown(
                machine=machine,
                mill_df=st.session_state.get('mill_df', pd.DataFrame()),
                hole_info=st.session_state.get('hole_info', {}),
                bbox_mm=st.session_state['step_bbox_mm'],
                machine_rate_per_hour=float(st.session_state.get('cost_machine_rate_default', 80.0)),
                setup_charge_sgd=setup_charge_sgd,
                max_setups=2
            )
        except Exception as e:
            st.error(f"Setup breakdown failed: {type(e).__name__}: {e}")
            st.stop()

        for idx2, (setup_name, sdat) in enumerate(setups.items(), start=1):
            st.markdown("---")
            top_cols = st.columns([4,2,2])
            with top_cols[0]:
                st.markdown(f"**Setup {idx2}**  \n${setup_charge_sgd:.2f} Setup Charge  \n_{setup_name}_")
            with top_cols[1]:
                st.markdown(f"**Machining Time**  \n{_fmt_hms(sdat['machining_time_s'])}")
            with top_cols[2]:
                st.markdown(f"**Total Cost**  \n${sdat['total_cost_sgd']:.2f}")
            for it in sdat['items']:
                with st.expander(it['label']):
                    st.write(f"Time: **{_fmt_hms(it['time_s'])}**")
                    st.write(f"Cost: **${it['cost_sgd']:.2f}**")
        # CSV download for setups
        if len(setups):
            rows = []
            for setup_name, sdat in setups.items():
                for it in sdat['items']:
                    rows.append({"Setup": setup_name, "Operation": it['label'], "Time_s": round(it['time_s'], 3), "Cost_SGD": round(it['cost_sgd'], 2)})
                rows.append({"Setup": setup_name, "Operation": "SETUP CHARGE", "Time_s": 0.0, "Cost_SGD": round(setup_charge_sgd, 2)})
            df_dl = pd.DataFrame(rows)
            st.download_button("Download setup breakdown CSV", data=df_dl.to_csv(index=False).encode("utf-8"), file_name="setup_breakdown.csv", mime="text/csv")

    # ---- G-code preview (beta) ----
    st.divider()
    st.subheader("G-code preview (beta)")
    if not HAVE_PLOTLY:
        st.caption("Install Plotly to enable 3D toolpath preview: `pip install plotly`.")

    colg1, colg2, colg3, colg4 = st.columns(4)
    with colg1:
        gc_tool_d = st.number_input("Tool ⌀ for facing/profile (mm)", value=8.0, min_value=0.1)
    with colg2:
        gc_stepover = st.number_input("Stepover ratio", value=0.6, min_value=0.05, max_value=0.95)
    with colg3:
        gc_stepdown = st.number_input("Stepdown (mm)", value=1.5, min_value=0.05)
    with colg4:
        gc_safe_z = st.number_input("Safe Z (mm)", value=5.0, min_value=1.0)

    colg5, colg6, colg7 = st.columns(3)
    with colg5:
        gc_feed = st.number_input("Feed (mm/min)", value=2200.0, min_value=0.0)
    with colg6:
        gc_spindle = st.number_input("Spindle (RPM)", value=8000, min_value=0, step=100)
    with colg7:
        gc_peck = st.checkbox("Peck drill preview", value=False)

    gen_gc = st.button("Generate preview G-code")
    if gen_gc and st.session_state.get('step_bbox_mm') is not None:
        bbox = st.session_state['step_bbox_mm']
        ztop = bbox[5]
        zthk = max(bbox[5] - bbox[2], 0.0)
        g = []
        g += gcode_header(safe_z=gc_safe_z, spindle=gc_spindle)
        g += gcode_facing(bbox, stepdown=min(gc_stepdown, 0.5), stepover=gc_stepover, tool_d=gc_tool_d, feed=gc_feed, safe_z=gc_safe_z)
        g += gcode_perimeter(bbox, depth=zthk, stepdown=gc_stepdown, feed=gc_feed, safe_z=gc_safe_z)
        holes_xy = (st.session_state.get('hole_info') or {}).get("holes_xy", [])
        if holes_xy:
            g += gcode_drilling(holes_xy, top_z=ztop, safe_z=gc_safe_z, peck=gc_peck, feed=max(600.0, gc_feed*0.4))
        g += gcode_footer()
        if HAVE_PLOTLY:
            segs = _parse_gcode_to_lines(g)
            fig = plot_gcode_segments(segs, bbox_mm=bbox)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        st.text_area("Preview G-code", value="\n".join(g), height=240)
        st.download_button("Download preview G-code", data="\n".join(g).encode("utf-8"), file_name="preview.nc", mime="text/plain")
