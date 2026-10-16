#!/usr/bin/env python
"""
compute_surface_features.py: Standalone script to compute a MaSIF molecular
surface mesh (triangulated with NanoShaper) and its per-vertex geometric and
chemical features -- including Poisson-Boltzmann electrostatics via APBS and
MaSIF-site interface predictions -- for an arbitrary input PDB file, saving
the result as a .ply file.

Unlike source/data_preparation/01-pdb_extract_and_triangulate.py, this script
does not require the fixed data_preparation/ directory layout -- it takes a
PDB file and an output directory as plain arguments.

Usage:
    python compute_surface_features.py input.pdb output_dir
    python compute_surface_features.py input.pdb output_dir --chains A
    python compute_surface_features.py input.pdb output_dir --no-apbs --no-iface

Required external tools (see README.md "Software prerequisites"):
    NanoShaper, reduce, pdb2pqr, APBS, multivalue.

    If this script's directory has a sibling "venv/bin" (i.e. you're running
    from the venv/ environment set up alongside this repo), its NanoShaper,
    reduce, pdb2pqr30, apbs and multivalue binaries are used automatically.
    Otherwise, pass --nanoshaper-bin/--pdb2pqr-bin/--apbs-bin/--multivalue-bin
    (and put reduce on PATH), or set NANOSHAPER_BIN/PDB2PQR_BIN/APBS_BIN/
    MULTIVALUE_BIN/REDUCE_BIN.

    Electrostatics (pdb2pqr/APBS/multivalue) are computed by default; pass
    --no-apbs to skip them and avoid needing those three binaries.

    MaSIF-site interface prediction is also computed by default, using the
    pretrained checkpoint shipped at
    data/masif_site/nn_models/all_feat_3l/model_data/ (found automatically
    relative to this repo, or pass --iface-model-dir / set
    MASIF_SITE_MODEL_DIR). Requires TensorFlow 1.x and networkx; pass
    --no-iface to skip it and avoid needing those.
"""
import argparse
import os
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENV_BIN = os.path.join(_REPO_ROOT, "venv", "bin")
_DEFAULT_IFACE_MODEL_DIR = os.path.join(
    _REPO_ROOT, "data", "masif_site", "nn_models", "all_feat_3l", "model_data"
)


def _default_bin(name):
    """A bundled venv/bin binary, if this repo has one, else None."""
    candidate = os.path.join(_VENV_BIN, name)
    return candidate if os.path.isfile(candidate) else None


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_pdb", help="Input PDB file.")
    p.add_argument("output_dir", help="Directory to write the .ply (and matching .pdb) to.")
    p.add_argument("--chains", default=None,
                   help="Chain ID(s) to keep, e.g. 'A' or 'AB'. Default: keep all standard-residue chains.")
    p.add_argument("--mesh-res", type=float, default=1.0,
                   help="Mesh regularization resolution (default: 1.0, matches masif_opts). Lower is finer/slower.")
    p.add_argument("--no-hbond", action="store_true", help="Skip the hydrogen-bond potential feature.")
    p.add_argument("--no-hphob", action="store_true", help="Skip the hydrophobicity feature.")
    p.add_argument("--no-apbs", action="store_true",
                   help="Skip Poisson-Boltzmann electrostatics (computed by default via pdb2pqr/APBS/multivalue).")
    p.add_argument("--no-iface", action="store_true",
                   help="Skip MaSIF-site interface prediction (computed by default; requires TensorFlow 1.x and networkx).")
    p.add_argument("--iface-model-dir", default=None,
                   help="Directory with the MaSIF-site model checkpoint (model.index/.meta/.data-...). "
                        "Default: data/masif_site/nn_models/all_feat_3l/model_data/ in this repo.")
    p.add_argument("--nanoshaper-bin", default=None, help="Path to the NanoShaper binary (overrides NANOSHAPER_BIN).")
    p.add_argument("--reduce-bin", default=None, help="Path to reduce (overrides REDUCE_BIN / PATH lookup).")
    p.add_argument("--pdb2pqr-bin", default=None, help="Path to pdb2pqr (only used unless --no-apbs).")
    p.add_argument("--apbs-bin", default=None, help="Path to apbs (only used unless --no-apbs).")
    p.add_argument("--multivalue-bin", default=None, help="Path to multivalue (only used unless --no-apbs).")
    return p.parse_args()


def _resolve_env(args):
    # default_config.global_vars reads these env vars (and hard-exits if
    # missing) at import time, so they must be resolved before any import
    # that transitively pulls it in.
    nanoshaper_bin = args.nanoshaper_bin or os.environ.get("NANOSHAPER_BIN") or _default_bin("NanoShaper")
    if not nanoshaper_bin:
        sys.exit("ERROR: NanoShaper not found. Pass --nanoshaper-bin or set NANOSHAPER_BIN.")
    os.environ["NANOSHAPER_BIN"] = nanoshaper_bin

    reduce_bin = args.reduce_bin or os.environ.get("REDUCE_BIN") or _default_bin("reduce") or shutil.which("reduce")
    if not reduce_bin:
        sys.exit("ERROR: 'reduce' not found. Pass --reduce-bin, set REDUCE_BIN, or put it on PATH.")
    os.environ["REDUCE_BIN"] = reduce_bin
    # input_output/protonate.py invokes the literal command "reduce" via
    # PATH lookup rather than an env var, so make sure it resolves to the
    # same binary we just picked.
    reduce_dir = os.path.dirname(reduce_bin)
    if reduce_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = reduce_dir + os.pathsep + os.environ.get("PATH", "")
    if "REDUCE_HET_DICT" not in os.environ:
        het_dict = os.path.join(os.path.dirname(reduce_dir), "reduce_wwPDB_het_dict.txt")
        if os.path.isfile(het_dict):
            os.environ["REDUCE_HET_DICT"] = het_dict

    pdb2pqr_bin = args.pdb2pqr_bin or os.environ.get("PDB2PQR_BIN") or _default_bin("pdb2pqr30") or _default_bin("pdb2pqr")
    apbs_bin = args.apbs_bin or os.environ.get("APBS_BIN") or _default_bin("apbs")
    multivalue_bin = args.multivalue_bin or os.environ.get("MULTIVALUE_BIN") or _default_bin("multivalue")

    if not args.no_apbs:
        missing = [flag for val, flag in [
            (pdb2pqr_bin, "--pdb2pqr-bin/PDB2PQR_BIN"),
            (apbs_bin, "--apbs-bin/APBS_BIN"),
            (multivalue_bin, "--multivalue-bin/MULTIVALUE_BIN"),
        ] if not val]
        if missing:
            sys.exit("ERROR: electrostatics requested but missing: {}. "
                      "Pass the flag(s) shown, set the env var(s), or pass --no-apbs to skip electrostatics."
                      .format(", ".join(missing)))

    os.environ["PDB2PQR_BIN"] = pdb2pqr_bin or ""
    os.environ["APBS_BIN"] = apbs_bin or ""
    os.environ["MULTIVALUE_BIN"] = multivalue_bin or ""

    iface_model_dir = (args.iface_model_dir or os.environ.get("MASIF_SITE_MODEL_DIR")
                        or (_DEFAULT_IFACE_MODEL_DIR if os.path.isdir(_DEFAULT_IFACE_MODEL_DIR) else None))
    if not args.no_iface and not iface_model_dir:
        sys.exit("ERROR: interface prediction requested but no model checkpoint found. "
                  "Pass --iface-model-dir, set MASIF_SITE_MODEL_DIR, or pass --no-iface to skip it.")
    args.iface_model_dir = iface_model_dir


def _preflight(args):
    nanoshaper_bin = os.environ["NANOSHAPER_BIN"]
    if not (os.path.isfile(nanoshaper_bin) and os.access(nanoshaper_bin, os.X_OK)):
        sys.exit("ERROR: NANOSHAPER_BIN does not point to an executable file: {}".format(nanoshaper_bin))
    reduce_bin = os.environ["REDUCE_BIN"]
    if not (os.path.isfile(reduce_bin) and os.access(reduce_bin, os.X_OK)):
        sys.exit("ERROR: REDUCE_BIN does not point to an executable file: {}".format(reduce_bin))
    if not args.no_apbs:
        for var in ("PDB2PQR_BIN", "APBS_BIN", "MULTIVALUE_BIN"):
            path = os.environ[var]
            if not (os.path.isfile(path) and os.access(path, os.X_OK)):
                sys.exit("ERROR: {} does not point to an executable file: {}".format(var, path))
    if not args.no_iface:
        checkpoint = os.path.join(args.iface_model_dir, "model.index")
        if not os.path.isfile(checkpoint):
            sys.exit("ERROR: no MaSIF-site checkpoint found at {} "
                      "(expected model.index/.meta/.data-...). "
                      "Pass --iface-model-dir, set MASIF_SITE_MODEL_DIR, or pass --no-iface."
                      .format(args.iface_model_dir))
        if args.no_apbs or args.no_hbond or args.no_hphob:
            sys.exit("ERROR: interface prediction needs charge, hbond and hphob features, "
                      "so it's incompatible with --no-apbs/--no-hbond/--no-hphob. "
                      "Pass --no-iface too if you want to skip one of those.")
        try:
            import tensorflow  # noqa: F401
            import networkx  # noqa: F401
        except ImportError as e:
            sys.exit("ERROR: interface prediction requires TensorFlow 1.x and networkx: {}. "
                      "Pass --no-iface to skip it.".format(e))
    if not os.path.isfile(args.input_pdb):
        sys.exit("ERROR: input PDB not found: {}".format(args.input_pdb))


def main():
    args = _parse_args()
    _resolve_env(args)
    _preflight(args)

    # Make the masif "source" directory importable regardless of PYTHONPATH/cwd.
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    import pymesh
    from default_config.masif_opts import masif_opts
    from input_output.extractPDB import extractPDB
    from input_output.protonate import protonate
    from input_output.save_ply import save_ply
    from triangulation.computeSurface import computeSurface
    from triangulation.fixmesh import fix_mesh
    from triangulation.compute_normal import compute_normal
    from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
    from triangulation.computeHydrophobicity import computeHydrophobicity
    if not args.no_apbs:
        from triangulation.computeAPBS import computeAPBS
    if not args.no_iface:
        from masif_site.predict_iface import predict_iface

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = masif_opts["tmp_dir"]
    base_name = os.path.splitext(os.path.basename(args.input_pdb))[0]
    if args.chains:
        base_name += "_" + args.chains

    protonated_pdb = os.path.join(tmp_dir, base_name + "_protonated.pdb")
    print("Protonating {} -> {}".format(args.input_pdb, protonated_pdb))
    protonate(args.input_pdb, protonated_pdb)

    extracted_pdb = os.path.join(tmp_dir, base_name + "_extracted.pdb")
    extractPDB(protonated_pdb, extracted_pdb, args.chains)

    print("Computing surface with NanoShaper...")
    vertices1, faces1, normals1, names1, areas1 = computeSurface(extracted_pdb, protonate=True)

    vertex_hbond = None
    if not args.no_hbond:
        vertex_hbond = computeCharges(extracted_pdb[:-4], vertices1, names1)

    vertex_hphobicity = None
    if not args.no_hphob:
        vertex_hphobicity = computeHydrophobicity(names1)

    print("Regularizing mesh...")
    mesh = pymesh.form_mesh(vertices1, faces1)
    regular_mesh = fix_mesh(mesh, args.mesh_res)

    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

    if vertex_hbond is not None:
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)
    if vertex_hphobicity is not None:
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)

    vertex_charges = None
    if not args.no_apbs:
        print("Computing APBS electrostatics...")
        vertex_charges = computeAPBS(regular_mesh.vertices, extracted_pdb, extracted_pdb[:-4])

    out_ply = os.path.join(args.output_dir, base_name + ".ply")
    save_ply(
        out_ply,
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True,
        hbond=vertex_hbond,
        hphob=vertex_hphobicity,
    )

    if not args.no_iface:
        print("Predicting interface with MaSIF-site...")
        vertex_iface = predict_iface(out_ply, args.iface_model_dir)
        save_ply(
            out_ply,
            regular_mesh.vertices,
            regular_mesh.faces,
            normals=vertex_normal,
            charges=vertex_charges,
            normalize_charges=True,
            hbond=vertex_hbond,
            hphob=vertex_hphobicity,
            iface=vertex_iface,
        )

    out_pdb = os.path.join(args.output_dir, base_name + ".pdb")
    shutil.copy(extracted_pdb, out_pdb)

    for f in (protonated_pdb, extracted_pdb):
        if os.path.exists(f):
            os.remove(f)

    print("Wrote {}".format(out_ply))
    print("Wrote {}".format(out_pdb))


if __name__ == "__main__":
    main()
