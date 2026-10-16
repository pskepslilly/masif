import os
import shutil
import sys
from subprocess import Popen, PIPE

from input_output.read_nanoshaper import read_nanoshaper_mesh
from triangulation.xyzr import output_pdb_as_xyzr
from default_config.global_vars import nanoshaper_bin
from default_config.masif_opts import masif_opts
import random

# Calls NanoShaper and returns the vertices of the computed molecular surface.
def computeSurface(pdb_file, protonate=True):
    randnum = random.randint(1, 10000000)
    work_dir = masif_opts["tmp_dir"] + "/nanoshaper_" + str(randnum)
    os.makedirs(work_dir)

    xyzr_file = work_dir + "/mol.xyzr"

    if not protonate:
        print("Error - pdb2xyzrn is deprecated.")
        sys.exit(1)
    atom_names = output_pdb_as_xyzr(pdb_file, xyzr_file)

    config_file = work_dir + "/surfaceConfiguration.prm"
    with open(config_file, "w") as f:
        f.write("Compute_Vertex_Normals = true\n")
        f.write("Save_Mesh_MSMS_Format = true\n")
        f.write("Vertex_Atom_Info = true\n")
        f.write("Grid_scale = 3.0\n")
        f.write("Grid_perfil = 80.0\n")
        f.write("XYZR_FileName = mol.xyzr\n")
        f.write("Build_epsilon_maps = false\n")
        f.write("Build_status_map = false\n")
        f.write("Surface = ses\n")
        f.write("Smooth_Mesh = true\n")
        f.write("Patch_Based_Algorithm = true\n")
        f.write("Analytical_Ray_Vs_Torus_Intersection = true\n")
        f.write("Surface_File_Name = triangulatedSurf\n")
        f.write("Cavity_Detection_Filling = false\n")
        f.write("Probe_Radius = 1.5\n")
        f.write("Accurate_Triangulation = true\n")
        f.write("Triangulation = true\n")

    args = [nanoshaper_bin, "surfaceConfiguration.prm"]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=work_dir)
    stdout, stderr = p2.communicate()
    if p2.returncode != 0:
        shutil.rmtree(work_dir)
        raise RuntimeError(
            "NanoShaper failed with return code {}: {}".format(
                p2.returncode, stderr.decode(errors="replace")
            )
        )

    file_root = work_dir + "/triangulatedSurf"
    vertices, faces, normals, names = read_nanoshaper_mesh(file_root, atom_names)
    areas = {}

    shutil.rmtree(work_dir)
    return vertices, faces, normals, names, areas
