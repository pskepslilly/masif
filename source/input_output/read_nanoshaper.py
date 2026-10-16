import numpy as np
"""
read_nanoshaper.py: Read a NanoShaper output mesh (NanoShaper built with
Save_Mesh_MSMS_Format=true and Vertex_Atom_Info=true writes {file_root}.vert
and {file_root}.face files in a format very close to MSMS's).
Released under an Apache License 2.0
"""

def read_nanoshaper_mesh(file_root, atom_names):
    # read the surface from the NanoShaper output. NanoShaper outputs two
    # files: {file_root}.vert and {file_root}.face
    #
    # atom_names: list of per-atom name strings (chain_resid_insertion_resname_atomname_color),
    # indexed the same way (1-based) as the atoms in the .xyzr file given to NanoShaper.
    # This is used to recover the per-vertex name string, since NanoShaper's
    # vert file stores a 1-based atom index rather than an embedded name string.

    vertfile = open(file_root + ".vert")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices.
    count = {}
    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    ## Data Structures
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    res_id = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        atom_idx = int(fields[7])
        res_id[vi] = atom_names[atom_idx - 1] if atom_idx > 0 else ""
        count["vertices"] -= 1

    # Read faces.
    facefile = open(file_root + ".face")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of faces.
    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, normalv, res_id
