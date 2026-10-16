from Bio.PDB import *
from default_config.chemistry import radii, polarHydrogens

"""
xyzr.py: Read a pdb file and output it in xyzr format for use in NanoShaper.
Released under an Apache License 2.0
"""

def output_pdb_as_xyzr(pdbfilename, xyzrfilename):
    """
        pdbfilename: input pdb filename
        xyzrfilename: output in xyzr format (x y z radius per atom, no name field).

        Returns the list of per-atom name strings, in the same order as the
        lines written to xyzrfilename, in the format
        chain_resid_insertion_resname_atomname_color (the same convention
        output_pdb_as_xyzrn used to embed directly in the xyzrn file).
    """
    parser = PDBParser()
    struct = parser.get_structure(pdbfilename, pdbfilename)
    names = []
    outfile = open(xyzrfilename, "w")
    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        if residue.get_id()[0] != " ":
            continue
        resname = residue.get_resname()
        chain = residue.get_parent().get_id()
        atomtype = name[0]

        color = "Green"
        coords = None
        if atomtype in radii and resname in polarHydrogens:
            if atomtype == "O":
                color = "Red"
            if atomtype == "N":
                color = "Blue"
            if atomtype == "H":
                if name in polarHydrogens[resname]:
                    color = "Blue"  # Polar hydrogens
            coords = "{:.06f} {:.06f} {:.06f}".format(
                atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            )
            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]
            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain, residue.get_id()[1], insertion, resname, name, color
            )
        if coords is not None:
            outfile.write(coords + " " + radii[atomtype] + "\n")
            names.append(full_id)
    outfile.close()
    return names
