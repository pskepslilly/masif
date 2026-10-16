import os

"""
predict_iface.py: Load a pretrained MaSIF-site model and predict per-vertex
interface scores for a single surface mesh (.ply file).
Released under an Apache License 2.0
"""


def predict_iface(
    ply_path,
    model_dir,
    max_distance=9.0,
    max_shape_size=100,
    n_conv_layers=3,
    feat_mask=(1.0, 1.0, 1.0, 1.0, 1.0),
    idx_gpu="/cpu:0",
):
    """
    Predict the MaSIF-site interface score for every vertex of the mesh in
    ply_path, using the pretrained model checkpoint in model_dir (a
    directory containing model.index/model.meta/model.data-00000-of-00001).

    ply_path must already have the vertex_nx/ny/nz, vertex_charge,
    vertex_hbond and vertex_hphob attributes MaSIF-site's features are
    computed from (i.e. the mesh produced by save_ply with normals, charges,
    hbond and hphob all populated).

    Returns a 1-D array of length n_vertices, in the same vertex order as
    the mesh, with scores in [0, 1].
    """
    from masif_modules.read_data_from_surface import read_data_from_surface
    from masif_modules.MaSIF_site import MaSIF_site
    from masif_modules.train_masif_site import run_masif_site, mask_input_feat

    params = {
        "max_distance": max_distance,
        "max_shape_size": max_shape_size,
    }
    input_feat, rho, theta, mask, neigh_indices, _, _ = read_data_from_surface(
        ply_path, params
    )
    input_feat = mask_input_feat(input_feat, feat_mask)

    # These hyperparameters (n_thetas/n_rhos/n_rotations) are fixed by the
    # pretrained checkpoint -- they are hardcoded the same way in
    # masif_site_predict.py, not read from masif_opts.
    learning_obj = MaSIF_site(
        max_distance,
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        idx_gpu=idx_gpu,
        feat_mask=list(feat_mask),
        n_conv_layers=n_conv_layers,
    )
    learning_obj.saver.restore(learning_obj.session, os.path.join(model_dir, "model"))

    scores = run_masif_site(
        params, learning_obj, rho, theta, input_feat, mask, neigh_indices
    )
    return scores[0]
