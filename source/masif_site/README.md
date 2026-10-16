### source/masif_ppi_site
Contains entry functions to for MaSIF-site, including for training and evaluation.

+ *masif_site_train.py*: Entry function to train MaSIF-site
+ *masif_site_predict.py*: Entry function to evaluate a protein using MaSIF-site.
+ *masif_site_label_surface.py*: Program to color a ply file by the MaSIF-site predicted score.
+ *predict_iface.py*: Reusable function to predict per-vertex interface scores for a single .ply file, given a model checkpoint directory. Used by source/data_preparation/compute_surface_features.py.
