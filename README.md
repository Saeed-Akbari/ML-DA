# ML-DA

"Blending machine learning and sequential data assimilation over latent spaces for surrogate modeling of Boussinesq systems"

Authors: Saeed Akbari, Pedram H. Dabaghian, Omer San

Journal: Physica D: Nonlinear Phenomena

# Dataset

All the files required to simulate Rayleigh-Bénard convection are in the FOM folder.
Inputs of the full order model is in yaml files ("FOM/config/"). After specifying desired inputs, "FOM/ns2d_ws_rbc.py" must be executed to simulate flow.
The FOM data must be collected to be used for building the nonintrusive reduced order model.
Details of the numerical schemes are in the paper.

# Building NIROM

Training and testing the NIROM can be done by running "NIROM_DA/POD.py" file. Inputs can be specified in "NIROM_DA/input/".

# DA

The filr "random_sensors_da.py" uses NIROM model to create bunch of imperfect model and use sensors data through DEnKF to simulate the flow.
