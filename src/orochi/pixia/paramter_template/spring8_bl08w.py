import numpy as np

def use_2018a(pix_par):
    pix_par["rambda"] = 0.107600
    pix_par["polar"] = 1.0

    pix_par["pixx"] = 2048
    pix_par["pixy"] = 2048
    pix_par["pix_sizex"] = 0.2
    pix_par["pix_sizey"] = 0.2    
    
    pix_par["x0"] = 971.6833
    pix_par["y0"] = 79.2709
    pix_par["tiltx"] = 0.6470
    pix_par["tilty"] = 0.3091
    pix_par["distance"] = 225.3693

    pix_par["betamin"] = 70
    pix_par["betamax"] = 110
    pix_par["ttmin"] = 0.78
    pix_par["ttmax"] = 40.0
    pix_par["ttstep"] = 0.002

    pix_par["thickness"] = 0.015
    pix_par["mur"] = 0.
    pix_par["outlier"] = 0.2  

    pix_par["edge_x1"] = 24
    pix_par["edge_x2"] = 2024
    pix_par["edge_y1"] = 24
    pix_par["edge_y2"] = 2024
    pix_par["grid_xp"] = 16
    pix_par["grid_yp"] = 2
    pix_par["grid_xw"] = 2
    pix_par["grid_yw"] = 2
    pix_par["beamstop_circle"] = 30

    return pix_par