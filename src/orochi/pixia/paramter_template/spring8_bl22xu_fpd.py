import numpy as np

def use_2023a(pix_par):
    pix_par["rambda"] = 0.181050
    pix_par["polar"] = 1.0

    pix_par["pixx"] = 2048
    pix_par["pixy"] = 2048
    pix_par["pix_sizex"] = 0.2
    pix_par["pix_sizey"] = 0.2    
    
    pix_par["x0"] = 1028.9356
    pix_par["y0"] = 1053.3887
    pix_par["tiltx"] = -0.0967
    pix_par["tilty"] = 0.0966
    pix_par["distance"] = 225.7755

    pix_par["betamin"] = -180
    pix_par["betamax"] = 180
    pix_par["ttmin"] = 0.78
    pix_par["ttmax"] = 50.0
    pix_par["ttstep"] = 0.002

    pix_par["thickness"] = 0.001
    pix_par["mur"] = 0.5
    pix_par["outlier"] = 0.2  

    pix_par["edge_x1"] = 20
    pix_par["edge_x2"] = 2028
    pix_par["edge_y1"] = 20
    pix_par["edge_y2"] = 2028
    pix_par["grid_xp"] = 16
    pix_par["grid_yp"] = 2
    pix_par["grid_xw"] = 2
    pix_par["grid_yw"] = 2
    pix_par["beamstop_circle"] = 35

    return pix_par


def use_2023b(pix_par):
    pix_par["rambda"] = 0.18182545918787368
    pix_par["polar"] = 1.0

    pix_par["pixx"] = 2048
    pix_par["pixy"] = 2048
    pix_par["pix_sizex"] = 0.2
    pix_par["pix_sizey"] = 0.2    
    
    pix_par["x0"] = 1028.9279411170846
    pix_par["y0"] = 1053.4707868246737
    pix_par["tiltx"] = -0.08213022646292817
    pix_par["tilty"] = 0.1093476730964918
    pix_par["distance"] = 225.04690095633904

    pix_par["betamin"] = -180
    pix_par["betamax"] = 180
    pix_par["ttmin"] = 0.78
    pix_par["ttmax"] = 50.0
    pix_par["ttstep"] = 0.002

    pix_par["thickness"] = 0.001
    pix_par["mur"] = 0.5
    pix_par["outlier"] = 0.2  

    pix_par["edge_x1"] = 20
    pix_par["edge_x2"] = 2028
    pix_par["edge_y1"] = 20
    pix_par["edge_y2"] = 2028
    pix_par["grid_xp"] = 16
    pix_par["grid_yp"] = 2
    pix_par["grid_xw"] = 2
    pix_par["grid_yw"] = 2
    pix_par["beamstop_circle"] = 35

    return pix_par


def use_2024a(pix_par):
    pix_par["rambda"] = 0.18123949345740054
    pix_par["polar"] = 1.0

    pix_par["pixx"] = 2048
    pix_par["pixy"] = 2048
    pix_par["pix_sizex"] = 0.2
    pix_par["pix_sizey"] = 0.2    
    
    pix_par["x0"] = 1030.0482 #1030.0275628901672
    pix_par["y0"] = 1053.6989 #1053.698864733319
    pix_par["tiltx"] = -0.1151 #-0.10343917244162922
    pix_par["tilty"] = 0.0967 #0.08709873420566316
    pix_par["distance"] = 225.4767 #225.4728688208398

    pix_par["betamin"] = -180
    pix_par["betamax"] = 180
    pix_par["ttmin"] = 0.78
    pix_par["ttmax"] = 50.0
    pix_par["ttstep"] = 0.002

    pix_par["thickness"] = 0.2
    pix_par["mur"] = 1.
    pix_par["outlier"] = 0.2  

    pix_par["edge_x1"] = 20
    pix_par["edge_x2"] = 2028
    pix_par["edge_y1"] = 20
    pix_par["edge_y2"] = 2028
    pix_par["grid_xp"] = 16
    pix_par["grid_yp"] = 2
    pix_par["grid_xw"] = 2
    pix_par["grid_yw"] = 2
    pix_par["beamstop_circle"] = 35

    return pix_par

def use_2024a_new(pix_par):
    pix_par["rambda"] = 0.1815225352995298 #0.18155674074930678 ceo2
    pix_par["polar"] = 1.0

    pix_par["pixx"] = 2048 *2
    pix_par["pixy"] = 2048 *2
    pix_par["pix_sizex"] = 0.2/2
    pix_par["pix_sizey"] = 0.2/2  
    
    pix_par["x0"] = 2060.9515
    pix_par["y0"] = 2108.7611
    pix_par["tiltx"] = 0.0236 
    pix_par["tilty"] = 0.1640 
    pix_par["distance"] = 225.30206605347473 #224.46493383240622

    pix_par["betamin"] = -180
    pix_par["betamax"] = 180
    pix_par["ttmin"] = 0.78
    pix_par["ttmax"] = 50.0
    pix_par["ttstep"] = 0.002

    pix_par["thickness"] = 0.2
    pix_par["mur"] = 1.
    pix_par["outlier"] = 0.2  

    pix_par["edge_x1"] = 20
    pix_par["edge_x2"] = 4056
    pix_par["edge_y1"] = 20
    pix_par["edge_y2"] = 4056
    pix_par["grid_xp"] = 0
    pix_par["grid_yp"] = 0
    pix_par["grid_xw"] = 0
    pix_par["grid_yw"] = 0
    pix_par["beamstop_circle"] = 45
    return pix_par



def use_2024a_730(pix_par):
    pix_par["rambda"] = 0.17982456326566518
    pix_par["polar"] = 1.0

    pix_par["pixx"] = 2048
    pix_par["pixy"] = 2048
    pix_par["pix_sizex"] = 0.2
    pix_par["pix_sizey"] = 0.2    
    
    pix_par["x0"] = 1031.4448550233242
    pix_par["y0"] = 1051.3237793792773
    pix_par["tiltx"] = -0.058885819187681
    pix_par["tilty"] = 0.17272995164104496
    pix_par["distance"] = 655.0802403474885

    pix_par["betamin"] = -180
    pix_par["betamax"] = 180
    pix_par["ttmin"] = 0.78
    pix_par["ttmax"] = 50.0
    pix_par["ttstep"] = 0.002

    pix_par["thickness"] = 0.001
    pix_par["mur"] = 0.5
    pix_par["outlier"] = 0.2  

    pix_par["edge_x1"] = 20
    pix_par["edge_x2"] = 2028
    pix_par["edge_y1"] = 20
    pix_par["edge_y2"] = 2028
    pix_par["grid_xp"] = 16
    pix_par["grid_yp"] = 2
    pix_par["grid_xw"] = 2
    pix_par["grid_yw"] = 2
    pix_par["beamstop_circle"] = 35

    return pix_par