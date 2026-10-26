"""Air absorption coefficients for X-ray photons.

References
----------
S. M. Seltzer, Radiat. Res. 136, 147 (1993).
J. H. Hubbell, Int. J. Appl. Radiat. Isot. 33, 1269 (1982).
"""



import numpy as np
from scipy.interpolate import interp1d

#energy (kev), absorption coefficient (cm2/g), non-trasmited (cm2/g)
def air_absorption_function():

	list = [			
			[5.0E+00,4.027E+01,3.931E+01],
			[6.0E+00,2.341E+01,2.270E+01],
			[8.0E+00,9.921E+00,9.446E+00],
			[1.0E+01,5.120E+00,4.742E+00],
			[1.5E+01,1.614E+00,1.334E+00],
			[2.0E+01,7.779E-01,5.389E-01],
			[3.0E+01,3.538E-01,1.537E-01],
			[4.0E+01,2.485E-01,6.833E-02],
			[5.0E+01,2.080E-01,4.098E-02],
			[6.0E+01,1.875E-01,3.041E-02],
			[8.0E+01,1.662E-01,2.407E-02],
			[1.0E+02,1.541E-01,2.325E-02],
			[1.5E+02,1.356E-01,2.496E-02],
			[2.0E+02,1.233E-01,2.672E-02],
			[3.0E+02,1.067E-01,2.872E-02],
			[4.0E+02,9.549E-02,2.949E-02]
			]
	
	data = np.array(list)
	
	def aa(energy):
		f1 = interp1d(data[:,0], data[:,1], fill_value = "extrapolate")(energy / 1000.)
		f2 = interp1d(data[:,0], data[:,2], fill_value = "extrapolate")(energy / 1000.)
		return [f1, f2]
	
	return aa
	


