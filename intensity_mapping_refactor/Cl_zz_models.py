import numpy as np

class Cl_zz_model(object):

   def Cl_zz(self, params, cl_range):
       #Should be overwritten for each specific model.
       return None

class Cl_zz_from_templates(Cl_zz_model):
    cl_zz_templates = []
    cl_zz_templates_binned_mixed = []

    def __init__(self, cl_zz_templates, TomographicPair = None):
        self.cl_zz_templates = cl_zz_templates
        #Bin and mix cl_zz_templates with TomographicPair and save in self.cl_zz_templates_binned_mixed.

    def _set_template_multipliers(self, params):
        #Should be overwritten for each model.
        self._params = params
        self.multipliers_z1 = [None]
        self.multipliers_z2 = [None]
        #Each element of the multipliers should be a 2d matrix in redshift space.
        #For multipliers_z1, First index describes redshift kernel in observed redshift-space, useful for continuum emission. 
        #For multipliers_z1, Second index describes z-dependent change in the emitting redshift, for things like redshift-dep. bias or line/continuum intensity.
        #For multipliers_z2, this order should be transposed. Allows use of quick matmul without computing and transpose.

    def multiply_template(multiplier_z1, multiplier_z2, template):
        intermediate = np.matmul(template, multiplier_z2)
        return np.matmul(multiplier1, intermediate)
       
    def multiply_templates(self):
        self.Cl_zz = np.zeros_like(self.cl_zz_templates_binned_mixed[0])
        for multiplier_z1, multiplier_z2, template in zip(self.multipliers_z1, self.multipliers_z2, self.cl_zz_templates_binned_mixed):
            self.Cl_zz += self.multiply_template(multiplier_z1, multiplier_z2, template)

    def Cl_zz(self, params, cl_range):
        if self._params != params:
            self._set_template_multipliers(params)
        self.multiply_templates()
        return self.Cl_zz
