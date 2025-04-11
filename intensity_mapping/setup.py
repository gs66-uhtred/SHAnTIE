from distutils.core import setup
from Cython.Build import cythonize

#setup(name="cl_cov_cy", ext_modules=cythonize('cl_cov_cy.pyx'),)
setup(name="box_car_conv", ext_modules=cythonize('box_car_conv.pyx'),)
#setup(name="box_car2", ext_modules=cythonize('box_car2.pyx'),)
