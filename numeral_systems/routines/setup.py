from distutils.core import setup
from Cython.Build import cythonize

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.cpp'

setup(
    ext_modules = cythonize(["find.py", "get_term_num_matrix.py", "compute_base_n_complexities.py", "compute_P_w_i_variants.py"])
)
