from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# Setup necessary paths for compiling realtimestft library
audio_dir_base = "../../../" # we have audio/python/py_fft/pa_tools
CFLAGS = [
    "-I/opt/local/include",
    "-I/usr/include/Python2.7",
    "-I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers/",
    "-I" + audio_dir_base + "include/"
    ]
LDFLAGS = [
    "-L/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A",
    "-lvDSP"
    ]
c_src = audio_dir_base + "src/"

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("stftmanager",
                  ["stftmanager.pyx", c_src + "realtimestft.c"],
                  include_dirs=[np.get_include()],
                  extra_link_args=LDFLAGS,
                  extra_compile_args=CFLAGS)
    ]
)
