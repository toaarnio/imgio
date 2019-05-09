from setuptools import setup, find_packages

import imgio


def read_deps(filename):
    with open(filename) as f:
        deps = f.read().split('\n')
        deps.remove("")
    return deps


setup(name='imgio',
      version=imgio.__version__,
      description='Easy image reading & writing. Supports PGM/PPM/PNM/PFM/PNG/TIFF/JPG/INSP/RAW.',
      url='http://github.com/toaarnio/imgio',
      author='Tomi Aarnio',
      author_email='tomi.p.aarnio@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=read_deps("requirements.txt"),
      test_suite='imgio',
      zip_safe=False)
