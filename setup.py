from setuptools import setup, find_packages

setup(name='imgio',
      version='0.4.3',
      description='Easy image reading & writing. Supports PGM/PPM/PNM/PFM/PNG/TIFF/JPG/INSP/RAW.',
      url='http://github.com/toaarnio/imgio',
      author='Tomi Aarnio',
      author_email='tomi.p.aarnio@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['imread>=0.7.1', 'piexif>=1.1.2', 'numpy'],
      test_suite='imgio',
      zip_safe=False)
