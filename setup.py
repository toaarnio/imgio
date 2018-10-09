from setuptools import setup, find_packages

setup(name='imgio',
      version='0.4.0',
      description='Easy image reading & writing. Supports PGM/PPM/PNM/PFM/PNG/JPG/INSP/RAW.',
      url='http://github.com/toaarnio/imgio',
      author='Tomi Aarnio',
      author_email='tomi.p.aarnio@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['imread>=0.6.1', 'piexif>=1.0.13', 'numpy'],
      test_suite='imgio',
      zip_safe=False)
