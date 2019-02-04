from setuptools import setup, find_packages

setup(name='imgio',
      version='0.4.1',
      description='Easy image reading & writing. Supports PGM/PPM/PNM/PFM/PNG/JPG/INSP/RAW.',
      url='http://github.com/toaarnio/imgio',
      author='Tomi Aarnio',
      author_email='tomi.p.aarnio@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['imread>=0.7.0', 'piexif>=1.1.2', 'numpy'],
      test_suite='imgio',
      zip_safe=False)
