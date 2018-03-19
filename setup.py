from setuptools import setup, find_packages

setup(name='imgio',
      version='0.3.0',
      description='Easy image reading & writing. Supports PGM/PPM/PFM/PNG/JPG.',
      url='http://github.com/toaarnio/imgio',
      author='Tomi Aarnio',
      author_email='tomi.p.aarnio@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['imread>=0.6.1', 'numpy'],
      test_suite='imgio',
      zip_safe=False)
