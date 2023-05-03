from setuptools import setup, find_packages

setup(
  name = 'coordinate-descent-attention',
  packages = find_packages(exclude=[]),
  version = '0.0.5',
  license='MIT',
  description = 'Coordinate Descent Attention - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/coodinate-descent-attention',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanism'
  ],
  install_requires=[
    'einops>=0.6.1',
    'torch>=1.6',
    'colt5-attention>=0.5.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
