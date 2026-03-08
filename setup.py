from setuptools import setup, find_packages

setup(
  name='tmdbot',
  version='0.1',
  #author='...',
  #description='...',
  install_requires=[
      "tmdbv3api",
      "python-telegram-bot",
      "pyyaml",
      "requests",
  ],
  packages=find_packages(),
  entry_points={
    'console_scripts': [
      'tmdbot=tmdbot:main',
      'bookbot=bookbot:main',
    ]
  },
)
