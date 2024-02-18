from setuptools import setup

setup(
  name='tmdbot',
  version='0.1',
  #author='...',
  #description='...',
  install_requires=[
      "tmdbv3api",
      "python-telegram-bot",
      "pyyaml"
  ],
  scripts=[
    'tmdbot.py',
  ],
  entry_points={
    'console_scripts': ['tmdbot=tmdbot:main']
  },
)