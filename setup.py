#from distutils.core import setup
from setuptools import setup

setup(name="MechanicalEngModule",
      version='1.1',
      description='Variables and Dictionaries used for Mechanical Calculations',
      author = 'Victor J. Marrero-Fontanez',
      author_email = 'victor.marrero@gmail.com',
      install_requires = ['numpy'],
      packages=["MechanicalEngModule"],
)