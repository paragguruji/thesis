#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
module to setup the repository

:REQUIRES: setuptools


:AUTHOR: Parag Guruji
:ORGANIZATION: Purdue University
:CONTACT: pguruji@purdue.edu
:SINCE: Mon Aug 14 18:36:59 2017
:VERSION: 0.1
"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================
from setuptools import setup, find_packages


# =============================================================================
# PROGRAM METADATA
# =============================================================================
__author__ = 'Parag Guruji'
__contact__ = 'pguruji@purdue.edu'
__copyright__ = 'Copyright (C) 2017 Parag Guruji, Purdue University, USA'
__date__ = 'Mon Aug 14 18:36:59 2017'
__version__ = '0.1'


# =============================================================================
# CLASSES / METHODS
# =============================================================================

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='thesis',
    version='0.0.1',
    description='Experiments in the MS thesis',
    long_description=readme,
    author='Parag Guruji',
    author_email='pguruji@purdue.edu',
    url='https://github.com/paragguruji/thesis',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'data', 'log'))
)
