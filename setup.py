#!/usr/bin/env python
"""Legacy compatibility shim.

All package metadata now lives in pyproject.toml. This file only exists so
that older workflows such as ``python setup.py develop`` keep working;
prefer ``pip install -e .`` instead.
"""
from setuptools import setup

setup()
