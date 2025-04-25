# setup.py
from setuptools import setup, find_packages

setup(
    name="myproj",
    version="0.1.0",
    packages=find_packages(include=["main", "main.*"]),
)
