from setuptools import setup, find_packages

setup(
    name='inpcdiff',
    version="0.0.1",
    packages=find_packages(include=['diffcsp', 'diffcsp.*']),
)
