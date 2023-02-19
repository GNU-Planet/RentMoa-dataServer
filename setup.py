# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='richJinju',
    version='0.1.0',
    description='richJinju package',
    long_description=readme,
    author='Dongho Jang',
    author_email='dongho18@gnu.ac.kr',
    url='https://github.com/JangDongHo/rich-jinju-dataServer',
    license=license
)