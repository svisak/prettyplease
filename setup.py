#!/usr/bin/env python3

from setuptools import setup
import subprocess

cmd = ['git', 'describe', '--tags', '--abbrev=0']
version = subprocess.run(cmd, stdout=subprocess.PIPE)
version = version.stdout.decode('utf-8').strip()

setup(
    name='prettyplease',
    version=version,
    description='Create plots.',
    author='Isak Svensson',
    author_email='isak.svensson@chalmers.se',
    packages=['prettyplease'],
    install_requires=['numpy'],
)
