#!/usr/bin/env python3
"""
wikidump processor
==================

"""

from setuptools import setup, find_packages

setup(
    name='wikidump',
    version='0.0.0',
    author='Unknown Author',
    description='A utility for processing wikidump archive',
    license='MIT',
    long_description='',
    packages=find_packages(),
    zip_safe=True,
    python_requires='>3.4, <4.0.0',
    install_requires=[
        'mwparserfromhell==0.5.2',
        'lxml~=4.3.2',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Data Processing',
        'Topic :: Software Development'
    ]
)
