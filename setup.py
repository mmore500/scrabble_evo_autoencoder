#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    'numpy',
    'deap',
    'tqdm'
]

setup_requirements = [
    # TODO(mmore500): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='scrabble_evo_autoencoder',
    version='0.8.0',
    description="Experiments using autoencoders to learn evolvable encodings for scrabble strings.",
    author="Matthew Andres Moreno",
    author_email='mmore500@msu.edu',
    url='https://github.com/mmore500/scrabble_evo_autoencoder',
    packages=find_packages(include=['screvaut_evo', 'screvaut_learn']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='scrabble_evo_autoencoder',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
