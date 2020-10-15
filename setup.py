from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Use Graph Network to perform analysis"

setup(
    name="root_gnn",
    version="1.0.0",
    description="Library for using Graph Nural Networks in HEP analysis",
    long_description=description,
    author="Xiangyang Ju",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "HEP", "analysis", "machine learning"],
    url="https://github.com/xju2/root_gnn",
    packages=find_packages(),
    install_requires=[
        "graph_nets >= 1.1",
        "future",
        "networkx",
        "numpy",
        "scipy",
        "pandas",
        "setuptools",
        "six",
        "matplotlib",
        "sklearn",
        'pyyaml>=5.1',
    ],
    setup_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    scripts=[
        'scripts/train_classifier',
        'scripts/view_training_log',
        'scripts/create_tfrecord',
        'scripts/evaluate_global_classifier',
        'scripts/calculate_metrics',
        'scripts/evaluate_w_qcd_classifier',
        'scripts/evaluate_wtagger',
        'scripts/calculate_wtagger_metrics',
        'scripts/train_decay_simulator',
        'scripts/evaluate_herwig_hadron',
        'scripts/calculate_herwig_metrics'
    ],
)
