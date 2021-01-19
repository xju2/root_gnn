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
    include_package_data=True,
    install_requires=[
        "graph_nets@ https://github.com/deepmind/graph_nets/tarball/master",
        "future",
        "networkx",
        "numpy",
        "scipy",
        "pandas",
        "tables",
        "setuptools",
        "six",
        "matplotlib",
        "sklearn",
        'pyyaml>=5.1',
        'tqdm',
    ],
    package_data = {
        "root_gnn": ["config/*.yaml"]
    },
    setup_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    scripts=[
        'root_gnn/scripts/train_classifier',
        'root_gnn/scripts/view_training_log',
        'root_gnn/scripts/create_tfrecord',
        'root_gnn/scripts/evaluate_global_classifier',
        'root_gnn/scripts/calculate_metrics',
        'root_gnn/scripts/evaluate_w_qcd_classifier',
        'root_gnn/scripts/evaluate_wtagger',
        'root_gnn/scripts/calculate_wtagger_metrics',
        'root_gnn/scripts/train_decay_simulator',
        'root_gnn/scripts/evaluate_herwig_hadron',
        'root_gnn/scripts/calculate_herwig_metrics',
        'root_gnn/scripts/split_files_for_nn',
        'root_gnn/scripts/train_top_reco.py',
        'root_gnn/scripts/evaluate_top_reco',
        'root_gnn/scripts/calculate_topreco_metrics',
        'root_gnn/scripts/train_HI.py',
        'root_gnn/scripts/train_gan.py',
    ],
)
