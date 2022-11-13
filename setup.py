import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="paint_with_words",
    py_modules=["paint_with_words"],
    version="0.0.2",
    description="Stable diffusion's Implementation of Paint with words, from eDiffi Paper https://arxiv.org/abs/2211.01324 ",
    author="Simo Ryu",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
