import os
from setuptools import setup, find_packages

from digits.extensions.data import GROUP as DIGITS_PLUGIN_GROUP

setup(
    name="u_net_sunnybrook_data_plugin",
    version="0.0.1",
    author="Windforces",
    description=("A data ingestion plugin for the Sunnybrook cardiac dataset"),
    license="Apache",
    packages=find_packages(),
    entry_points={
        DIGITS_PLUGIN_GROUP: [
            'class=src:DataIngestion',
        ]
    },
    include_package_data=True,
    install_requires=['pydicom'],
)
