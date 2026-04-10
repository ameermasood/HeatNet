from setuptools import find_packages
from setuptools import setup


setup(
    name="heatnet",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "heatnet=heatnet.cli:main",
        ]
    },
)
