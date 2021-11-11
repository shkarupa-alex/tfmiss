from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution

__VERSION__ = '0.15.4'


def get_ext_modules():
    ext_modules = []
    if "--platlib-patch" in sys.argv:
        if sys.platform.startswith("linux"):
            # Manylinux2010 requires a patch for platlib
            ext_modules = [Extension("_foo", ["stub.cc"])]
        sys.argv.remove("--platlib-patch")
    return ext_modules


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True


with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tfmiss',
    version=__VERSION__,
    description='Missing layers, ops & etc. for TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/shkarupa-alex/tfmiss',
    author='Shkarupa Alex',
    author_email='shkarupa.alex@gmail.com',
    packages=find_packages(),
    ext_modules=get_ext_modules(),
    install_requires=Path("requirements.txt").read_text().splitlines(),
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='MIT',
    keywords='tensorflow layers ops',
)
