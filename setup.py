from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.dist import Distribution

__VERSION__ = '0.10.0'


class InstallPlatlib(install):
    """This class is needed due to a bug in
    distutils.command.install.finalize_options().
    See https://github.com/google/or-tools/issues/616#issuecomment-371480314"""

    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


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
    install_requires=[
        'tensorflow==2.4.0',
        'scipy>=1.5.4',
        'matplotlib>=3.1.1',
        'tabulate>=0.8.6',
    ],
    include_package_data=True,
    zip_safe=False,
    cmdclass={'install': InstallPlatlib},
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
