from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages
from setuptools.dist import Distribution

__VERSION__ = '0.1.0'


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
        # https://github.com/tensorflow/tensorflow/issues/7166
        # 'tensorflow>=2.0.0-alpha0',
    ],
    extras_require={
        'tf_cpu': ['tensorflow>=2.0.0-beta0'],
        'tf_gpu': ['tensorflow-gpu>=2.0.0-beta0'],
    },
    # TODO: test requires
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='MIT',
    keywords='tensorflow layers ops',
)
