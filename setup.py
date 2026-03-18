from setuptools import setup, find_packages


def readme():
    with open('README.md', 'r') as f:
        return f.read()


setup(
    name='SpikingNN',
    version='0.0.1',
    author='Kovalev Nickolai',
    author_email='kovalev.na@phystech.edu',
    description='This is simple module to simulate spiking neural networks',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='your_url',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=['requests>=2.25.1'],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords='spiking neural networks cpg',
    project_urls={
        'GitHub':'https://github.com/brodayagabro/SpikingNN'
    },
    python_requires='>=3.6'
    )
