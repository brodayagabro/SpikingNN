from setuptools import setup, find_packages


def readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()


setup(
    name='SpikingNN',
    version='0.0.2',
    author='Kovalev Nickolai',
    author_email='kovalev.na@phystech.edu',
    description='This is simple module to simulate spiking neural networks',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/brodayagabro/SpikingNN',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    
    # ✅ ОБНОВЛЁННЫЕ ЗАВИСИМОСТИ
    install_requires=[
        'streamlit>=1.32.0',
        'numpy>=1.20.0',
        'plotly>=5.18.0',
        'pandas>=2.0.0',
        'networkx>=3.0',
        'matplotlib>=3.4.0',
        'requests>=2.25.1',
    ],
    
    # ✅ ENTRY POINTS ДЛЯ КОМАНДНОЙ СТРОКИ
    entry_points={
        'console_scripts': [
            'spikingnn=SpikingNN.cli:main',
            'spknn=SpikingNN.cli:main',  # Короткая версия
        ],
    },
    
    # ✅ ВКЛЮЧЕНИЕ ДОПОЛНИТЕЛЬНЫХ ФАЙЛОВ
    package_data={
        'SpikingNN': ['*.py', '*.md', '*.txt'],
    },
    include_package_data=True,
    
    # ✅ ОБНОВЛЁННЫЕ CLASSIFIERS
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
    ],
    
    keywords='spiking neural networks cpg izhikevich',
    project_urls={
        'GitHub': 'https://github.com/brodayagabro/SpikingNN',
        'Documentation': 'https://github.com/brodayagabro/SpikingNN#readme',
    },
    python_requires='>=3.8'
)