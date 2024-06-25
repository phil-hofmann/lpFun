from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='lpfun',
        author='Phil-Alexander Hofmann',
        author_email='mail@philhofmann.de',
        description="A package for fast polynomial interpolation and differentiation.", 
        python_requires='>=3.8',
        version='0.0.1',
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        install_requires=[
            'numpy',
            'numba',
            'pytest',
            'scipy',
        ],
    )
