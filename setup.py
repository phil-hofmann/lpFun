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
            'numpy==1.22.4',
            'numba==0.58.1',
            'scipy==1.7.1',
            'pytest==8.2.2',
            'setuptools==70.1.1'
        ],
    )
