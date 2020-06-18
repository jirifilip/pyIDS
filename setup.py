from setuptools import setup

def read(fname):
    with open(fname, 'r') as fhandle:
        return fhandle.read()

def read_reqs(fname):
    # Skip first requirement from pipenv --lock
    reqs = read(fname).splitlines()
    return [req.strip() for req in reqs if req.strip()]

base_reqs = read_reqs('requirements.txt')

setup(
    name='viaduct-pyids',
    version='0.1.0',
    author='Viaduct Inc.',
    author_email='bora@viaduct.ai',
    packages=['pyids'],
    license='LICENSE.txt',
    long_description=open('README.md').read(),
    install_requires=base_reqs
)

