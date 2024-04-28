from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    requires = [i.strip() for i in  fp.readlines() if i[0] != "#"]

setup(
    name='minimal-bpe',
    version='0.0.0', # Version can be maintained separately
    description='A library for Byte Pair Encoding experiments',
    author='Andrej Karpathy',
    author_email='andrej.karpathy@gmail.com',
    packages=["minbpe"],
    install_requires=requires,
    python_requires='>=3.10' # I have tested it works for 3.10. Didn't test yet for other versions.
)
