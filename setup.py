from setuptools import setup, find_packages


setup(
    name="ReFrame",
    description="Re(view) (data)Frame",
    long_description="Quickly spot common issues in a Pandas dataframe",
    version="0.1",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy", "pandas", "scipy", "tabulate"],
)
