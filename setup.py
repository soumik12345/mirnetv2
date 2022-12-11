import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

setuptools.setup(
    name="restorers",
    version="0.0.1",
    description="Image Restoration toolkit in TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/soumik12345/mirnetv2",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
