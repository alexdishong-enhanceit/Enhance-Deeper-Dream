import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="DeeperDream",
    version="0.0.1",
    author="Jacob Thompson",
    author_email="JaThompson@EnhanceIT.com",
    description="Enhance IT Machine Learning training group (January 2020)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JaThompsonEnhanceIT/Enhance-Deeper-Dream",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires='>=3',
    package_data={
    },
)
