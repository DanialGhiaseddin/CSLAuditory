from setuptools import setup, find_packages

setup(
    name="my_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "librosa",  # Specify the required version or leave it without version constraints
        "numpy",
        "soundfile",
        "matplotlib",
        # Add any other dependencies your package needs
    ],
    author="Danial Ghiaseddin",
    author_email="danial.ghiaseddin@gmail.com",
    description="A description of your package.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/my_package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
