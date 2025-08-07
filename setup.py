from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flash-sink-attn",
    version="0.1.0",
    author="Wenhao Li",
    author_email="wenhaoli20000901@gmail.com",
    description="A simple implementation of Sink Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wenhaoli-xmu/flash_sink_attn",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "triton>=3.1.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)