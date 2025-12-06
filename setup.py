"""Setup script for RL Agent Ludo."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rl-agent-ludo",
    version="0.1.0",
    author="Hitesh Balegar",
    author_email="hiteshbalegar@gmail.com",
    description="A research-grade experimental framework for training and comparing RL agents in Ludo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RLagentLudo",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    # entry_points={
    #     "console_scripts": [
    #         "rl-agent-ludo=rl_agent_ludo.main:main",
    #     ],
    # },
)
