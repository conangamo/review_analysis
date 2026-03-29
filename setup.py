"""Setup script for Product Review Analyzer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="product-review-analyzer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered product review analysis with aspect-based sentiment detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/product-review-analyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pra-setup=scripts.setup_database:main",
            "pra-download=scripts.download_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.sql"],
    },
    zip_safe=False,
)
