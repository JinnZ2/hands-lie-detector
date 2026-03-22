from setuptools import setup, find_packages

setup(
    name="hands-lie-detector",
    version="0.1.0",
    description="Experience detection through persistent physical markers in hands",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[],  # Core scoring module has no deps
    extras_require={
        "vision": ["torch>=2.0", "torchvision>=0.15", "Pillow>=9.0"],
        "prompt": ["anthropic>=0.40", "openai>=1.0"],
        "all": [
            "torch>=2.0",
            "torchvision>=0.15",
            "Pillow>=9.0",
            "anthropic>=0.40",
            "openai>=1.0",
        ],
    },
    license="MIT",
)
