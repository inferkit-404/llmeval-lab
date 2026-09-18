"""LLMEval-Lab - Lightweight Multimodal Evaluation Laboratory"""

from setuptools import setup, find_packages

setup(
    name="llmeval-lab",
    version="0.1.0",
    author="LLMEval Team",
    author_email="team@llmeval.dev",
    description="Lightweight Multimodal Evaluation Laboratory for LLMs",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ai-never/LLMEval-Lab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "rich>=13.0.0",
        "plotly>=5.15.0",
        "dash>=2.14.0",
    ],
    extras_require={
        "vllm": ["vllm>=0.2.0"],
        "api": ["openai>=1.0.0", "anthropic>=0.8.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "ruff>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "llmeval=cli:main",
        ],
    },
    license="Apache 2.0",
    keywords="llm evaluation benchmark multimodal",
)
