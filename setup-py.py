from setuptools import setup, find_packages

setup(
    name="credit_card_fraud",
    version="0.1.0",
    description="Advanced credit card fraud detection using ensemble learning",
    author="Alexander Clarke",
    author_email="alexanderclarke365@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "imbalanced-learn>=0.10.0",
        "joblib>=1.3.0",
        "streamlit>=1.25.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)