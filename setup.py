from setuptools import find_packages, setup

setup(
    name="phishing-url-analyzer",
    version="1.0.0",
    description="MLOps pipeline for phishing URL analysis with sandboxed browser agent",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "playwright",
        "express",
        "mlflow",
        "dvc",
        "boto3",
        "fastapi",
        "uvicorn",
        "evidently",
    ],
)