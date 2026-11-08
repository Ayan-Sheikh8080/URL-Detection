from setuptools import find_packages, setup

setup(
    name="phishing-url-analyzer",
    version="1.0.0",
    description="MLOps pipeline for phishing URL analysis with sandboxed browser agent",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.27.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "xgboost-cpu>=2.1.1,<4.0",
        "playwright>=1.42.0",
        "mlflow>=2.10.0",
        "boto3>=1.34.0",
        "joblib>=1.3.0",
        "prometheus-client>=0.20.0",
        "httpx>=0.27.0",
    ],
)