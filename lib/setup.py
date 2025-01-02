from setuptools import setup, find_packages

setup(
    name="vital_signs_lib",
    version="1.0.0",
    description="Librería para entrenar y predecir riesgos basados en signos vitales",
    author="Enrique Teruel Gutiérrez",
    author_email="unir@unir.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib",
        "pytest"
    ],
    python_requires=">=3.8",
)
