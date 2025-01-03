from setuptools import setup, find_packages

setup(
    name="vital_signs_lib",
    version="1.0.0",
    description="Librería para entrenar y predecir riesgos basados en signos vitales",
    author="Enrique Teruel Gutiérrez",
    author_email="unir@unir.com",
    url="https://github.com/enriqueteru/vital-signs-lib-ia", 
    packages=find_packages(where="src"), 
    package_dir={"": "src"}, 
    include_package_data=True, 
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib",
    ], 
    extras_require={
        "dev": ["pytest", "pytest-cov"], 
    },
    python_requires=">=3.8", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ], 
)
