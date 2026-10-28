from setuptools import setup

setup(
    name="MechanicalEngModule",
    version="2.1",
    description="Preloaded bolted joint analysis per NASA TM-106943, "
                "with random/sine vibration loads",
    author="Victor J. Marrero-Fontanez",
    author_email="victor.marrero@gmail.com",
    python_requires=">=3.10",
    packages=["MechanicalEngModule"],
    install_requires=["numpy", "pandas", "matplotlib"],
    extras_require={
        "gui": ["streamlit"],
        "notebook": ["jupyter"],
        "test": ["pytest"],
    },
)
