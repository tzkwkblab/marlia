from setuptools import setup

setup(
    name="marlia",
    packages=["marlia"],
    version="0.0.1",
    install_requires=[
        "stable-baselines3==1.6.2",
        "sb3-contrib==1.6.2"
    ]
)