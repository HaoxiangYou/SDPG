from setuptools import find_packages, setup

setup(
    name="rsl_rl",
    version="1.0.2",
    author="Nikita Rudin",
    author_email="rudinn@ethz.ch",
    license="BSD-3-Clause",
    packages=find_packages(),
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.6",
)
