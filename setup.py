from setuptools import setup

setup(
    name="Wingman",
    version="0.0.5",
    author="Jet",
    author_email="taijunjet@hotmail.com",
    description="Wingman for all your AI applications",
    url="https://github.com/jjshoots/Wingman",
    license_files=("LICENSE.txt"),
    long_description="Wingman library for AI projects.",
    long_description_content_type="text/markdown",
    keywords=["Machine Learning"],
    python_requires=">=3.7, <3.11",
    packages=["wingman"],
    include_package_data=True,
    install_requires=["numpy", "torch", "wandb", "pyyaml"],
)
