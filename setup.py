from setuptools import setup

def get_version():
    """Gets the pettingzoo version."""
    path = "pyproject.toml"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("version"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")

setup(
    name="Wingman",
    version=get_version(),
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
