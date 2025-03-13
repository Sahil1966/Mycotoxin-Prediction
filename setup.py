from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    Reads requirements.txt and returns a list of dependencies,
    ignoring empty lines and '-e .'
    """
    requirement_lst: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            # Process each line
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':  # Ignore '-e .'
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found!")

    return requirement_lst

setup(
    name="mycotoxin_pipeline",
    version="0.1",
    author="Sahil Patel",
    author_email="sahil94256@gmail.com",
    packages=find_packages(where="src"),  # Finds packages in 'src/'
    package_dir={"": "src"},  # Treats 'src/' as root
    install_requires=get_requirements(),  # Reads dependencies from requirements.txt
    python_requires=">=3.8",  # Ensures compatibility
)
