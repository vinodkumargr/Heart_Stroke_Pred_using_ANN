from setuptools import find_packages, setup
from typing import List

requirement_file_name = "requirements.txt"
REMOVE_PACKAGES = "-e ."

def get_requirements()-> List[str]:
    with open(requirement_file_name) as requirement_file:
        requirement_list = requirement_file.readline()
    requirement_list = [requirement_name.replace("\n","") for requirement_name in requirement_list]
    
    if REMOVE_PACKAGES in requirement_list:
        requirement_list.remove(REMOVE_PACKAGES)
    return requirement_list



setup(
    name="heart-stroke-predictior",
    version="0.0.1",
    author="Vinod",
    author_email="vks7483437@gmail.com",
    packages=find_packages(),
    install_requires=[],
)
