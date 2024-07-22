import subprocess
from setuptools import setup, find_packages


def install_pytorch():
    subprocess.check_call(
        [
            "python",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu118",
            "torch==2.1.1",
            "torchvision==0.16.1",
        ]
    )


def parse_requirements(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    requirements = []
    for line in lines:
        line = line.strip()
        if line.startswith("--extra-index-url"):
            continue
        elif line.startswith("git+"):
            if "#egg=" in line:
                package_name = line.split("#egg=")[-1]
            else:
                package_name = line.split("/")[-1].split(".git")[0]
            requirements.append(f"{package_name} @ {line}")
        else:
            requirements.append(line)
    return requirements


setup(
    name="hocap-toolkit",
    version="0.1.0",
    author="Jikai Wang",
    author_email="jikai.wang@utdallas.edu",
    description="HOCap Toolkit is a Python package that provides evaluation and visualization tools for the HOCap dataset.",
    license="GPLv3",
    packages=find_packages(exclude=["docker", "examples", "external"]),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">3.9, <3.11",
    install_requires=parse_requirements("requirements.txt"),
)
