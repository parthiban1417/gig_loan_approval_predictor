from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name = 'gigloanpredictor',
    version= '0.0.1',
    author= 'Parthiban',
    author_email= 'parthiban1315@gmail.com',
    description="Loan approval predictor for gig community",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parthiban1417/gig_loan_approval_predictor",
    project_urls={
        "Bug Tracker": "https://github.com/parthiban1417/gig_loan_approval_predictor/issues",
        },
    package_dir={"": "src"},
    packages= find_packages(where="src"),
    install_requires = install_requires
)