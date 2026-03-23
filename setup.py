from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mammoth-moe",
    version="0.1.2",
    
    long_description=long_description,
    long_description_content_type="text/markdown", 
)
