from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="vaa_llm_backend",
    version="0.1.0",
    packages=find_packages(include=['chatbot_api', 'chatbot_api.*']),
    install_requires=parse_requirements('requirements.txt'),
    # author="Your Name",
    # author_email="your_email@example.com",
    # description="A brief description of your package",
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    url="https://github.com/maximusvitutus/argument-condensation",
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
)
