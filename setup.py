from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

    setup(
        name='crop_recommendation',
        version='1.0.0', 
        description='This is a Crop Recommendation model for Crop Based on Whether',
        author='Anirudh Johare',
        author_email='anirudhjohare@gmail.com')
       