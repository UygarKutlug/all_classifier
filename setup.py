from setuptools import setup, find_packages
import all_classifier

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name='all-classifier-uygar',
    version=all_classifier.__version__,
    author='uygar',
    author_email='uygarkutlug@gmail.com',
    description=' Test multiple classification models and return their performance metric values.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['pandas','scikit-lear','xgboost','lightgbm'],
    licence='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT license',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    url=''

)