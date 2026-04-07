from setuptools import setup, find_packages

setup(
    name='all-classifier-uygar-final-v3', # İsmi tekrar değiştirdik
    version='0.1.3',
    author='uygar',
    author_email='uygarkutlug@gmail.com',
    description='Test multiple classification models.',
    long_description='Test multiple classification models and return their performance metrics.',
    long_description_content_type="text/plain", # README okumayı iptal ettik, düz metin yaptık
    packages=find_packages(),
    install_requires=['pandas', 'scikit-learn', 'xgboost', 'lightgbm'],
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)