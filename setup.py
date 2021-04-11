from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='shopee',
    version='0.0.1',
    packages=[
        'shopee',
    ],
    url='',
    license='MIT',
    author='andrei',
    author_email='popow.andrej2009@yandex.ru',
    description='Library for Shopee contest.',
    install_requires=requirements,
)
