import os
from typing import List
from urllib.parse import parse_qsl

from setuptools import setup

GITHUB_TOKEN_ENV = 'GITHUB_TOKEN'
GITHUB_TOKEN_VAR1 = '${' + GITHUB_TOKEN_ENV + '}'
GITHUB_TOKEN_VAR2 = '$' + GITHUB_TOKEN_ENV


def read_requirements(req_file_path: str) -> List[str]:
    token = os.environ.get(GITHUB_TOKEN_ENV, '')
    requirement_url_list = []
    with open(req_file_path) as req_file:
        for s in req_file.readlines():
            if not s or s.startswith('-f '):
                continue
            requirement_url = s.replace(GITHUB_TOKEN_VAR1, token).replace(GITHUB_TOKEN_VAR2, token)
            if '#' in requirement_url:
                _, query = requirement_url.split('#')
                query_dict = dict(parse_qsl(query))
                requirement_url_list.append(query_dict['egg'] + ' @ ' + requirement_url)
            else:
                requirement_url_list.append(requirement_url)
        return requirement_url_list


requirements = read_requirements('requirements.txt')

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
