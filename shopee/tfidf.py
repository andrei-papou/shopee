import re
from typing import FrozenSet, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DIGIT_PATTERN = '(\d+(\.\d*)?)'
ALPHA_PATTERN = '[a-zA-Z]+'
UNITS_PATTERN = '[a-wyzA-WYZ]+'
UNICODE_RE = r'((\\x[a-zA-Z0-9]{2}){2,4})'
digit_alpha_re = re.compile(f'^(?P<digit>{DIGIT_PATTERN})(?P<alpha>{ALPHA_PATTERN})$')
alpha_digit_re = re.compile(f'^(?P<alpha>{ALPHA_PATTERN})(?P<digit>{DIGIT_PATTERN})$')
digit_alpha_digit_re = re.compile(
    f'^(?P<digit1>{DIGIT_PATTERN})(?P<alpha>{ALPHA_PATTERN})(?P<digit2>{DIGIT_PATTERN})$')
alpha_digit_alpha_re = re.compile(
    f'^(?P<alpha1>{ALPHA_PATTERN})(?P<digit>{DIGIT_PATTERN})(?P<alpha2>{ALPHA_PATTERN})$')
size_2d_re = re.compile(
    f'^(?P<width_val>{DIGIT_PATTERN})(?P<width_unit>{UNITS_PATTERN})x'
    f'(?P<length_val>{DIGIT_PATTERN})(?P<length_unit>{UNITS_PATTERN})$')
size_3d_re = re.compile(
    f'^(?P<width_val>{DIGIT_PATTERN})(?P<width_unit>{UNITS_PATTERN})x'
    f'(?P<length_val>{DIGIT_PATTERN})(?P<length_unit>{UNITS_PATTERN})x'
    f'(?P<height_val>{DIGIT_PATTERN})(?P<height_unit>{UNITS_PATTERN})$')


def preprocess_alpha_token(tok: str) -> str:
    tok = tok.strip()
    if tok == 'g':
        return 'gr'
    return tok


def preprocess_title(title: str) -> str:
    title = title.lower() \
        .replace('=', '') \
        .replace(',', '') \
        .replace('&', '') \
        .replace('!', '') \
        .replace('#', '') \
        .replace('(', ' ') \
        .replace(')', ' ') \
        .replace('[', ' ') \
        .replace(']', ' ') \
        .replace('/', ' ') \
        .replace('|', ' ') \
        .replace('_', ' ') \
        .replace('{', ' ') \
        .replace('}', ' ') \
        .replace('-', ' ') \
        .replace('\\xc3\\x97', 'x')
    for x, _ in re.findall(UNICODE_RE, title):
        title = title.replace(x, '')
    if title.startswith('b"'):
        title = title[2:]
    if title.endswith('"'):
        title = title[:-1]
    title_part_list = title.split(' ')
    new_title_part_list = []
    for title_part in title_part_list:
        if not title_part.replace(' ', ''):
            continue

        if title_part.startswith('+'):
            title_part = f'+ {title_part[1:]}'
        if title_part.endswith('+'):
            title_part = f'{title_part[:-1]} +'
        m = digit_alpha_re.match(title_part)
        if m is not None:
            title_part = m.group('digit').strip() + ' ' + preprocess_alpha_token(m.group('alpha').strip())
        m = alpha_digit_re.match(title_part)
        if m is not None:
            title_part = preprocess_alpha_token(m.group('alpha').strip()) + ' ' + m.group('digit').strip()
        m = digit_alpha_digit_re.match(title_part)
        if m is not None:
            title_part = m.group('digit1').strip() + ' ' + \
                         preprocess_alpha_token(m.group('alpha').strip()) + ' ' + m.group('digit2').strip()
        m = alpha_digit_alpha_re.match(title_part)
        if m is not None:
            title_part = preprocess_alpha_token(m.group('alpha1').strip()) + ' ' + \
                         m.group('digit').strip() + ' ' + preprocess_alpha_token(m.group('alpha2').strip())
        m = size_2d_re.match(title_part)
        if m is not None:
            title_part = \
                m.group('width_val').strip() + ' ' + preprocess_alpha_token(m.group('width_unit').strip()) + \
                ' x ' + \
                m.group('length_val').strip() + ' ' + preprocess_alpha_token(m.group('length_unit').strip())
        m = size_3d_re.match(title_part)
        if m is not None:
            title_part = \
                m.group('width_val').strip() + ' ' + preprocess_alpha_token(m.group('width_unit').strip()) + \
                ' x ' + \
                m.group('length_val').strip() + ' ' + preprocess_alpha_token(m.group('length_unit').strip()) + \
                ' x ' + \
                m.group('height_val').strip() + ' ' + preprocess_alpha_token(m.group('height_unit').strip())
        title_part = title_part.strip()
        if title_part.endswith('.'):
            title_part = title_part[:-1]
        new_title_part_list.append(title_part)
    return ' '.join(new_title_part_list)


def get_embedding_tuple(
        df: pd.DataFrame,
        stop_words: Optional[Union[str, FrozenSet[str]]] = None,
        preprocess: bool = False,
        max_features: int = 25_000,
        token_pattern: str = r'(?u)\b\w\w+\b') -> Tuple[np.ndarray, List[str]]:
    title_list, posting_id_list = df.title.tolist(), df.posting_id.tolist()
    if preprocess:
        title_list = [preprocess_title(title) for title in title_list]
    model = TfidfVectorizer(stop_words=stop_words, max_features=max_features, token_pattern=token_pattern)
    return model.fit_transform(raw_documents=title_list), posting_id_list
