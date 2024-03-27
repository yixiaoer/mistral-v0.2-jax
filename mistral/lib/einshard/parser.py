from types import EllipsisType
from typing import Callable
import unicodedata

from ..parsec import anyof, const, literal, many, many1, parse_eof, pchain, pjoin, pmap, pvoid, satisfy, sepby1, with_default

ElementLeft = str | EllipsisType
ElementRight = tuple[str | None, int] | EllipsisType
Expression = tuple[list[ElementLeft], list[ElementRight]]

is_identifier_char: Callable[[str], bool] = lambda c: unicodedata.category(c)[0] == 'L' or c == '_'
is_0_to_9: Callable[[str], bool] = lambda c: c.isdigit()
is_1_to_9: Callable[[str], bool] = lambda c: c.isdigit() and c != '0'
is_space: Callable[[str], bool] = lambda c: c.isspace()

parse_0_to_9 = satisfy(is_0_to_9, 'digit 0-9')
parse_1_to_9 = satisfy(is_1_to_9, 'digit 1-9')
parse_identifier_char = satisfy(is_identifier_char, 'identifier')
parse_identifier = pjoin(many1(parse_identifier_char))
parse_space = satisfy(is_space, 'space')
parse_spaces = many1(parse_space)
parse_spaces_optional = pvoid(many(parse_space))
parse_right_arrow = pvoid(literal('->'))
parse_ellipsis = pmap(const(...), literal('...', desc='ellipsis'))
parse_integer = pmap(int, pjoin(pchain(parse_1_to_9, pjoin(many(parse_0_to_9)))))

parse_element_left = anyof(parse_identifier, parse_ellipsis)
parse_element_right = anyof(
    pchain(with_default(parse_identifier, default=None), parse_integer),
    pchain(parse_identifier, with_default(parse_integer, default=0)),
    parse_ellipsis,
)
parse_elements_left = sepby1(parse_element_left, parse_spaces)
parse_elements_right = sepby1(parse_element_right, parse_spaces)
parse_expression = pchain(
    parse_spaces_optional,
    parse_elements_left,
    parse_spaces_optional,
    parse_right_arrow,
    parse_spaces_optional,
    parse_elements_right,
    parse_spaces_optional,
    parse_eof,
)

# print(pchain(literal('123 '), parse_element_left)('123 .a..', 0))
# print(parse_element_left('a', 0))
# print(parse_element_left('...', 0))
# print(parse_element_right('...', 0))
# print(parse_expression('a ... b -> b ... a1', 0))
