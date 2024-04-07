from functools import partial
from operator import itemgetter
from typing import Callable, TypeVar

from .Either import Either, success, fail
from .Void import Void, VoidType

U = TypeVar('U')
V = TypeVar('V')

type Position = int
type ErrorType = str

type ParseResultSucceeded[U] = tuple[Position, U]
type ParseResultFailed = tuple[Position, ErrorType]
type ParseResult[U] = Either[ParseResultSucceeded[U], ParseResultFailed]

type ParserSucceeded[U] = Callable[[str, Position], ParseResultSucceeded[U]]
type ParserFailed = Callable[[str, Position], ParseResultFailed]
type Parser[U] = Callable[[str, Position], ParseResult[U]]

def satisfy(predicate: Callable[[str], bool], desc: str) -> Parser[str]:
    def f(s: str, idx: Position) -> ParseResult[str]:
        if idx == len(s):
            return fail((idx, desc))
        c = s[idx]
        if not predicate(c):
            return fail((idx, desc))
        idx += 1
        return success((idx, c))
    return f

def many(parser: Parser[str]) -> Parser[list[str]]:
    def f(s: str, idx: Position) -> ParseResult[list[str]]:
        out = []
        while True:
            res = parser(s, idx)
            if not res.is_success():
                return success((idx, out))
            idx, token = res.get()
            out.append(token)
    return f

def many1(parser: Parser[str]) -> Parser[list[str]]:
    def f(s: str, idx: Position) -> ParseResult[list[str]]:
        out = []

        res = parser(s, idx)
        if not res.is_success():
            return res
        idx, token = res.get()
        out.append(token)

        res = many(parser)(s, idx)
        if res.is_success():
            idx, token = res.get()
            out.extend(token)

        return success((idx, out))
    return f

def literal(s: str, *, desc: str | None = None) -> Parser[str]:
    if desc is None:
        desc = f'string "{s}"'
    len_literal = len(s)
    def f(s_: str, idx: Position) -> ParseResult[str]:
        if idx > len(s_) - len_literal or s_[idx:idx+len_literal] != s:
            return fail((idx, desc))
        idx += len_literal
        return success((idx, s))
    return f

def parse_eof(s: str, idx: Position) -> ParseResult[VoidType]:
    if idx != len(s):
        return fail((idx, 'eof'))
    return success((idx, Void))

def with_default(parser: Parser[U], *, default: U) -> ParserSucceeded[U]:
    def f(s: str, idx: Position) -> ParseResult[U]:
        res = parser(s, idx)
        if res.is_success():
            return res
        return success((idx, default))
    return f

def with_description(parser: Parser[U], desc: str) -> Parser[U]:
    def f(s: str, idx: Position) -> ParseResult[U]:
        res = parser(s, idx)
        if res.is_success():
            return res
        idx, _ = res.get()
        return fail((idx, desc))
    return f

def pmap(func: Callable[[U], V], parser: Parser[U]) -> Parser[V]:
    def f(s: str, idx: Position) -> ParseResult[V]:
        res = parser(s, idx)
        if not res.is_success():
            return res
        idx, token = res.get()
        return success((idx, func(token)))
    return f

def pchain(*parsers: Parser[U]) -> Parser[list[U]]:
    def f(s: str, idx: Position) -> ParseResult[list[U]]:
        out = []

        for parser in parsers:
            res = parser(s, idx)
            if not res.is_success():
                return res
            idx, val = res.get()
            if val is not Void:
                out.append(val)

        return success((idx, out))
    return f

join: Callable[[str], str] = lambda xs: ''.join(xs)
pjoin = partial(pmap, join)

def _join_descriptions(xs: list[str]) -> str:
    if len(xs) == 0:
        return ''
    if len(xs) == 1:
        return xs[0]
    *init, tail = xs
    return ', '.join(init) + ' or ' + tail

def anyof(*parsers: Parser[U]) -> Parser[U]:
    def f(s: str, idx: Position) -> ParseResult[U]:
        descriptions = []

        for parser in parsers:
            res = parser(s, idx)
            if res.is_success():
                return res
            _, desc = res.get()
            descriptions.append(desc)

        return fail((idx, _join_descriptions(descriptions)))
    return f

pselect = lambda idx, parser: pmap(itemgetter(idx), parser)

sepby1: Callable[[Parser[U], Parser[V]], Parser[U]] = lambda parser, separator: pmap(
    lambda xs: [xs[0], *xs[1]],
    pchain(
        parser,
        many(pselect(1, pchain(separator, parser))),
    ),
)

const: Callable[[U], Callable[[V], U]] = lambda x: lambda _: x
pvoid: Callable[[Parser[U]], Parser[VoidType]] = partial(pmap, const(Void))

lazy = lambda ff: lambda *args: ff()(*args)
