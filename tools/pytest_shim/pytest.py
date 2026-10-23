# -*- coding: utf-8 -*-
"""pytest 가 설치되지 않은 환경(오프라인 연산보드·CI 샌드박스)용 최소 대역.

run_tests.py 가 sys.path 의 **맨 뒤**에 이 폴더를 붙인다. 진짜 pytest 가
설치돼 있으면 그쪽이 먼저 잡히므로 이 파일은 쓰이지 않는다.
"""
import math


def fixture(fn=None, **_kw):
    def wrap(f):
        f._is_fixture = True
        return f
    return wrap(fn) if fn is not None else wrap


class approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected, self.rel, self.abs = expected, rel, abs

    def _close(self, a, b):
        if self.abs is None and self.rel is None:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)
        return math.isclose(a, b,
                            rel_tol=self.rel if self.rel is not None else 0.0,
                            abs_tol=self.abs if self.abs is not None else 0.0)

    def __eq__(self, other):
        try:
            return self._close(float(other), float(self.expected))
        except TypeError:
            return all(self._close(float(x), float(y))
                       for x, y in zip(other, self.expected))

    def __req__(self, other):
        return self.__eq__(other)

    def __repr__(self):
        return f"approx({self.expected})"


class _Raises:
    def __init__(self, exc):
        self.exc = exc
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f"{self.exc.__name__} 이 발생하지 않았다")
        self.value = v
        return issubclass(t, self.exc)


def raises(exc):
    return _Raises(exc)


class mark:
    @staticmethod
    def parametrize(*a, **k):
        def deco(f):
            return f
        return deco

    @staticmethod
    def skipif(*a, **k):
        def deco(f):
            return f
        return deco
