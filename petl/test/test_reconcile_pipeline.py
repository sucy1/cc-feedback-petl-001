# -*- coding: utf-8 -*-
"""Tests for the reconciliation (duizhang) pipeline behaviour.

Covers four requirements:

1. join / leftjoin / rightjoin / antijoin (both the sort-based and the
   hash-based implementations) do not match rows whose key is null, where
   null means ``None`` or an empty / whitespace-only string. The legacy
   "nulls match nulls" behaviour is available via ``match_nulls=True``.
2. ``sort`` orders amounts with nulls last, numeric strings by value,
   other values by text code point, stably and without raising TypeError.
3. the ``sum`` and ``mean`` aggregations cope with mixed Decimal, float,
   int and None values.
4. ``tocsv`` accepts an already-open text-mode file object.
"""
from __future__ import absolute_import, print_function, division

import io
import os
import tempfile
from decimal import Decimal

import pytest

from petl.errors import ArgumentError
from petl.test.helpers import ieq
from petl import (
    join, leftjoin, rightjoin, antijoin,
    hashjoin, hashleftjoin, hashrightjoin, hashantijoin,
    sort, mergesort, aggregate, sum, mean, tocsv, appendcsv, wrap,
)


# ---------------------------------------------------------------------------
# 1. null keys do not participate in matching
# ---------------------------------------------------------------------------

LEFT = [['k', 'v'],
        [None, 'L-none'],
        ['', 'L-empty'],
        ['  ', 'L-ws'],
        ['a', 'L-a']]

RIGHT = [['k', 'w'],
         [None, 'R-none'],
         ['', 'R-empty'],
         ['a', 'R-a']]

INNER_JOIN_IMPLS = [join, hashjoin]
LEFT_JOIN_IMPLS = [leftjoin, hashleftjoin]
RIGHT_JOIN_IMPLS = [rightjoin, hashrightjoin]
ANTI_JOIN_IMPLS = [antijoin, hashantijoin]


@pytest.mark.parametrize('join_impl', INNER_JOIN_IMPLS)
def test_inner_join_null_keys_never_match(join_impl):
    # none of the null-key rows on either side may pair up
    expect = (('k', 'v', 'w'),
              ('a', 'L-a', 'R-a'))
    ieq(expect, join_impl(wrap(LEFT), wrap(RIGHT), key='k'))


@pytest.mark.parametrize('join_impl', INNER_JOIN_IMPLS)
def test_inner_join_match_nulls_restores_legacy(join_impl):
    result = list(join_impl(wrap(LEFT), wrap(RIGHT), key='k',
                            match_nulls=True))
    # the two non-whitespace null forms pair exactly as in the old behaviour
    assert (None, 'L-none', 'R-none') in result
    assert ('', 'L-empty', 'R-empty') in result
    assert ('a', 'L-a', 'R-a') in result
    # the whitespace-only key has no exact equal on the right, so no row
    assert not any(row[1] == 'L-ws' for row in result[1:])


@pytest.mark.parametrize('leftjoin_impl', LEFT_JOIN_IMPLS)
def test_left_join_keeps_left_null_rows(leftjoin_impl):
    result = list(leftjoin_impl(wrap(LEFT), wrap(RIGHT), key='k'))
    rows = dict((row[0], row) for row in result[1:])
    # left null rows are retained with None for the right columns
    assert rows[None] == (None, 'L-none', None)
    assert rows[''] == ('', 'L-empty', None)
    assert rows['  '] == ('  ', 'L-ws', None)
    # the matching row still pairs normally
    assert rows['a'] == ('a', 'L-a', 'R-a')


@pytest.mark.parametrize('rightjoin_impl', RIGHT_JOIN_IMPLS)
def test_right_join_keeps_right_null_rows(rightjoin_impl):
    result = list(rightjoin_impl(wrap(LEFT), wrap(RIGHT), key='k'))
    rows = dict((row[0], row) for row in result[1:])
    # right null rows are retained with None for the left columns
    assert rows[None] == (None, None, 'R-none')
    assert rows[''] == ('', None, 'R-empty')
    assert rows['a'] == ('a', 'L-a', 'R-a')


@pytest.mark.parametrize('antijoin_impl', ANTI_JOIN_IMPLS)
def test_antijoin_null_rows_are_differences(antijoin_impl):
    # every left null row is unmatched, so all of them are differences
    result = list(antijoin_impl(wrap(LEFT), wrap(RIGHT), key='k'))
    rows = [row[0] for row in result[1:]]
    assert set(rows) == {None, '', '  '}


@pytest.mark.parametrize('antijoin_impl', ANTI_JOIN_IMPLS)
def test_antijoin_match_nulls_excludes_paired_nulls(antijoin_impl):
    # with legacy matching, None and '' rows pair off and disappear; only
    # the whitespace-only key (no exact counterpart) remains
    result = list(antijoin_impl(wrap(LEFT), wrap(RIGHT), key='k',
                                match_nulls=True))
    rows = [row[0] for row in result[1:]]
    assert rows == ['  ']


def test_null_key_handling_compound_key():
    left = wrap([['k1', 'k2', 'v'],
                 [None, 'x', 'L1'],
                 ['a', '', 'L2'],
                 ['a', 'b', 'L3']])
    right = wrap([['k1', 'k2', 'w'],
                  [None, 'x', 'R1'],
                  ['a', '', 'R2'],
                  ['a', 'b', 'R3']])
    # any null component makes the whole compound key null
    expect = (('k1', 'k2', 'v', 'w'),
              ('a', 'b', 'L3', 'R3'))
    ieq(expect, join(left, right, key=['k1', 'k2']))
    ieq(expect, hashjoin(left, right, key=['k1', 'k2']))

    # both null-key left rows survive as differences (order independent here)
    anti = list(antijoin(left, right, key=['k1', 'k2']))
    assert {row[2] for row in anti[1:]} == {'L1', 'L2'}


def test_null_key_join_presorted_path():
    # the presorted branch must apply the same null rules
    left = sort(wrap(LEFT), 'k')
    right = sort(wrap(RIGHT), 'k')
    expect_join = (('k', 'v', 'w'),
                   ('a', 'L-a', 'R-a'))
    ieq(expect_join, join(left, right, key='k', presorted=True))
    anti = list(antijoin(left, right, key='k', presorted=True))
    # all three left null-key rows are differences; the matched 'a' is not
    assert {row[0] for row in anti[1:]} == {None, '', '  '}


# ---------------------------------------------------------------------------
# 2. amount-column sorting
# ---------------------------------------------------------------------------

def test_sort_numeric_strings_by_value():
    table = wrap([['k'], ['10'], ['9'], [10], [9], ['010']])
    result = [row[0] for row in sort(table, 'k')][1:]
    # 9 (string and int) before 10 (string, int, zero-padded string)
    assert result == ['9', 9, '10', 10, '010']


def test_sort_nulls_last_ascending_and_descending():
    table = wrap([['k'], ['10'], [9], [None], [''], ['  '], [1]])
    asc = [row[0] for row in sort(table, 'k')][1:]
    # null forms (None, '', whitespace) all come after the real values
    assert asc[:3] == [1, 9, '10']
    assert set(asc[3:]) == {None, '', '  '}

    desc = [row[0] for row in sort(table, 'k', reverse=True)][1:]
    # descending non-null values first, nulls still last and stable
    assert desc[0] == '10'
    assert set(desc[3:]) == {None, '', '  '}


def test_sort_text_values_by_code_point():
    # '9' parses as a number so it ranks ahead of plain text; the numeric
    # column (100) and numeric string ('9') are compared by value, then text
    # values are ordered lexicographically by code point
    table = wrap([['k'], ['abc'], ['def'], [100], ['9']])
    result = [row[0] for row in sort(table, 'k')][1:]
    assert result == ['9', 100, 'abc', 'def']

    # genuine non-numeric strings fall back to code-point ordering
    text_only = wrap([['k'], ['def'], ['abc'], ['ghi']])
    assert [r[0] for r in sort(text_only, 'k')][1:] == ['abc', 'def', 'ghi']


def test_sort_is_stable_for_equal_values():
    table = wrap([['k', 'tag'],
                  [1, 'first'],
                  [1, 'second'],
                  ['1', 'third']])
    result = [row[1] for row in sort(table, 'k')][1:]
    # equal numeric value keeps input order
    assert result == ['first', 'second', 'third']


def test_sort_never_raises_type_error():
    junk = [['k'], ['abc'], [1.5], ['1.5'], [None], [-2], ['-2']]
    # mixed types that native Python 3 cannot compare must still sort
    rows = list(sort(wrap(junk), 'k'))
    assert rows[0] == ('k',)
    values = [r[0] for r in rows[1:]]
    assert values[-1] is None
    # numeric values (including the numeric strings) lead, text last
    assert values[:4] == [-2, '-2', 1.5, '1.5']
    assert values[4] == 'abc'


def test_sort_buffered_matches_in_memory():
    table = wrap([['k'], ['10'], [9], [None], ['2'], [1], ['']])
    in_mem = list(sort(table, 'k'))
    buffered = list(sort(table, 'k', buffersize=2))
    assert in_mem == buffered
    # reverse path including the merge step keeps nulls last
    in_mem_rev = list(sort(table, 'k', reverse=True))
    buffered_rev = list(sort(table, 'k', reverse=True, buffersize=2))
    assert in_mem_rev == buffered_rev
    # the two null forms are the final rows, after every non-null value
    assert {row[0] for row in buffered_rev[-2:]} == {None, ''}


def test_mergesort_uses_same_ordering():
    t1 = wrap([['k'], [None], [2], ['10']])
    t2 = wrap([['k'], [''], [9], ['2']])
    result = [row[0] for row in mergesort(t1, t2, key='k')][1:]
    # nulls (None, '') at the end, numeric-aware ordering otherwise
    assert set(result[-2:]) == {None, ''}
    assert result[:4] == [2, '2', 9, '10']


# ---------------------------------------------------------------------------
# 3. sum / mean with mixed Decimal, float, int and None
# ---------------------------------------------------------------------------

AMOUNTS = wrap([['k', 'amt'],
                ['a', Decimal('1.5')],
                ['a', 2.5],
                ['a', None],
                ['a', 3]])


def test_sum_decimal_column_returns_decimal():
    result = list(aggregate(AMOUNTS, 'k', sum, 'amt'))
    value = result[1][1]
    assert value == Decimal('7.0')
    assert isinstance(value, Decimal)


def test_mean_decimal_column_returns_decimal():
    result = list(aggregate(AMOUNTS, 'k', mean, 'amt'))
    value = result[1][1]
    assert value == Decimal('7.0') / Decimal(3)
    assert isinstance(value, Decimal)


def test_sum_mean_all_none_return_none():
    table = wrap([['k', 'amt'], ['a', None], ['a', None]])
    assert list(aggregate(table, 'k', sum, 'amt'))[1][1] is None
    assert list(aggregate(table, 'k', mean, 'amt'))[1][1] is None


def test_sum_without_decimal_keeps_legacy_types():
    int_table = wrap([['k', 'amt'], ['a', 1], ['a', 2], ['a', None]])
    int_sum = list(aggregate(int_table, 'k', sum, 'amt'))[1][1]
    assert int_sum == 3 and isinstance(int_sum, int)

    float_table = wrap([['k', 'amt'], ['a', 1], ['a', 2.0]])
    float_sum = list(aggregate(float_table, 'k', sum, 'amt'))[1][1]
    assert float_sum == 3.0 and isinstance(float_sum, float)

    float_mean = list(aggregate(float_table, 'k', mean, 'amt'))[1][1]
    assert float_mean == 1.5 and isinstance(float_mean, float)


def test_sum_mean_accept_one_shot_iterators():
    # aggregate hands a generator to the reducer, so the helpers must not
    # rely on iterating the sequence more than once
    assert sum(x for x in [1, 2, None]) == 3
    assert mean(x for x in [1, 2, None]) == 1.5
    assert sum(iter([])) is None
    assert mean(iter([])) is None


def test_float_converted_via_decimal_of_str():
    # 0.1 cannot be represented exactly as a float; Decimal(str(0.1))
    # preserves the short decimal spelling rather than the binary noise
    table = wrap([['k', 'amt'], ['a', Decimal('0.1')], ['a', 0.2]])
    value = list(aggregate(table, 'k', sum, 'amt'))[1][1]
    assert isinstance(value, Decimal)
    assert value == Decimal('0.1') + Decimal('0.2')


# ---------------------------------------------------------------------------
# 4. tocsv accepts an already-open text file object
# ---------------------------------------------------------------------------

TABLE = [['foo', 'bar'],
         ['a', 1],
         ['b', 2],
         ['c', 2]]


def test_tocsv_open_text_file_matches_path_output():
    f = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='',
                                    suffix='.csv')
    obj_name = f.name
    try:
        tocsv(TABLE, f)
        assert not f.closed  # petl must not close the caller's file
        f.close()
        with open(obj_name, newline='') as g:
            via_object = g.read()

        path_name = obj_name + '.ref'
        tocsv(TABLE, path_name)
        try:
            with open(path_name, newline='') as g:
                via_path = g.read()
            assert via_object == via_path
        finally:
            os.unlink(path_name)
    finally:
        if not f.closed:
            f.close()
        os.unlink(obj_name)


def test_tocsv_stringio():
    sio = io.StringIO(newline='')
    tocsv(TABLE, sio)
    assert sio.getvalue() == 'foo,bar\r\na,1\r\nb,2\r\nc,2\r\n'


def test_tocsv_binary_file_rejected():
    bf = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin')
    name = bf.name
    try:
        with pytest.raises(ArgumentError):
            tocsv(TABLE, bf)
    finally:
        bf.close()
        os.unlink(name)


def test_appendcsv_open_text_file_matches_path_output():
    # a table without a header row is appended as data (the first row is
    # treated as a header by appendcsv/write_header=False, exactly as when a
    # path is supplied)
    path_name = tempfile.mktemp(suffix='.csv')
    obj_name = tempfile.mktemp(suffix='.csv')
    try:
        tocsv(TABLE, path_name)
        appendcsv([['x'], ['d', 4]], path_name)
        with open(path_name, newline='') as g:
            via_path = g.read()

        tocsv(TABLE, obj_name)
        f = open(obj_name, 'a', newline='')
        try:
            appendcsv([['x'], ['d', 4]], f)
        finally:
            f.close()
        with open(obj_name, newline='') as g:
            via_object = g.read()
        assert via_object == via_path
    finally:
        os.unlink(path_name)
        os.unlink(obj_name)
