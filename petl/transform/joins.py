from __future__ import absolute_import, division, print_function

import itertools
import operator

from petl.comparison import comparable_itemgetter, is_null_key
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.transform.basics import cut, cutout, stack
from petl.transform.dedup import distinct
from petl.transform.sorts import sort
from petl.util.base import Table, asindices, data, header, rowgetter, rowgroupby


def natural_key(left, right):
    # determine key field or fields
    lhdr = header(left)
    lflds = list(map(str, lhdr))
    rhdr = header(right)
    rflds = list(map(str, rhdr))
    key = [f for f in lflds if f in rflds]
    assert len(key) > 0, 'no fields in common'
    if len(key) == 1:
        key = key[0]  # deal with singletons
    return key


def keys_from_args(left, right, key, lkey, rkey):

    if key is lkey is rkey is None:
        # no keys specified, attempt natural join
        lkey = rkey = natural_key(left, right)
    elif key is not None and lkey is rkey is None:
        # common key specified
        lkey = rkey = key
    elif key is None and lkey is not None and rkey is not None:
        # left and right keys specified
        pass
    else:
        raise ArgumentError(
            'bad key arguments: either specify key, or specify both lkey and '
            'rkey, or provide no key/lkey/rkey arguments at all (natural join)'
        )
    return lkey, rkey


def _nonkey_indices(hdr, indices):
    """Return non-key positions, including when keys use negative indexes.

    >>> _nonkey_indices(('id', 'value'), (-1,))
    [0]
    """
    key_indices = {i if i >= 0 else len(hdr) + i for i in indices}
    return [i for i in range(len(hdr)) if i not in key_indices]


def join(left, right, key=None, lkey=None, rkey=None, presorted=False,
         buffersize=None, tempdir=None, cache=True, lprefix=None, rprefix=None,
         match_nulls=False):
    """
    Perform an equi-join on the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [3, 'purple']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square'],
        ...           [4, 'ellipse']]
        >>> table3 = etl.join(table1, table2, key='id')
        >>> table3
        +----+----------+----------+
        | id | colour   | shape    |
        +====+==========+==========+
        |  1 | 'blue'   | 'circle' |
        +----+----------+----------+
        |  3 | 'purple' | 'square' |
        +----+----------+----------+

        >>> # if no key is given, a natural join is tried
        ... table4 = etl.join(table1, table2)
        >>> table4
        +----+----------+----------+
        | id | colour   | shape    |
        +====+==========+==========+
        |  1 | 'blue'   | 'circle' |
        +----+----------+----------+
        |  3 | 'purple' | 'square' |
        +----+----------+----------+

        >>> # note behaviour if the key is not unique in either or both tables
        ... table5 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [1, 'red'],
        ...           [2, 'purple']]
        >>> table6 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [1, 'square'],
        ...           [2, 'ellipse']]
        >>> table7 = etl.join(table5, table6, key='id')
        >>> table7
        +----+----------+-----------+
        | id | colour   | shape     |
        +====+==========+===========+
        |  1 | 'blue'   | 'circle'  |
        +----+----------+-----------+
        |  1 | 'blue'   | 'square'  |
        +----+----------+-----------+
        |  1 | 'red'    | 'circle'  |
        +----+----------+-----------+
        |  1 | 'red'    | 'square'  |
        +----+----------+-----------+
        |  2 | 'purple' | 'ellipse' |
        +----+----------+-----------+

        >>> # compound keys are supported
        ... table8 = [['id', 'time', 'height'],
        ...           [1, 1, 12.3],
        ...           [1, 2, 34.5],
        ...           [2, 1, 56.7]]
        >>> table9 = [['id', 'time', 'weight'],
        ...           [1, 2, 4.5],
        ...           [2, 1, 6.7],
        ...           [2, 2, 8.9]]
        >>> table10 = etl.join(table8, table9, key=['id', 'time'])
        >>> table10
        +----+------+--------+--------+
        | id | time | height | weight |
        +====+======+========+========+
        |  1 |    2 |   34.5 |    4.5 |
        +----+------+--------+--------+
        |  2 |    1 |   56.7 |    6.7 |
        +----+------+--------+--------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    If `match_nulls` is False (the default), rows with null keys do not
    participate in matching. ``None`` and empty or whitespace-only strings
    are treated as null keys. Set `match_nulls` to True to restore the
    legacy behaviour where null keys on both sides are joined together.

    """

    # TODO don't read data twice (occurs if using natural key)
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return JoinView(left, right, lkey=lkey, rkey=rkey,
                    presorted=presorted, buffersize=buffersize, tempdir=tempdir,
                    cache=cache, lprefix=lprefix, rprefix=rprefix,
                    match_nulls=match_nulls)


Table.join = join


class JoinView(Table):

    def __init__(self, left, right, lkey, rkey,
                 presorted=False, leftouter=False, rightouter=False,
                 missing=None, buffersize=None, tempdir=None, cache=True,
                 lprefix=None, rprefix=None, match_nulls=False):
        self.lkey = lkey
        self.rkey = rkey
        self.left = stack(left, missing=missing)
        self.right = stack(right, missing=missing)
        if not presorted:
            self.left = sort(self.left, lkey, buffersize=buffersize,
                             tempdir=tempdir, cache=cache)
            self.right = sort(self.right, rkey, buffersize=buffersize,
                              tempdir=tempdir, cache=cache)
        self.leftouter = leftouter
        self.rightouter = rightouter
        self.missing = missing
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.match_nulls = match_nulls

    def __iter__(self):
        return iterjoin(self.left, self.right, self.lkey, self.rkey,
                        leftouter=self.leftouter, rightouter=self.rightouter,
                        missing=self.missing, lprefix=self.lprefix,
                        rprefix=self.rprefix, match_nulls=self.match_nulls)


def leftjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
             presorted=False, buffersize=None, tempdir=None, cache=True,
             lprefix=None, rprefix=None, match_nulls=False):
    """
    Perform a left outer join on the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [3, 'purple']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square'],
        ...           [4, 'ellipse']]
        >>> table3 = etl.leftjoin(table1, table2, key='id')
        >>> table3
        +----+----------+----------+
        | id | colour   | shape    |
        +====+==========+==========+
        |  1 | 'blue'   | 'circle' |
        +----+----------+----------+
        |  2 | 'red'    | None     |
        +----+----------+----------+
        |  3 | 'purple' | 'square' |
        +----+----------+----------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    # TODO don't read data twice (occurs if using natural key)
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return JoinView(left, right, lkey=lkey, rkey=rkey,
                    presorted=presorted, leftouter=True, rightouter=False,
                    missing=missing, buffersize=buffersize, tempdir=tempdir,
                    cache=cache, lprefix=lprefix, rprefix=rprefix,
                    match_nulls=match_nulls)


Table.leftjoin = leftjoin


def rightjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
              presorted=False, buffersize=None, tempdir=None, cache=True,
              lprefix=None, rprefix=None, match_nulls=False):
    """
    Perform a right outer join on the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [3, 'purple']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square'],
        ...           [4, 'ellipse']]
        >>> table3 = etl.rightjoin(table1, table2, key='id')
        >>> table3
        +----+----------+-----------+
        | id | colour   | shape     |
        +====+==========+===========+
        |  1 | 'blue'   | 'circle'  |
        +----+----------+-----------+
        |  3 | 'purple' | 'square'  |
        +----+----------+-----------+
        |  4 | None     | 'ellipse' |
        +----+----------+-----------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    # TODO don't read data twice (occurs if using natural key)
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return JoinView(left, right, lkey=lkey, rkey=rkey,
                    presorted=presorted, leftouter=False, rightouter=True,
                    missing=missing, buffersize=buffersize,
                    tempdir=tempdir, cache=cache, lprefix=lprefix,
                    rprefix=rprefix, match_nulls=match_nulls)


Table.rightjoin = rightjoin


def outerjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
              presorted=False, buffersize=None, tempdir=None, cache=True,
              lprefix=None, rprefix=None):
    """
    Perform a full outer join on the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [3, 'purple']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square'],
        ...           [4, 'ellipse']]
        >>> table3 = etl.outerjoin(table1, table2, key='id')
        >>> table3
        +----+----------+-----------+
        | id | colour   | shape     |
        +====+==========+===========+
        |  1 | 'blue'   | 'circle'  |
        +----+----------+-----------+
        |  2 | 'red'    | None      |
        +----+----------+-----------+
        |  3 | 'purple' | 'square'  |
        +----+----------+-----------+
        |  4 | None     | 'ellipse' |
        +----+----------+-----------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    # TODO don't read data twice (occurs if using natural key)
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return JoinView(left, right, lkey=lkey, rkey=rkey,
                    presorted=presorted, leftouter=True, rightouter=True,
                    missing=missing, buffersize=buffersize, tempdir=tempdir,
                    cache=cache, lprefix=lprefix, rprefix=rprefix)


Table.outerjoin = outerjoin


def iterjoin(left, right, lkey, rkey, leftouter=False, rightouter=False,
             missing=None, lprefix=None, rprefix=None, match_nulls=False):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)

    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)

    # construct functions to extract key values from both tables
    lgetk = comparable_itemgetter(*lkind)
    rgetk = comparable_itemgetter(*rkind)
    # raw (unwrapped) key getters matching the key shapes emitted by
    # groupby: a scalar for a single key, a tuple for compound keys
    if len(lkind) == 1:
        lgetkraw = operator.itemgetter(lkind[0])
    else:
        lgetkraw = rowgetter(*lkind)
    if len(rkind) == 1:
        rgetkraw = operator.itemgetter(rkind[0])
    else:
        rgetkraw = rowgetter(*rkind)

    # determine indices of non-key fields in the right table
    # (in the output, we only include key fields from the left table - we
    # don't want to duplicate fields)
    rvind = _nonkey_indices(rhdr, rkind)
    rgetv = rowgetter(*rvind)

    # determine the output fields
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [(text_type(lprefix) + text_type(f)) for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([(text_type(rprefix) + text_type(f)) for f in rgetv(rhdr)])
    yield tuple(outhdr)

    # define a function to join two groups of rows
    def joinrows(_lrowgrp, _rrowgrp):
        if _rrowgrp is None:
            for lrow in _lrowgrp:
                outrow = list(lrow)  # start with the left row
                # extend with missing values in place of the right row
                outrow.extend([missing] * len(rvind))
                yield tuple(outrow)
        elif _lrowgrp is None:
            for rrow in _rrowgrp:
                # start with missing values in place of the left row
                outrow = [missing] * len(lhdr)
                # set key values
                for li, ri in zip(lkind, rkind):
                    outrow[li] = rrow[ri]
                # extend with non-key values from the right row
                outrow.extend(rgetv(rrow))
                yield tuple(outrow)
        else:
            _rrowgrp = list(_rrowgrp)  # may need to iterate more than once
            for lrow in _lrowgrp:
                for rrow in _rrowgrp:
                    # start with the left row
                    outrow = list(lrow)
                    # extend with non-key values from the right row
                    outrow.extend(rgetv(rrow))
                    yield tuple(outrow)

    # construct group iterators for both tables
    lgit = itertools.groupby(lit, key=lgetk)
    rgit = itertools.groupby(rit, key=rgetk)

    # Streaming merge state (a mutable dict keeps Python 2 compatibility
    # without nonlocal). Each side has a current group whose first row is
    # stored (groupby groups are one-shot), a "pending" flag saying whether
    # that group still needs processing, and an explicit exhaustion marker.
    state = {'lgrp': None, 'lfirst': None, 'l_pending': False,
             'rgrp': None, 'rfirst': None, 'r_pending': False,
             'l_done': False, 'r_done': False}

    def lgrp_is_null():
        return not match_nulls and state['lfirst'] is not None \
            and is_null_key(lgetkraw(state['lfirst']))

    def rgrp_is_null():
        return not match_nulls and state['rfirst'] is not None \
            and is_null_key(rgetkraw(state['rfirst']))

    def l_rows():
        return itertools.chain([state['lfirst']], state['lgrp'])

    def r_rows():
        return itertools.chain([state['rfirst']], state['rgrp'])

    def advance_left():
        state['l_pending'] = False
        try:
            kval, grp = next(lgit)
        except StopIteration:
            state['l_done'] = True
            return
        state['lkval'] = kval
        state['lgrp'] = grp
        state['lfirst'] = next(iter(grp))
        state['l_pending'] = True

    def advance_right():
        state['r_pending'] = False
        try:
            kval, grp = next(rgit)
        except StopIteration:
            state['r_done'] = True
            return
        state['rkval'] = kval
        state['rgrp'] = grp
        state['rfirst'] = next(iter(grp))
        state['r_pending'] = True

    # pick off initial row groups
    advance_left()
    advance_right()

    while state['l_pending'] and state['r_pending']:
        if lgrp_is_null():
            # null keys never match; keep left null rows only in a left
            # (or full) outer join, then move on
            if leftouter:
                for row in joinrows(l_rows(), None):
                    yield tuple(row)
            advance_left()
        elif rgrp_is_null():
            # null keys never match; keep right null rows only in a right
            # (or full) outer join, then move on
            if rightouter:
                for row in joinrows(None, r_rows()):
                    yield tuple(row)
            advance_right()
        elif state['lkval'] < state['rkval']:
            if leftouter:
                for row in joinrows(l_rows(), None):
                    yield tuple(row)
            advance_left()
        elif state['lkval'] > state['rkval']:
            if rightouter:
                for row in joinrows(None, r_rows()):
                    yield tuple(row)
            advance_right()
        else:
            for row in joinrows(l_rows(), r_rows()):
                yield tuple(row)
            # advance both; each records its own exhaustion, so a group that
            # was just matched is never left pending on the surviving side
            advance_left()
            advance_right()

    # one side is exhausted. The other side's pending group (if any) and all
    # its remaining groups are unprocessed; emit them under the outer-join
    # flags. Null groups never matched, so they are emitted like any other
    # leftover group.
    if leftouter and state['l_pending']:
        for row in joinrows(l_rows(), None):
            yield tuple(row)
        for _, lgrp2 in lgit:
            for row in joinrows(lgrp2, None):
                yield tuple(row)

    if rightouter and state['r_pending']:
        for row in joinrows(None, r_rows()):
            yield tuple(row)
        for _, rgrp2 in rgit:
            for row in joinrows(None, rgrp2):
                yield tuple(row)


def crossjoin(*tables, **kwargs):
    """
    Form the cartesian product of the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square']]
        >>> table3 = etl.crossjoin(table1, table2)
        >>> table3
        +----+--------+----+----------+
        | id | colour | id | shape    |
        +====+========+====+==========+
        |  1 | 'blue' |  1 | 'circle' |
        +----+--------+----+----------+
        |  1 | 'blue' |  3 | 'square' |
        +----+--------+----+----------+
        |  2 | 'red'  |  1 | 'circle' |
        +----+--------+----+----------+
        |  2 | 'red'  |  3 | 'square' |
        +----+--------+----+----------+

    If `prefix` is `True` then field names in the output table header will be
    prefixed by the index of the input table.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    return CrossJoinView(*tables, **kwargs)


Table.crossjoin = crossjoin


class CrossJoinView(Table):

    def __init__(self, *sources, **kwargs):
        self.prefix = kwargs.get('prefix', False)
        self.missing = kwargs.get('missing', None)
        self.sources = [stack(source, missing=self.missing) for source in sources]

    def __iter__(self):
        return itercrossjoin(self.sources, self.prefix)


def itercrossjoin(sources, prefix):

    # construct fields
    outhdr = list()
    for i, s in enumerate(sources):
        if prefix:
            # use one-based numbering
            outhdr.extend([text_type(i+1) + '_' + text_type(f) for f in header(s)])
        else:
            outhdr.extend(header(s))
    yield tuple(outhdr)

    datasrcs = [data(src) for src in sources]
    for prod in itertools.product(*datasrcs):
        outrow = list()
        for row in prod:
            outrow.extend(row)
        yield tuple(outrow)


def antijoin(left, right, key=None, lkey=None, rkey=None, presorted=False,
             buffersize=None, tempdir=None, cache=True, match_nulls=False):
    """
    Return rows from the `left` table where the key value does not occur in
    the `right` table. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [0, 'black'],
        ...           [1, 'blue'],
        ...           [2, 'red'],
        ...           [4, 'yellow'],
        ...           [5, 'white']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square']]
        >>> table3 = etl.antijoin(table1, table2, key='id')
        >>> table3
        +----+----------+
        | id | colour   |
        +====+==========+
        |  0 | 'black'  |
        +----+----------+
        |  2 | 'red'    |
        +----+----------+
        |  4 | 'yellow' |
        +----+----------+
        |  5 | 'white'  |
        +----+----------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return AntiJoinView(left=left, right=right, lkey=lkey, rkey=rkey,
                        presorted=presorted, buffersize=buffersize,
                        tempdir=tempdir, cache=cache,
                        match_nulls=match_nulls)


Table.antijoin = antijoin


class AntiJoinView(Table):

    def __init__(self, left, right, lkey, rkey, presorted=False,
                 buffersize=None, tempdir=None, cache=True,
                 match_nulls=False):
        if presorted:
            self.left = left
            self.right = right
        else:
            self.left = sort(left, lkey, buffersize=buffersize,
                             tempdir=tempdir, cache=cache)
            self.right = sort(right, rkey, buffersize=buffersize,
                              tempdir=tempdir, cache=cache)
        self.lkey = lkey
        self.rkey = rkey
        self.match_nulls = match_nulls

    def __iter__(self):
        return iterantijoin(self.left, self.right, self.lkey, self.rkey,
                            match_nulls=self.match_nulls)


def iterantijoin(left, right, lkey, rkey, match_nulls=False):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)
    yield tuple(lhdr)

    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)

    # construct functions to extract key values from both tables
    lgetk = comparable_itemgetter(*lkind)
    rgetk = comparable_itemgetter(*rkind)
    if len(lkind) == 1:
        lgetkraw = operator.itemgetter(lkind[0])
    else:
        lgetkraw = rowgetter(*lkind)
    if len(rkind) == 1:
        rgetkraw = operator.itemgetter(rkind[0])
    else:
        rgetkraw = rowgetter(*rkind)

    # construct group iterators for both tables
    lgit = itertools.groupby(lit, key=lgetk)
    rgit = itertools.groupby(rit, key=rgetk)

    # Streaming merge state (mutable dict for Python 2 compatibility).
    state = {'lgrp': None, 'lfirst': None, 'l_pending': False,
             'rgrp': None, 'rfirst': None, 'r_pending': False,
             'l_done': False, 'r_done': False}

    def lgrp_is_null():
        return not match_nulls and state['lfirst'] is not None \
            and is_null_key(lgetkraw(state['lfirst']))

    def rgrp_is_null():
        return not match_nulls and state['rfirst'] is not None \
            and is_null_key(rgetkraw(state['rfirst']))

    def l_rows():
        return itertools.chain([state['lfirst']], state['lgrp'])

    def advance_left():
        state['l_pending'] = False
        try:
            kval, grp = next(lgit)
        except StopIteration:
            state['l_done'] = True
            return
        state['lkval'] = kval
        state['lgrp'] = grp
        state['lfirst'] = next(iter(grp))
        state['l_pending'] = True

    def advance_right():
        state['r_pending'] = False
        try:
            kval, grp = next(rgit)
        except StopIteration:
            state['r_done'] = True
            return
        state['rkval'] = kval
        state['rgrp'] = grp
        state['rfirst'] = next(iter(grp))
        state['r_pending'] = True

    # pick off initial row groups
    advance_left()
    advance_right()

    while state['l_pending'] and state['r_pending']:
        if lgrp_is_null():
            # null keys never match, so the left rows are differences
            for row in l_rows():
                yield tuple(row)
            advance_left()
        elif rgrp_is_null():
            # right-side null rows cannot match anything, skip them
            advance_right()
        elif state['lkval'] < state['rkval']:
            for row in l_rows():
                yield tuple(row)
            advance_left()
        elif state['lkval'] > state['rkval']:
            # advance right
            advance_right()
        else:
            # matching keys: the left rows are not differences, advance both
            advance_left()
            advance_right()

    # left groups remaining when (or after) the right side is exhausted are
    # all differences, including any null-key groups
    if state['l_pending']:
        for row in l_rows():
            yield tuple(row)
    for _, lgrp2 in lgit:
        for row in lgrp2:
            yield tuple(row)


def lookupjoin(left, right, key=None, lkey=None, rkey=None, missing=None,
               presorted=False, buffersize=None, tempdir=None, cache=True,
               lprefix=None, rprefix=None):
    """
    Perform a left join, but where the key is not unique in the right-hand
    table, arbitrarily choose the first row and ignore others. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'color', 'cost'],
        ...           [1, 'blue', 12],
        ...           [2, 'red', 8],
        ...           [3, 'purple', 4]]
        >>> table2 = [['id', 'shape', 'size'],
        ...           [1, 'circle', 'big'],
        ...           [1, 'circle', 'small'],
        ...           [2, 'square', 'tiny'],
        ...           [2, 'square', 'big'],
        ...           [3, 'ellipse', 'small'],
        ...           [3, 'ellipse', 'tiny']]
        >>> table3 = etl.lookupjoin(table1, table2, key='id')
        >>> table3
        +----+----------+------+-----------+---------+
        | id | color    | cost | shape     | size    |
        +====+==========+======+===========+=========+
        |  1 | 'blue'   |   12 | 'circle'  | 'big'   |
        +----+----------+------+-----------+---------+
        |  2 | 'red'    |    8 | 'square'  | 'tiny'  |
        +----+----------+------+-----------+---------+
        |  3 | 'purple' |    4 | 'ellipse' | 'small' |
        +----+----------+------+-----------+---------+

    See also :func:`petl.transform.joins.leftjoin`.

    .. versionchanged:: 1.7.16
        To ensure correct results for tables with uneven rows, tables will be
        squared up before joining to ensure correct results.

    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return LookupJoinView(left, right, lkey, rkey, presorted=presorted,
                          missing=missing, buffersize=buffersize,
                          tempdir=tempdir, cache=cache,
                          lprefix=lprefix, rprefix=rprefix)


Table.lookupjoin = lookupjoin


class LookupJoinView(Table):

    def __init__(self, left, right, lkey, rkey, presorted=False, missing=None,
                 buffersize=None, tempdir=None, cache=True,
                 lprefix=None, rprefix=None):
        self.left = stack(left, missing=missing)
        self.right = stack(right, missing=missing)
        if not presorted:
            self.left = sort(self.left, lkey, buffersize=buffersize,
                             tempdir=tempdir, cache=cache)
            self.right = sort(self.right, rkey, buffersize=buffersize,
                              tempdir=tempdir, cache=cache)
        self.lkey = lkey
        self.rkey = rkey
        self.missing = missing
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        return iterlookupjoin(self.left, self.right, self.lkey, self.rkey,
                              missing=self.missing, lprefix=self.lprefix,
                              rprefix=self.rprefix)


def iterlookupjoin(left, right, lkey, rkey, missing=None, lprefix=None,
                   rprefix=None):
    lit = iter(left)
    rit = iter(right)

    lhdr = next(lit)
    rhdr = next(rit)

    # determine indices of the key fields in left and right tables
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)

    # construct functions to extract key values from both tables
    lgetk = comparable_itemgetter(*lkind)
    rgetk = comparable_itemgetter(*rkind)

    # determine indices of non-key fields in the right table
    # (in the output, we only include key fields from the left table - we
    # don't want to duplicate fields)
    rvind = _nonkey_indices(rhdr, rkind)
    rgetv = rowgetter(*rvind)

    # determine the output fields
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [(text_type(lprefix) + text_type(f)) for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([(text_type(rprefix) + text_type(f)) for f in rgetv(rhdr)])
    yield tuple(outhdr)

    # define a function to join two groups of rows
    def joinrows(_lrowgrp, _rrowgrp):
        if _rrowgrp is None:
            for lrow in _lrowgrp:
                outrow = list(lrow)  # start with the left row
                # extend with missing values in place of the right row
                outrow.extend([missing] * len(rvind))
                yield tuple(outrow)
        else:
            rrow = next(iter(_rrowgrp))  # pick first arbitrarily
            for lrow in _lrowgrp:
                # start with the left row
                outrow = list(lrow)
                # extend with non-key values from the right row
                outrow.extend(rgetv(rrow))
                yield tuple(outrow)

    # construct group iterators for both tables
    lgit = itertools.groupby(lit, key=lgetk)
    rgit = itertools.groupby(rit, key=rgetk)
    lrowgrp = []

    # whether lrowgrp holds a group that has been picked off lgit but not yet
    # yielded; tracked explicitly because no key value can stand in for
    # "iterator exhausted" - None is itself a valid key
    lgrp_pending = False

    # loop until *either* of the iterators is exhausted
    try:

        # pick off initial row groups
        lkval, lrowgrp = next(lgit)
        lgrp_pending = True
        rkval, rrowgrp = next(rgit)

        while True:
            if lkval < rkval:
                for row in joinrows(lrowgrp, None):
                    yield tuple(row)
                # advance left
                lgrp_pending = False
                lkval, lrowgrp = next(lgit)
                lgrp_pending = True
            elif lkval > rkval:
                # advance right
                rkval, rrowgrp = next(rgit)
            else:
                for row in joinrows(lrowgrp, rrowgrp):
                    yield tuple(row)
                # advance both
                lgrp_pending = False
                lkval, lrowgrp = next(lgit)
                lgrp_pending = True
                rkval, rrowgrp = next(rgit)

    except StopIteration:
        pass

    # make sure any left rows remaining are yielded
    if lgrp_pending:
        # yield anything that got left hanging
        for row in joinrows(lrowgrp, None):
            yield tuple(row)
    # yield the rest
    for lkval, lrowgrp in lgit:
        for row in joinrows(lrowgrp, None):
            yield tuple(row)


def unjoin(table, value, key=None, autoincrement=(1, 1), presorted=False,
           buffersize=None, tempdir=None, cache=True):
    """
    Split a table into two tables by reversing an inner join. E.g.::

        >>> import petl as etl
        >>> # join key is present in the table
        ... table1 = (('foo', 'bar', 'baz'),
        ...           ('A', 1, 'apple'),
        ...           ('B', 1, 'apple'),
        ...           ('C', 2, 'orange'))
        >>> table2, table3 = etl.unjoin(table1, 'baz', key='bar')
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   1 |
        +-----+-----+
        | 'B' |   1 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+

        >>> table3
        +-----+----------+
        | bar | baz      |
        +=====+==========+
        |   1 | 'apple'  |
        +-----+----------+
        |   2 | 'orange' |
        +-----+----------+

        >>> # an integer join key can also be reconstructed
        ... table4 = (('foo', 'bar'),
        ...           ('A', 'apple'),
        ...           ('B', 'apple'),
        ...           ('C', 'orange'))
        >>> table5, table6 = etl.unjoin(table4, 'bar')
        >>> table5
        +-----+--------+
        | foo | bar_id |
        +=====+========+
        | 'A' |      1 |
        +-----+--------+
        | 'B' |      1 |
        +-----+--------+
        | 'C' |      2 |
        +-----+--------+

        >>> table6
        +----+----------+
        | id | bar      |
        +====+==========+
        |  1 | 'apple'  |
        +----+----------+
        |  2 | 'orange' |
        +----+----------+

    The `autoincrement` parameter controls how an integer join key is
    reconstructed, and should be a tuple of (`start`, `step`).

    """

    if key is None:
        # first sort the table by the value field
        if presorted:
            tbl_sorted = table
        else:
            tbl_sorted = sort(table, value, buffersize=buffersize,
                              tempdir=tempdir, cache=cache)
        # on the left, return the original table but with the value field
        # replaced by an incrementing integer
        left = ConvertToIncrementingCounterView(tbl_sorted, value,
                                                autoincrement)
        # on the right, return a new table with distinct values from the
        # given field
        right = EnumerateDistinctView(tbl_sorted, value, autoincrement)
    else:
        # on the left, return distinct rows from the original table
        # with the value field cut out
        left = distinct(cutout(table, value))
        # on the right, return distinct rows from the original table
        # with all fields but the key and value cut out
        right = distinct(cut(table, key, value))
    return left, right


class ConvertToIncrementingCounterView(Table):

    def __init__(self, tbl, value, autoincrement):
        self.table = tbl
        self.value = value
        self.autoincrement = autoincrement

    def __iter__(self):
        it = iter(self.table)
        hdr = next(it)
        table = itertools.chain([hdr], it)
        value = self.value
        vidx = hdr.index(value)
        outhdr = list(hdr)
        outhdr[vidx] = '%s_id' % value
        yield tuple(outhdr)
        offset, multiplier = self.autoincrement
        for n, (_, group) in enumerate(rowgroupby(table, value)):
            for row in group:
                outrow = list(row)
                outrow[vidx] = (n * multiplier) + offset
                yield tuple(outrow)


Table.unjoin = unjoin


class EnumerateDistinctView(Table):

    def __init__(self, tbl, value, autoincrement):
        self.table = tbl
        self.value = value
        self.autoincrement = autoincrement

    def __iter__(self):
        offset, multiplier = self.autoincrement
        yield ('id', self.value)
        for n, (v, _) in enumerate(rowgroupby(self.table, self.value)):
            yield ((n * multiplier) + offset, v)
