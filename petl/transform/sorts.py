from __future__ import absolute_import, print_function, division


import os
import re
import heapq
from decimal import Decimal, InvalidOperation
from tempfile import NamedTemporaryFile
import itertools
import logging
from collections import namedtuple
from functools import partial
import operator
from petl.compat import pickle, next, text_type, binary_type


import petl.config as config
from petl.comparison import is_null_key
from petl.util.base import Table, asindices


logger = logging.getLogger(__name__)
warning = logger.warning
info = logger.info
debug = logger.debug


# strings that are plain (possibly signed, decimal or exponent notation)
# numbers, e.g. '010', '-2.5', '1e3'
_NUMBER_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def _parsed_number(value):
    """Return a Decimal for numeric values and pure-number strings, else
    None. Floats are converted via their string representation so that
    Decimal(str(x)) arithmetic is used throughout."""
    if isinstance(value, bool):
        # bool is a subclass of int but must not be treated as an amount
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(text_type(value))
    if isinstance(value, (text_type, binary_type)):
        if isinstance(value, binary_type):
            try:
                value = value.decode('ascii')
            except UnicodeDecodeError:
                return None
        if _NUMBER_RE.match(value.strip()):
            try:
                return Decimal(value.strip())
            except InvalidOperation:
                return None
    return None


class ReconcileKey(object):
    """Sort key implementing the reconciliation ordering rules:

    * null values (None and empty/whitespace-only strings) sort last, in
      both ascending and descending order;
    * numbers and pure-number strings compare by numeric value;
    * everything else compares by text code points;
    * equal keys compare equal so the original row order is preserved.

    The key defines one fixed total order (nulls last). Descending output
    is obtained by reversing the *non-null* ordering at the output layer
    while leaving nulls last, see :func:`reconcile_itemgetter`.

    The ordering never raises TypeError.
    """

    __slots__ = ['obj', 'num']

    def __init__(self, obj):
        self.obj = obj
        self.num = _parsed_number(obj)

    # category ranks: null last, then numbers, then everything else
    def _rank(self):
        return _rank_of(self.obj, self.num)

    def _cmp(self, other):
        if isinstance(other, ReconcileKey):
            other_obj = other.obj
            other_num = other.num
        else:
            other_obj = other
            other_num = _parsed_number(other)
        ra, rb = self._rank(), _rank_of(other_obj, other_num)
        if ra != rb:
            return -1 if ra < rb else 1
        if ra == 2:
            # all nulls are equal, so null rows keep their input order
            return 0
        if ra == 0:
            if self.num == other_num:
                return 0
            return -1 if self.num < other_num else 1
        # text comparison by code point; fall back to type names if the
        # native comparison fails
        sa = self.obj if isinstance(self.obj, text_type) \
            else text_type(self.obj)
        sb = other_obj if isinstance(other_obj, text_type) \
            else text_type(other_obj)
        try:
            if sa == sb:
                return 0
            return -1 if sa < sb else 1
        except TypeError:
            ta, tb = type(self.obj).__name__, type(other_obj).__name__
            if ta == tb:
                return 0
            return -1 if ta < tb else 1

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __eq__(self, other):
        return self._cmp(other) == 0

    def __le__(self, other):
        return self < other or self == other

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return not (self < other or self == other)

    def __ge__(self, other):
        return not (self < other)

    def __repr__(self):
        return 'ReconcileKey(%r)' % (self.obj,)


class ReconcileKeyReverse(ReconcileKey):
    """Descending variant of :class:`ReconcileKey`: non-null values are
    ordered in reverse, while null values still sort last and stay
    stable."""

    __slots__ = ()

    def __lt__(self, other):
        c = self._cmp(other)
        if c == 0:
            return False
        if self._rank() == 2:
            # a null is never less than anything, so it stays last
            return False
        if isinstance(other, ReconcileKey) and other._rank() == 2:
            # any non-null value precedes a null
            return True
        return c > 0

    def __eq__(self, other):
        return self._cmp(other) == 0

    def __le__(self, other):
        return self < other or self == other

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        # derive strictly from the base relations to keep a total order
        return not (self < other or self == other)

    def __ge__(self, other):
        return not (self < other)


def _rank_of(obj, num):
    if is_null_key(obj):
        return 2
    if num is not None:
        return 0
    return 1


def reconcile_itemgetter(*indices, **kwargs):
    """Item getter returning reconciliation sort keys. A single field index
    yields a scalar key; multiple indices yield a tuple of keys so compound
    keys sort lexicographically under the same rules. When `reverse` is
    True non-null values order descending while nulls remain last."""
    reverse = kwargs.get('reverse', False)
    keyclass = ReconcileKeyReverse if reverse else ReconcileKey
    getter = operator.itemgetter(*indices)
    getter_with_default = _itemgetter_with_default(*indices)

    def _getter_with_fallback(obj):
        try:
            value = getter(obj)
        except (IndexError, KeyError):
            value = getter_with_default(obj)
        if len(indices) == 1:
            return keyclass(value)
        return tuple(keyclass(v) for v in value)

    return _getter_with_fallback


def _itemgetter_with_default(*args):
    """itemgetter filling missing values with None instead of raising
    IndexError or KeyError"""
    def _get_default(obj, item, default):
        try:
            return obj[item]
        except (IndexError, KeyError):
            return default
    if len(args) == 1:
        return partial(_get_default, item=args[0], default=None)
    return lambda obj: tuple(_get_default(obj, item=item, default=None)
                             for item in args)


def sort(table, key=None, reverse=False, buffersize=None, tempdir=None,
         cache=True):
    """
    Sort the table. Field names or indices (from zero) can be used to specify
    the key. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['C', 2],
        ...           ['A', 9],
        ...           ['A', 6],
        ...           ['F', 1],
        ...           ['D', 10]]
        >>> table2 = etl.sort(table1, 'foo')
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   9 |
        +-----+-----+
        | 'A' |   6 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+

        >>> # sorting by compound key is supported
        ... table3 = etl.sort(table1, key=['foo', 'bar'])
        >>> table3
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |   9 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+

        >>> # if no key is specified, the default is a lexical sort
        ... table4 = etl.sort(table1)
        >>> table4
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |   9 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+

        >>> # sorting is stable, so fields can be sorted in different
        ... # directions by chaining sorts, starting with the last field
        ... table5 = etl.sort(table1, 'bar').sort('foo', reverse=True)
        >>> table5
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'F' |   1 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |   9 |
        +-----+-----+

    The `buffersize` argument should be an `int` or `None`.

    If the number of rows in the table is less than `buffersize`, the table
    will be sorted in memory. Otherwise, the table is sorted in chunks of
    no more than `buffersize` rows, each chunk is written to a temporary file,
    and then a merge sort is performed on the temporary files.

    If `buffersize` is `None`, the value of
    `petl.config.sort_buffersize` will be used. By default this is
    set to 100000 rows, but can be changed, e.g.::

        >>> import petl.config
        >>> petl.config.sort_buffersize = 500000

    If `petl.config.sort_buffersize` is set to `None`, this forces
    all sorting to be done entirely in memory.

    By default the results of the sort will be cached, and so a second pass over
    the sorted table will yield rows from the cache and will not repeat the
    sort operation. To turn off caching, set the `cache` argument to `False`.

    """

    return SortView(table, key=key, reverse=reverse, buffersize=buffersize,
                    tempdir=tempdir, cache=cache)


Table.sort = sort


def _iterchunk(fn):
    # reopen so iterators from file cache are independent
    debug('iterchunk, opening %s' % fn)
    with open(fn, 'rb') as f:
        try:
            while True:
                yield pickle.load(f)
        except EOFError:
            pass
    debug('end of iterchunk, closed %s' % fn)


class _Keyed(namedtuple('Keyed', ['key', 'obj'])):
    #  Override default behavior of namedtuple comparisons, only keys need to be compared for heapmerge
    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key < other.key

    def __le__(self, other):
        return self.key <= other.key

    def __ne__(self, other):
        return self.key != other.key

    def __gt__(self, other):
        return self.key > other.key

    def __ge__(self, other):
        return self.key >= other.key


def _heapqmergesorted(key=None, *iterables):
    """Return a single iterator over the given iterables, sorted by the
    given `key` function, assuming the input iterables are already sorted by
    the same function. (I.e., the merge part of a general merge sort.) Uses
    :func:`heapq.merge` for the underlying implementation."""

    if key is None:
        keyed_iterables = iterables
        for element in heapq.merge(*keyed_iterables):
            yield element
    else:
        keyed_iterables = [(_Keyed(key(obj), obj) for obj in iterable)
                           for iterable in iterables]
        for element in heapq.merge(*keyed_iterables):
            yield element.obj


def _shortlistmergesorted(key=None, reverse=False, *iterables):
    """Return a single iterator over the given iterables, sorted by the
    given `key` function, assuming the input iterables are already sorted by
    the same function. (I.e., the merge part of a general merge sort.) Uses
    :func:`min` (or :func:`max` if ``reverse=True``) for the underlying
    implementation."""

    if reverse:
        op = max
    else:
        op = min
    if key is not None:
        opkwargs = {'key': key}
    else:
        opkwargs = dict()
    # populate initial shortlist
    # (remember some iterables might be empty)
    iterators = list()
    shortlist = list()
    for iterable in iterables:
        it = iter(iterable)
        try:
            first = next(it)
            iterators.append(it)
            shortlist.append(first)
        except StopIteration:
            pass
    # do the mergesort
    while iterators:
        nxt = op(shortlist, **opkwargs)
        yield nxt
        nextidx = shortlist.index(nxt)
        try:
            shortlist[nextidx] = next(iterators[nextidx])
        except StopIteration:
            del shortlist[nextidx]
            del iterators[nextidx]


def _mergesorted(key=None, reverse=False, *iterables):
    # N.B., I've used heapq for normal merge sort and shortlist merge sort for
    # reverse merge sort because I've assumed that heapq.merge is faster and
    # so is preferable but it doesn't support reverse sorting so the shortlist
    # merge sort has to be used for reverse sorting. Some casual profiling
    # suggests there isn't much between the two in terms of speed, but might be
    # worth profiling more carefully

    if reverse:
        return _shortlistmergesorted(key, True, *iterables)
    else:
        return _heapqmergesorted(key, *iterables)


class SortView(Table):
    def __init__(self, source, key=None, reverse=False, buffersize=None,
                 tempdir=None, cache=True):
        self.source = source
        self.key = key
        self.reverse = reverse
        if buffersize is None:
            self.buffersize = config.sort_buffersize
        else:
            self.buffersize = buffersize
        self.tempdir = tempdir
        self._cache = cache
        self._hdrcache = None
        self._memcache = None
        self._filecache = None
        self._getkey = None

    def clearcache(self):
        debug('clear cache')
        self._hdrcache = None
        self._memcache = None
        self._filecache = None
        self._getkey = None

    def __iter__(self):
        source = self.source
        key = self.key
        reverse = self.reverse
        if self._cache and self._memcache is not None:
            return self._iterfrommemcache()
        elif self._cache and self._filecache is not None:
            return self._iterfromfilecache()
        else:
            return self._iternocache(source, key, reverse)

    def _iterfrommemcache(self):
        debug('iterate from memory cache')
        yield tuple(self._hdrcache)
        for row in self._memcache:
            yield tuple(row)

    def _iterfromfilecache(self):
        # create a reference to the filecache here, so cleanup happens in the
        # correct order
        filecache = self._filecache
        filenames = list(map(operator.attrgetter('name'), filecache))
        debug('iterate from file cache: %r', filenames)
        yield tuple(self._hdrcache)
        chunkiters = [_iterchunk(fn) for fn in filenames]
        # the cached key already encodes the direction, so merge on its
        # natural order
        rows = _mergesorted(self._getkey, False, *chunkiters)
        try:
            for row in rows:
                yield tuple(row)
        finally:
            debug('attempt cleanup from generator')
            # N.B., need to ensure that any open files are closed **before**
            # temporary files are deleted, as deletion will fail on Windows
            # if file is in use (i.e., still open)
            del chunkiters
            del rows
            del filecache
            debug('exiting generator')

    def _iternocache(self, source, key, reverse):
        debug('iterate without cache')
        self.clearcache()
        it = iter(source)

        try:
            hdr = next(it)
        except StopIteration:
            if key is None:
                return  # nothing to do on a table without headers
            hdr = []
        yield tuple(hdr)

        if key is not None:
            # convert field selection into field indices
            indices = asindices(hdr, key)
        else:
            indices = range(len(hdr))
        # now use field indices to construct a _getkey function.
        # reconciliation ordering: nulls last, numeric-aware comparison,
        # never raises TypeError. The key itself carries the direction
        # (nulls stay last when reversed), so rows are sorted without the
        # built-in reverse flag and merged on the same key order.
        getkey = reconcile_itemgetter(*indices, reverse=reverse)

        # TODO support native comparison

        # initialise the first chunk
        rows = list(itertools.islice(it, 0, self.buffersize))
        rows.sort(key=getkey)

        # have we exhausted the source iterator?
        if self.buffersize is None or len(rows) < self.buffersize:
            # yes, table fits within sort buffer

            if self._cache:
                debug('caching mem')
                self._hdrcache = hdr
                self._memcache = rows
                # actually not needed to iterate from memcache
                self._getkey = getkey

            for row in rows:
                yield tuple(row)

        else:
            # no, table is too big, need to sort in chunks

            chunkfiles = []

            while rows:

                # dump the chunk
                with NamedTemporaryFile(dir=self.tempdir, delete=False,
                                        mode='wb') as f:
                    # N.B., we **don't** want the file to be deleted on close,
                    # but we **do** want the file to be deleted when self
                    # is garbage collected, or when the program exits. When
                    # all references to the wrapper are gone, the file should
                    # get deleted.
                    wrapper = _NamedTempFileDeleteOnGC(f.name)
                    debug('created temporary chunk file %s' % f.name)
                    for row in rows:
                        pickle.dump(row, f, protocol=-1)
                    f.flush()
                    chunkfiles.append(wrapper)

                # grab the next chunk
                rows = list(itertools.islice(it, 0, self.buffersize))
                rows.sort(key=getkey)

            if self._cache:
                debug('caching files')
                self._hdrcache = hdr
                self._filecache = chunkfiles
                self._getkey = getkey

            chunkiters = [_iterchunk(f.name) for f in chunkfiles]
            # the key already encodes the direction (including nulls-last
            # under reverse), so always merge ascending on the key order
            for row in _mergesorted(getkey, False, *chunkiters):
                yield tuple(row)


class _NamedTempFileDeleteOnGC(object):

    def __init__(self, name):
        self.name = name

    def delete(self, unlink=os.unlink, log=logger.debug):
        name = self.name
        try:
            log('deleting %s' % name)
            unlink(name)
        except Exception as e:
            log('exception deleting %s: %s' % (name, e))
            raise
        else:
            log('deleted %s' % name)

    def __del__(self):
        self.delete()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def mergesort(*tables, **kwargs):
    """
    Combine multiple input tables into one sorted output table. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['A', 9],
        ...           ['C', 2],
        ...           ['D', 10],
        ...           ['A', 6],
        ...           ['F', 1]]
        >>> table2 = [['foo', 'bar'],
        ...           ['B', 3],
        ...           ['D', 10],
        ...           ['A', 10],
        ...           ['F', 4]]
        >>> table3 = etl.mergesort(table1, table2, key='foo')
        >>> table3.lookall()
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   9 |
        +-----+-----+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |  10 |
        +-----+-----+
        | 'B' |   3 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+
        | 'F' |   4 |
        +-----+-----+

    If the input tables are already sorted by the given key, give
    ``presorted=True`` as a keyword argument.

    This function is equivalent to concatenating the input tables using
    :func:`cat` then sorting, however this function will typically be more
    efficient, especially if the input tables are presorted.

    Keyword arguments:

    key : string or tuple of strings, optional
        Field name or tuple of fields to sort by (defaults to `None` lexical
        sort)
    reverse : bool, optional
        `True` if sort in reverse (descending) order (defaults to `False`)
    presorted : bool, optional
        `True` if inputs are already sorted by the given key (defaults to
        `False`)
    missing : object
        Value to fill with when input tables have different fields (defaults to
        `None`)
    header : sequence of strings, optional
        Specify a fixed header for the output table
    buffersize : int, optional
        Limit the number of rows in memory per input table when inputs are not
        presorted

    """

    return MergeSortView(tables, **kwargs)


Table.mergesort = mergesort


class MergeSortView(Table):
    def __init__(self, tables, key=None, reverse=False, presorted=False,
                 missing=None, header=None, buffersize=None, tempdir=None,
                 cache=True):
        self.key = key
        if presorted:
            self.tables = tables
        else:
            self.tables = [sort(t, key=key, reverse=reverse,
                                buffersize=buffersize, tempdir=tempdir,
                                cache=cache)
                           for t in tables]
        self.missing = missing
        self._header = header
        self.reverse = reverse

    def __iter__(self):
        return itermergesort(self.tables, self.key, self._header, self.missing,
                             self.reverse)


def itermergesort(sources, key, header, missing, reverse):
    # first need to standardise headers of all input tables
    # borrow this from itercat - TODO remove code smells

    its = [iter(t) for t in sources]
    src_hdrs = []
    for it in its:
        try:
            src_hdrs.append(next(it))
        except StopIteration:
            src_hdrs.append([])

    if header is None:
        # determine output fields by gathering all fields found in the sources
        outhdr = list()
        for hdr in src_hdrs:
            for f in list(map(text_type, hdr)):
                if f not in outhdr:
                    # add any new fields as we find them
                    outhdr.append(f)
    else:
        # predetermined output fields
        outhdr = header
    yield tuple(outhdr)

    def _standardisedata(it, hdr, ofs):
        flds = list(map(text_type, hdr))
        # now construct and yield the data rows
        for _row in it:
            try:
                # should be quickest to do this way
                yield tuple(_row[flds.index(fo)] if fo in flds else missing
                            for fo in ofs)
            except IndexError:
                # handle short rows
                outrow = [missing] * len(ofs)
                for i, fi in enumerate(flds):
                    try:
                        outrow[ofs.index(fi)] = _row[i]
                    except IndexError:
                        pass  # be relaxed about short rows
                yield tuple(outrow)

    # wrap all iterators to standardise fields
    sits = [_standardisedata(it, hdr, outhdr)
            for hdr, it in zip(src_hdrs, its)]

    # now determine key function
    getkey = None
    if key is not None:
        # convert field selection into field indices
        indices = asindices(outhdr, key)
        # reconciliation ordering: nulls last, numeric-aware comparison,
        # never raises TypeError
        getkey = reconcile_itemgetter(*indices, reverse=reverse)

    # OK, do the merge sort. The key encodes the direction (nulls stay last
    # when reversed), so the merge itself is always ascending on key order.
    for row in _shortlistmergesorted(getkey, False, *sits):
        yield row


def issorted(table, key=None, reverse=False, strict=False):
    """
    Return True if the table is ordered (i.e., sorted) by the given key. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['a', 1, True],
        ...           ['b', 3, True],
        ...           ['b', 2]]
        >>> etl.issorted(table1, key='foo')
        True
        >>> etl.issorted(table1, key='bar')
        False
        >>> etl.issorted(table1, key='foo', strict=True)
        False
        >>> etl.issorted(table1, key='foo', reverse=True)
        False

    """

    # determine the operator to use when comparing whole rows (no key given)
    if reverse and strict:
        row_op = operator.lt
    elif reverse:
        row_op = operator.le
    elif strict:
        row_op = operator.gt
    else:
        row_op = operator.ge

    it = iter(table)
    try:
        flds = [text_type(f) for f in next(it)]
    except StopIteration:
        flds = []
    if key is not None:
        # use the same reconciliation ordering as sort() so a table produced
        # by sort() is reported as sorted (nulls last, numeric-aware). The
        # reverse key already encodes the direction, so the comparison
        # operators below are the ascending (non-decreasing) ones.
        getkey = reconcile_itemgetter(*asindices(flds, key), reverse=reverse)
        key_op = operator.gt if strict else operator.ge
    try:
        prev = next(it)
    except StopIteration:
        return True  # no data rows
    if key is None:
        for curr in it:
            if not row_op(curr, prev):
                return False
            prev = curr
    else:
        prevkey = getkey(prev)
        for curr in it:
            currkey = getkey(curr)
            if not key_op(currkey, prevkey):
                return False
            prevkey = currkey
    return True


Table.issorted = issorted
