
#=============================================================================
# TSV Reader Decorator
#=============================================================================

class TSVReaderDecorator(object):
    """Decorator classs for TSV readers"""

    def __init__(self, tsvreader):
        self._tsvreader = tsvreader

    @property
    def fieldnames(self):
        return self._tsvreader.fieldnames

    @property
    def filename(self):
        return self._tsvreader.filename

    @property
    def quoting(self):
        return self._tsvreader.quoting

    @property
    def delimiter(self):
        return self._tsvreader.delimiter

    @property
    def encoding(self):
        return self._tsvreader.encoding

    @property
    def linenumber(self):
        return self._tsvreader.linenumber

    @property
    def strict(self):
        return self._tsvreader.strict

    def close(self):
        self._tsvreader.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._tsvreader)

