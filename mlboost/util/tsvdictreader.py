from .tsvreaderdecorator import TSVReaderDecorator

#=============================================================================
# TSV dict-reader
#=============================================================================

class TSVDictReader(TSVReaderDecorator):
    """
    The TSVDictReader is simply a tsv decorator. Really not much here, it 
    simply return a dict (fieldname : value) instead of a list of values
    everytime it returns a row.
    """

    def __init__(self, tsvreader):

        TSVReaderDecorator.__init__(self, tsvreader)

        if not self._tsvreader.strict:
            raise TSVDictReaderException("Cannot use decorator TSVDictReader "
                                         "on a TSVReader wich have strict = "
                                         "False")

    def __next__(self):
        line = next(self._tsvreader)
        return dict(list(zip(self.fieldnames, line)))

#=============================================================================
# Exception Class
#=============================================================================

class TSVDictReaderException(Exception):
    pass

