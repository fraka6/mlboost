import re
import csv
import codecs
import gzip

import sys
csv.field_size_limit(sys.maxsize)

#=============================================================================

QUOTE_NONE='none'
QUOTE_MINIMAL='minimal'
QUOTE_NONNUMERIC='nonnumeric'
QUOTE_ALL='all'
QUOTE_NCS='ncs'
QUOTE_CHOICES = [QUOTE_NONE, QUOTE_MINIMAL, QUOTE_NONNUMERIC, QUOTE_ALL, QUOTE_NCS]

#=============================================================================

def is_compressed(fileobj):
    if fileobj.name.endswith(".gz"):
        return True
    else:
        return False

#=============================================================================
# The basic class for all the other tsv reader
#=============================================================================

class TSVReader(object):
    """
    A single file TSV reader. The only non-keyword argument for the
    constructor is a File object.
    """

    def __init__(self, tsvfile, gunzip=False, delimiter='\t',
                 encoding='utf-8', quoting=QUOTE_NONE, strict=True,
                 autodetect_compressed_file=True, fieldnames=None):

        if quoting == QUOTE_NCS:
            if delimiter != '\t':
                raise TSVReaderException("when quoting is set to '%s', "
                                         "delimiter must be a '\\t', "
                                         "not a '%s' "
                                         % (QUOTE_NCS, delimiter))
            else:
                self._reader = NCSTSVReader(tsvfile, gunzip, encoding,
                                            strict, autodetect_compressed_file,
                                            fieldnames=fieldnames)
        else:
            self._reader = StandardTSVReader(tsvfile, gunzip, delimiter,
                                             encoding, quoting, strict,
                                             autodetect_compressed_file,
                                             fieldnames=fieldnames)

    @property
    def fieldnames(self):
        return self._reader.fieldnames

    @property
    def filename(self):
        return self._reader.filename

    @property
    def quoting(self):
        return self._reader.quoting

    @property
    def delimiter(self):
        return self._reader.delimiter

    @property
    def encoding(self):
        return self._reader.encoding

    @property
    def strict(self):
        return self._reader.strict

    @property
    def linenumber(self):
        return self._reader.linenumber

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._reader)

    def close(self):
        self._reader.close()

#=============================================================================
# The standard tsv reader class
#=============================================================================
    
class StandardTSVReader(object):
    """
    The basic (and most important) tsv file reader.

    The constructor takes a file object as argument and some keywords
    arguments which have default values.

    **Assumption**: the file object passed to the TSVReader constructor
    is not "decoded". TSVReader will decode each line of the file
    following the encoding provided to the constructor before return a
    row to the user. Therefore, a TSVReader instance return unicode
    object, not str object).
    
    It is possible to iterate through the lines of a tsv by using a for
    loop on a TSVReader instance. For each line in the tsv file,
    the TSVReader instance will throw a list containing every values
    contained on a line.
    """

    def __init__(self, tsvfile, gunzip=False, delimiter='\t',
                 encoding='utf-8', quoting=QUOTE_NONE, strict=True,
                 autodetect_compressed_file=True, fieldnames=None):
        
        self._filename = tsvfile.name

        if gunzip:
            self.file = gzip.GzipFile(fileobj=tsvfile)
        elif autodetect_compressed_file and is_compressed(tsvfile):
            self.file = gzip.GzipFile(fileobj=tsvfile)
        else:
            self.file = tsvfile

        # re-enconde in utf-8
        utf8file = UTF8Recoder(self.file, encoding)
        self._encoding = encoding

        # give to csvreader as utf-8 
        if quoting == QUOTE_NONE:
            self._quoting = csv.QUOTE_NONE
        elif quoting == QUOTE_ALL:
            self._quoting = csv.QUOTE_ALL
        elif quoting == QUOTE_MINIMAL:
            self._quoting = csv.QUOTE_MINIMAL
        elif quoting == QUOTE_NONNUMERIC:
            self._quoting = csv.QUOTE_NONNUMERIC
        else:
            raise TSVReaderException("Unknown quoting value : %s, choices are %s"
                                     % (quoting, ', '.join(QUOTE_CHOICES)))
        
        self._delimiter = delimiter
        self.csvreader = csv.reader(utf8file,
                                    delimiter=self._delimiter,
                                    quoting=self._quoting)
        if fieldnames is not None:
            self._fieldnames = []
            for fn in fieldnames:
                if isinstance(fn, str):
                    self._fieldnames.append(fn)
                else:
                    self._fieldnames.append(str(fn, encoding))
        else:
            try:
                self._fieldnames = [str(s, 'utf-8') for s in  next(self.csvreader)]
            except StopIteration:
                raise TSVReaderException('file %s is empty or closed' % self._filename)
        self._strict = strict

    @property
    def fieldnames(self):
        return self._fieldnames

    @property
    def filename(self):
        return self._filename

    @property
    def quoting(self):
        return self._quoting

    @property
    def delimiter(self):
        return self._delimiter

    @property
    def encoding(self):
        return self._encoding

    @property
    def strict(self):
        return self._strict

    @property
    def linenumber(self):
        return self.csvreader.line_num

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.csvreader)
        if self.strict and len(line) != len(self.fieldnames):
            raise TSVMalformedLineException('[file %s] line %d contains %d value, expected %d values' % (self.filename, self.linenumber, len(line), len(self.fieldnames)))
        return [str(s, 'utf-8') for s in line]

    def close(self):
        self.file.close()

#=============================================================================
# UTF-8 Encoder
#=============================================================================

class UTF8Recoder(object):
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8

    """

    # NOTE: normaly, the commented text would be used instead of what is 
    # currently implemented. The reason for this modification is that
    # if a file was containing a ^L (formfeed) character, that line was
    # would be separated in 2 lines with the commented way (and therefore
    # would need to next() call to be sent. This was causing invaled line
    # exception. I am not sure that the current way this class is implemented
    # is the right way to go though.

    def __init__(self, fileobject, encoding):
        #self.reader = codecs.getreader(encoding)(fileobject)
        self.fo = fileobject
        self.encoding = encoding

    def __iter__(self):
        return self

    def __next__(self):
        #return self.reader.next().encode("utf-8")
        return self.fo.next().decode(self.encoding).encode('utf-8')

#=============================================================================
# Exception Class
#=============================================================================

class TSVReaderException(Exception):
    pass

class TSVMalformedLineException(TSVReaderException):
    pass

class TSVFieldNotFoundException(TSVReaderException):

    def __init__(self, fieldname, tsvreader):
        self.fieldname = fieldname
        self.filename = tsvreader.filename

    def __str__(self):
        return ("The field '%s' does not exist in file %s."
                % (self.fieldname, self.filename))

#=============================================================================

_decode_replacements = {r'\t': '\t', r'\n': '\n', r'\\': '\\'}
_decode_pattern = re.compile(r'\\[tn\\]')

def _decode_field(field):
    return _decode_pattern.sub(lambda mo: _decode_replacements[mo.group(0)], field)

def _parse_line(line, encoding):

    return [_decode_field(field) for field in line.decode(encoding).rstrip('\n').split('\t')]

#=============================================================================

class NCSTSVReader(object):

    def __init__(self, tsvfile, gunzip=False, encoding='utf-8', 
                 strict=True, autodetect_compressed_file=True,
                 fieldnames=None):

        self._filename = tsvfile.name
        self._encoding = encoding

        if gunzip:
            self.file = gzip.GzipFile(fileobj=tsvfile)
        elif autodetect_compressed_file and is_compressed(tsvfile):
            self.file = gzip.GzipFile(fileobj=tsvfile)
        else:
            self.file = tsvfile

        if fieldnames is not None:
            self._fieldnames = []
            for fn in fieldnames:
                if isinstance(fn, str):
                    self._fieldnames.append(fn)
                else:
                    self._fieldnames.append(str(fn, encoding))
        else:
            try:
                self._fieldnames = _parse_line(next(self.file), self._encoding)
            except StopIteration:
                raise TSVReaderException('file %s is empty or closed' % self._filename)

        self._linenumber = 1
        self._strict = strict

    @property
    def fieldnames(self):
        return self._fieldnames

    @property
    def filename(self):
        return self._filename

    @property
    def quoting(self):
        return QUOTE_NCS

    @property
    def delimiter(self):
        return '\t'

    @property
    def encoding(self):
        return self._encoding

    @property
    def strict(self):
        return self._strict

    @property
    def linenumber(self):
        return self._linenumber

    def __iter__(self):
        return self

    def __next__(self):
        line = _parse_line(next(self.file), self._encoding)
        self._linenumber += 1
        if self.strict and len(line) != len(self.fieldnames):
            raise TSVMalformedLineException('[file %s] line %d contains %d value, expected %d values' % (self.filename, self.linenumber, len(line), len(self.fieldnames)))
        return line

    def close(self):
        self.file.close()

