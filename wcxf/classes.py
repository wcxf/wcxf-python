import json
import yaml
import logging
from collections import OrderedDict, Counter

# the following is necessary to get pretty representations of
# OrderedDict and defaultdict instances in YAML
def _represent_dict_order(self, data):
    return self.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, _represent_dict_order)

# the following is necessary to have Null values be represented by
# emptyness rather than 'null'
def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '')
yaml.add_representer(type(None), represent_none)


def _load_yaml_json(stream, **kwargs):
    """Load a JSON or YAML file from a string or stream."""
    if isinstance(stream, str):
        ss = stream
    else:
        ss = stream.read()
    try:
        return json.loads(ss, **kwargs)
    except ValueError:
        return yaml.load(ss, **kwargs)

def _dump_json(d, stream=None, **kwargs):
    """Dump to a JSON string (if `stream` is None) or stream."""
    if stream is not None:
        return json.dump(d, stream, **kwargs)
    else:
        return json.dumps(d, **kwargs)

def _yaml_to_json(stream_in, stream_out, **kwargs):
    d = yaml.load(stream_in)
    return _dump_json(d, stream_out, **kwargs)

def _json_to_yaml(stream_in, stream_out, **kwargs):
    d = json.load(stream_in)
    return yaml.dump(d, stream_out, **kwargs)

class NamedInstanceMetaclass(type):
    # this is just needed to implement the getitem method on NamedInstanceClass
    # to allow the syntax MyClass['instancename'] as shorthand for
    # MyClass.get_instance('instancename'); same for
    # del MyClass['instancename'] instead of MyClass.del_instance('instancename')
    def __getitem__(cls, item):
        return cls.get_instance(item)

    def __delitem__(cls, item):
        return cls.del_instance(item)

class NamedInstanceClass(object, metaclass=NamedInstanceMetaclass):
    """Base class for classes that have named instances that can be accessed
    by their name.

    Parameters
    ----------
     - name: string

    Methods
    -------
     - del_instance(name)
         Delete an instance
     - get_instance(name)
         Get an instance
     - set_description(description)
         Set the description
    """

    def __init__(self, _name):
        if not hasattr(self.__class__, 'instances'):
            self.__class__.instances = OrderedDict()
        self.__class__.instances[_name] = self
        self._name = _name

    @classmethod
    def get_instance(cls, _name):
        return cls.instances[_name]

    @classmethod
    def del_instance(cls, _name):
        del cls.instances[_name]

    @classmethod
    def clear_all(cls):
        """Delete all instances."""
        cls.instances = OrderedDict()


class WCxf(object):
    """Base class for WCxf files (not meant to be used directly)."""

    @classmethod
    def load(cls, stream, **kwargs):
        """Load the object data from a JSON or YAML file."""
        wcxf = _load_yaml_json(stream, **kwargs)
        return cls(**wcxf)

    def dump(self, stream=None, fmt='json', **kwargs):
        """Dump the object data to a JSON or YAML file."""
        d = {k: v for k,v in self.__dict__.items() if k[0] != '_'}
        if fmt.lower() == 'json':
            return _dump_json(d, stream=stream, **kwargs)
        elif fmt.lower() == 'yaml':
            return yaml.dump(d, stream, **kwargs)
        else:
            raise ValueError("Format {} unknown: use 'json' or 'yaml'.".format(fmt))

class EFT(WCxf, NamedInstanceClass):
    """Class representing EFT files."""
    def __init__(self, eft, sectors, **kwargs):
        """Instantiate the EFT file object."""
        self.eft = eft
        self.sectors = sectors
        super().__init__(eft)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def known_bases(self):
        """Return a list of known bases for this EFT."""
        return tuple(basis[1] for basis in Basis.instances if basis[0] == self.eft)

    @property
    def known_translators(self):
        """Return a list of known translators between bases of this EFT."""
        return tuple(t for t in Translator.instances if t[0] == self.eft)

class Basis(WCxf, NamedInstanceClass):
    """Class representing basis files."""
    def __init__(self, eft, basis, sectors, **kwargs):
        """Instantiate the basis file object."""
        self.eft = eft
        self.basis = basis
        super().__init__((eft, basis))
        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self, 'parent'):
            try:
                self.sectors = Basis[self.eft, self.parent].sectors
                self.sectors.update(sectors)
            except (AttributeError, KeyError):
                raise ValueError("Parent basis {} not found".format(self.parent))
        else:
            self.sectors = sectors

    @property
    def known_translators(self):
        """Return a list of known translators to and from this basis."""
        kt = {}
        kt['from'] = tuple(t for t in Translator.instances
                           if t[0] == self.eft and t[1] == self.basis)
        kt['to'] = tuple(t for t in Translator.instances
                           if t[0] == self.eft and t[2] == self.basis)
        return kt

    @property
    def all_wcs(self):
        """Return a list with all Wilson coefficients defined in this basis."""
        return [wc for sector, wcs in self.sectors.items() for wc in wcs]

    def validate(self):
        """Validate the basis file."""
        try:
            eft_instance = EFT[self.eft]
        except (AttributeError, KeyError):
            raise ValueError("EFT {} not defined".format(self.eft))
        unknown_sectors = set(self.sectors.keys()) - set(eft_instance.sectors.keys())
        if unknown_sectors:
            raise ValueError("Unkown sectors: {}".format(unknown_sectors))
        all_keys = [k for s in self.sectors.values() for k, v in s.items()]
        if len(all_keys) != len(set(all_keys)): # we have duplicate keys!
            cnt = Counter(all_keys)
            dupes = [k for k, v in cnt.items() if v > 1]
            raise ValueError("Duplicate coefficients in different sectors:"
                             " {}".format(dupes))


class WC(WCxf):
    """Class representing Wilson coefficient files."""
    def __init__(self, eft, basis, scale, values, **kwargs):
        """Instantiate the Wilson coefficient file object."""
        self.eft = eft
        self.basis = basis
        self.scale = float(scale)
        self.values = values
        self._dict = None
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _to_number(v):
        """Turn a Wilson coefficient value - that could be a number or a Re/Im
        dict - into a number."""
        if isinstance(v, dict):
            return float(v.get('Re', 0)) + 1j*float(v.get('Im', 0))
        else:
            return float(v)

    @staticmethod
    def _to_complex_dict(v):
        """Turn a numeric Wilson coefficient value into a Re/Im dict if it is
        complex."""
        if v.imag != 0:
            return {'Re': float(v.real), 'Im': float(v.imag)}
        else:
            return float(v.real)

    @classmethod
    def dict2values(cls, d):
        return {k: cls._to_complex_dict(v) for k, v in d.items()}

    def validate(self):
        """Validate the Wilson coefficient file."""
        try:
            eft_instance = EFT[self.eft]
        except (AttributeError, KeyError):
            raise ValueError("EFT {} not defined".format(self.eft))
        try:
            basis_instance = Basis[self.eft, self.basis]
        except (AttributeError, KeyError):
            raise ValueError("Basis {} not defined for EFT {}".format(self.basis, self.eft))
        unknown_keys = set(self.values.keys()) - set(basis_instance.all_wcs)
        assert unknown_keys == set(), \
            "Wilson coefficients do not exist in this basis: " + str(unknown_keys)

    @property
    def dict(self):
        """Return a dictionary with the Wilson coefficient values.
        The dictionary will be cached when called for the first time."""
        if self._dict is None:
            self._dict = {k: self._to_number(v) for k, v in self.values.items()}
        return self._dict

    def translate(self, to_basis):
        """Translate the Wilson coefficients to a different basis.
        Returns a WC instance."""
        try:
            translator = Translator[self.eft, self.basis, to_basis]
        except (KeyError, AttributeError):
            raise ValueError("No translator from basis {} to {} found.".format(self.basis, to_basis))
        return translator.translate(self)

    def match(self, to_eft, to_basis):
        """Match the Wilson coefficients to a different EFT.
        Returns a WC instance."""
        try:
            matcher = Matcher[self.eft, self.basis, to_eft, to_basis]
        except (KeyError, AttributeError):
            raise ValueError("No matcher from EFT {} in basis {} to EFT {} in basis {} found.".format(self.eft, self.basis, to_eft, to_basis))
        return matcher.match(self)


class Translator(NamedInstanceClass):
    """Class for translating between different bases of the same EFT."""
    def __init__(self, eft, from_basis, to_basis, function):
        """Initialize the Translator instance."""
        super().__init__((eft, from_basis, to_basis))
        self.eft = eft
        self.from_basis = from_basis
        self.to_basis = to_basis
        self.function = function

    def translate(self, WC_in):
        """Translate a WC object from `from_basis` to `to_basis`."""
        dict_out = self.function(WC_in.dict)
        # filter out zero values
        dict_out = {k: v for k, v in dict_out.items() if v != 0}
        values = WC.dict2values(dict_out)
        WC_out = WC(self.eft, self.to_basis, WC_in.scale, values)
        return WC_out


class Matcher(NamedInstanceClass):
    """Class for matching from a UV to an IR EFT."""
    def __init__(self, from_eft, from_basis, to_eft, to_basis, function):
        """Initialize the Matcher instance."""
        super().__init__((from_eft, from_basis, to_eft, to_basis))
        self.from_eft = from_eft
        self.from_basis = from_basis
        self.to_eft = to_eft
        self.to_basis = to_basis
        self.function = function

    def match(self, WC_in):
        """Translate a WC object in EFT `from_eft` and basis `from_basis`
        to EFT `to_eft` and basis `to_basis`."""
        dict_out = self.function(WC_in.dict)
        # filter out zero values
        dict_out = {k: v for k, v in dict_out.items() if v != 0}
        values = WC.dict2values(dict_out)
        WC_out = WC(self.to_eft, self.to_basis, WC_in.scale, values)
        return WC_out

def parametrized(dec):
    """Decorator for a decorator allowing it to take arguments.
    See https://stackoverflow.com/a/26151604."""
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

@parametrized
def translator(func, eft, from_basis, to_basis):
    """Decorator for basis translation functions.

    Usage:

    ```python
    @translator('myEFT', 'myBasis_from', 'myBasis_to')
    def myFunction(wc_dict_from):
        ... # do something
        return wc_dict_to
    ```
    """
    Translator(eft, from_basis, to_basis, func)
    return func


@parametrized
def matcher(func, from_eft, from_basis, to_eft, to_basis):
    """Decorator for matching functions.

    Usage:

    ```python
    @matcher('myEFT_from', 'myBasis_from', 'myEFT_to', 'myBasis_to')
    def myFunction(wc_dict_from):
        ... # do something
        return wc_dict_to
    ```
    """
    Matcher(from_eft, from_basis, to_eft, to_basis, func)
    return func
