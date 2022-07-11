from ctypes import *

def TypedEnumerationType(tp):
    class EnumerationType(type(tp)):  # type: ignore
        def __new__(metacls, name, bases, dict):
            if not "_members_" in dict:
                _members_ = {}
                for key, value in dict.items():
                    if not key.startswith("_"):
                        _members_[key] = value

                dict["_members_"] = _members_
            else:
                _members_ = dict["_members_"]

            dict["_reverse_map_"] = {v: k for k, v in _members_.items()}
            cls = type(tp).__new__(metacls, name, bases, dict)
            for key, value in cls._members_.items():
                globals()[key] = value
            return cls

        def __repr__(self):
            return "<Enumeration %s>" % self.__name__

    return EnumerationType

def TypedCEnumeration(tp):
    class CEnumeration(tp, metaclass=TypedEnumerationType(tp)):
        _members_ = {}

        def __repr__(self):
            value = self.value
            return f"<{self.__class__.__name__}.{self._reverse_map_.get(value, '(unknown)')}: {value}>"

        def __eq__(self, other):
            if isinstance(other, int):
                return self.value == other

            return type(self) == type(other) and self.value == other.value

    return CEnumeration