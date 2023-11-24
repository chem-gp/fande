from .data_module_ase import FandeDataModuleASE


class AttrDict(dict):
    # """ Dictionary subclass whose entries can be accessed by attributes (as well
    #     as normally).
    def __init__(self, *args, **kwargs):
        """Description"""
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(cls, data):
        """Construct nested AttrDicts from nested dictionaries."""
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})


# print("objects of fande.data module imported...")
