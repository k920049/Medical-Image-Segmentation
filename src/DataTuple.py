import collections

_DataTuple = collections.namedtuple("DataTuple", ("image", "label"))


class DataTuple(_DataTuple):

    __slots__ = ()

    @property
    def dtype(self):
        (image, label) = self

        if image.dtype == label.dtype:
            return image.dtype
        else:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(image.dtype), str(label.dtype)))