# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ExpOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsExpOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ExpOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ExpOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # ExpOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def ExpOptionsStart(builder): builder.StartObject(0)
def ExpOptionsEnd(builder): return builder.EndObject()


class ExpOptionsT(object):

    # ExpOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        expOptions = ExpOptions()
        expOptions.Init(buf, pos)
        return cls.InitFromObj(expOptions)

    @classmethod
    def InitFromObj(cls, expOptions):
        x = ExpOptionsT()
        x._UnPack(expOptions)
        return x

    # ExpOptionsT
    def _UnPack(self, expOptions):
        if expOptions is None:
            return

    # ExpOptionsT
    def Pack(self, builder):
        ExpOptionsStart(builder)
        expOptions = ExpOptionsEnd(builder)
        return expOptions
