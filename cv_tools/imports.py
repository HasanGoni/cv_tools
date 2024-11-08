import sys,os,re,typing,itertools,operator,functools,math,warnings,functools,io,enum
import numpy as np

from operator import itemgetter,attrgetter
from warnings import warn
from typing import Iterable,Generator,Sequence,Iterator,List,Set,Dict,Union,Optional,Tuple
from functools import partial,reduce
from pathlib import Path
from PIL import Image
import cv2
import shutil
import pandas as pd
from tqdm.auto import tqdm
#from fastcore.script import *
#from fastcore.all import *

try:
    from types import WrapperDescriptorType,MethodWrapperType,MethodDescriptorType
except ImportError:
    WrapperDescriptorType = type(object.__init__)
    MethodWrapperType = type(object().__str__)
    MethodDescriptorType = type(str.join)
from types import BuiltinFunctionType,BuiltinMethodType,MethodType,FunctionType,SimpleNamespace

NoneType = type(None)
string_classes = (str,bytes)

def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    #Rank 0 tensors in PyTorch are not really iterable
    return isinstance(o, (Iterable,Generator)) and getattr(o,'ndim',1)

def is_coll(o):
    "Test whether `o` is a collection (i.e. has a usable `len`)"
    #Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(o, '__len__') and getattr(o,'ndim',1)

def all_equal(a,b):
    "Compares whether `a` and `b` are the same length and have the same contents"
    if not is_iter(b): return a==b
    return all(equals(a_,b_) for a_,b_ in itertools.zip_longest(a,b))

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

def noops(self, x=None, *args, **kwargs):
    "Do nothing (method)"
    return x

def any_is_instance(t, *args): return any(isinstance(a,t) for a in args)

def isinstance_str(x, cls_name):
    "Like `isinstance`, except takes a type name instead of a type"
    return cls_name in [t.__name__ for t in type(x).__mro__]

def array_equal(a,b):
    if hasattr(a, '__array__'): a = a.__array__()
