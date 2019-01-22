#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


import warnings
import numpy as np
from database.util import get_unique_identifier
from .background import BackgroundGenerator


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class EndOfBatch(metaclass=Singleton):
    pass


class InputOutputSignatureMismatch(Exception):
    pass


def batchify(generator, signature, batch_size=32,
             incomplete=False, prefetch=0):
    """Pack and yield batches out of a generator

    Parameters
    ----------
    generator : iterable
        Generator
    signature : dict, optional
        Signature of the generator.
    batch_size : int, optional
        Batch size. Defaults to 32.
    incomplete : boolean, optional
        Set to True to yield final batch, even if it is incomplete (i.e.
        smaller than requested batch size). Default behavior is to not
        yield incomplete final batch.
    prefetch : int, optional
        Prefetch that many batches in a background thread.
        Defaults to not prefetch anything.


    Returns
    -------
    batch_generator : iterable
        Batch generator
    """

    class Generator(object):

        def __iter__(self):
            return self

        def next(self):
            return self.__next__()

        def __next__(self):
            return next(generator)

    batches = BaseBatchGenerator(
        Generator(), signature, batch_size=batch_size, incomplete=incomplete)

    if prefetch:
        batches = BackgroundGenerator(batches, max_prefetch=prefetch)

    for batch in batches:
        yield batch


class BaseBatchGenerator(object):
    """Base class to pack and yield batches out of a generator

    Parameters
    ----------
    generator : iterable
        Internal generator from which batches are packed
    batch_size : int, optional
        Defaults to 32.
    incomplete : boolean, optional
        Set to True to yield final batch, even if it is incomplete (i.e.
        smaller than requested batch size). Default behavior is to not
        yield incomplete final batch.
    """
    def __init__(self, generator, signature, batch_size=32, incomplete=False):
        super(BaseBatchGenerator, self).__init__()

        self.generator = generator
        self.signature = signature

        self.batch_size = batch_size
        self.incomplete = incomplete

        self.batch_generator_ = self.iter_batches()

    def init(self, signature=None):
        """Initialize new batch"""

        if signature is None:
            signature = self.signature

        if type(signature) == list:
            return [self.init(s) for s in signature]

        if type(signature) == tuple:
            return tuple([self.init(s) for s in signature])

        if '@' not in signature:
            return {key: self.init(s)
                    for key, s in signature.items()}

        return []

    def push(self, item, signature=None, batch=None, **kwargs):
        """Process item and push it to current batch"""

        if signature is None:
            signature = self.signature

        if batch is None:
            batch = self.batch_

        if type(signature) in (list, tuple):
            for i, s, b, in zip(item, signature, batch):
                self.push(i, s, batch=b, **kwargs)
                return

        if '@' not in signature:
            for key in signature:
                self.push(item[key], signature[key],
                          batch=batch[key], **kwargs)
            return

        process_func = signature['@'][0]
        processed = item if process_func is None \
                    else process_func(item, **kwargs)
        batch.append(processed)

    def pack(self, signature=None, batch=None):
        """Pack current batch"""

        if signature is None:
            signature = self.signature

        if batch is None:
            batch = self.batch_

        if type(signature) == list:
            return list(self.pack(s, batch=b)
                        for s, b in zip(signature, batch))

        if type(signature) == tuple:
            return tuple(self.pack(s, batch=b)
                          for s, b in zip(signature, batch))

        if '@' in signature:
            pack_func = signature['@'][1]
            packed = batch if pack_func is None else pack_func(batch)
            return packed

        return {key: self.pack(signature[key], batch=batch[key])
                for key in signature}

    def postprocess(self, batch):
        """Post-process current batch"""
        return batch

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.batch_generator_)

    def iter_batches(self):

        endOfBatch = EndOfBatch()

        # create new empty batch
        self.batch_ = self.init(self.signature)
        batch_size = 0
        complete = False

        for fragment in self.generator:

            if fragment is endOfBatch:
                complete = True
            else:
                self.push(fragment, self.signature)
                batch_size += 1

            complete |= self.batch_size > 0 and batch_size == self.batch_size

            if complete:
                if batch_size:
                    batch = self.pack(self.signature)
                    yield self.postprocess(batch)
                self.batch_ = self.init(self.signature)
                batch_size = 0
                complete = False

        # yield last incomplete batch
        if batch_size > 0 and self.incomplete:
            batch = self.pack(self.signature)
            yield self.postprocess(batch)


def forever(iterable, shuffle=False):
    """Loop over the iterable indefinitely.

    Parameters
    ----------
    iterable : iterable
    shuffle : bool, optional
        Shuffle iterable after each full consumption
    """
    saved = list(iterable)
    while saved:
        if shuffle:
            np.random.shuffle(saved)
        for element in saved:
              yield element


class FileBasedBatchGenerator(BaseBatchGenerator):
    """

    Parameters
    ----------
    generator :
        Must implement generator.from_file
    """

    def preprocess(self, current_file, **kwargs):
        """Returns pre-processed current_file
        (and optionally set internal state)
        """
        return current_file

    def from_file(self, current_file, incomplete=True):
        """

        Parameters
        ----------
        current_file :
        incomplete : boolean, optional
            Set to False to not yield final batch if its incomplete (i.e.
            smaller than requested batch size). Default behavior is to yield
            incomplete final batch.
        """
        def current_file_generator():
            yield current_file
        for batch in self.__call__(current_file_generator(),
                                   infinite=False,
                                   incomplete=incomplete):
            yield batch

    def __call__(self, file_generator, infinite=False,
                 robust=False, incomplete=False):
        """Generate batches by looping over a (possibly infinite) set of files

        Parameters
        ----------
        file_generator : iterable
            File generator yielding dictionaries at least containing the 'uri'
            key (uri = uniform resource identifier). Typically, one would use
            the 'train' method of a protocol available in pyannote.database.
        infinite : boolean, optional
            Loop over the file generator indefinitely, in random order.
            Defaults to exhaust the file generator only once, and then stop.
        robust : boolean, optional
            Set to True to skip files for which preprocessing fails.
            Default behavior is to raise an error.
        incomplete : boolean, optional
            Set to True to yield final batch, even if its incomplete (i.e.
            smaller than requested batch size). Default behavior is to not
            yield incomplete final batch. Has no effect when infinite is True.

        See also
        --------
        pyannote.database
        """

        # create new empty batch
        self.batch_ = self.init(self.signature)
        batch_size = 0

        if infinite:
            file_generator = forever(file_generator, shuffle=True)

        for current_file in file_generator:

            try:
                preprocessed_file = self.preprocess(current_file)
            except Exception as e:
                if robust:
                    uri = get_unique_identifier(current_file)
                    msg = 'Cannot preprocess file "{uri}".'
                    warnings.warn(msg.format(uri=uri))
                    continue
                else:
                    raise e

            for fragment in self.generator.from_file(preprocessed_file):

                # add item to batch
                self.push(fragment, self.signature,
                                current_file=preprocessed_file)
                batch_size += 1

                # fixed batch size
                if self.batch_size > 0 and batch_size == self.batch_size:
                    batch = self.pack(self.signature)
                    yield self.postprocess(batch)
                    self.batch_ = self.init(self.signature)
                    batch_size = 0

            # mono-batch
            if self.batch_size < 1:
                batch = self.pack(self.signature)
                yield self.postprocess(batch)
                self.batch_ = self.init(self.signature)
                batch_size = 0

        # yield incomplete final batch
        if batch_size > 0 and batch_size < self.batch_size and incomplete:
            batch = self.pack(self.signature)
            yield self.postprocess(batch)
