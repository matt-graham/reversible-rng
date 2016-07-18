#cython: embedsignature=True
""" Numpy-compatible reversible random number generation. """

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock


cdef extern from "revrand.h":

    cdef enum: KEY_LENGTH

    ctypedef struct rng_state:
        unsigned long seed
        unsigned long key[KEY_LENGTH]
        int pos
        int reversed
        int n_twists

    void init_state(unsigned long seed, rng_state *state)
    void reverse(rng_state *state)
    unsigned long random_int32(rng_state *state) nogil
    double random_uniform(rng_state *state) nogil
    void random_normal_pair(
        rng_state *state, double *ret_1, double *ret_2) nogil


ctypedef unsigned long (* ulong_rand_func)(rng_state *state) nogil
ctypedef double (* double_rand_func)(rng_state *state) nogil
ctypedef void (* double_pair_rand_func)(
    rng_state *state, double *ret_1, double *ret_2) nogil


cdef object assign_random_ulong_array(
        rng_state *state, ulong_rand_func func, object shape, object lock):
    cdef np.ndarray values
    cdef unsigned long* values_data
    cdef size_t values_size, i
    if shape is not None:
        values = <np.ndarray>np.empty(shape=shape, dtype=np.uint64)
        values_data = <unsigned long*>values.data
        values_size = <size_t>values.size
        with lock, nogil:
            if state.reversed == 0:
                for i in range(0, values_size):
                    values_data[i] = func(state)
            else:
                for i in range(values_size - 1, -1, -1):
                    values_data[i] = func(state)
        return values
    else:
        with lock, nogil:
            value = func(state)
        return value


cdef object assign_random_double_array(
        rng_state *state, double_rand_func func, object shape, object lock):
    cdef np.ndarray values
    cdef double* values_data
    cdef size_t values_size, i
    if shape is not None:
        values = <np.ndarray>np.empty(shape=shape, dtype=np.float64)
        values_data = <double*>values.data
        values_size = <size_t>values.size
        with lock, nogil:
            if state.reversed == 0:
                for i in range(0, values_size):
                    values_data[i] = func(state)
            else:
                for i in range(values_size - 1, -1, -1):
                    values_data[i] = func(state)
        return values
    else:
        with lock, nogil:
            value = func(state)
        return value


cdef object assign_random_double_pair_array(
        rng_state *state, double_pair_rand_func func, object shape,
        object lock):
    cdef np.ndarray values
    cdef double* values_data
    cdef int values_size_is_odd
    cdef size_t values_size, values_size_even, i
    cdef double value_1, value_2
    if shape is not None:
        values = <np.ndarray>np.empty(shape=shape, dtype=np.float64)
        values_data = <double*>values.data
        values_size = <size_t>values.size
        values_size_is_odd = <int>(values_size & 1)
        values_size_even = values_size - values_size_is_odd
        with lock, nogil:
            if state.reversed == 0:
                for i in range(0, values_size_even, 2):
                    func(state, &values_data[i], &values_data[i + 1])
                if values_size_is_odd:
                    func(state, &values_data[values_size - 1], &value_1)
            else:
                if values_size_is_odd:
                    func(state, &values_data[values_size - 1], &value_1)
                for i in range(values_size_even - 1, -1, -2):
                    func(state, &values_data[i - 1], &values_data[i])
        return values
    else:
        with lock, nogil:
            func(state, &value_1, &value_2)
        return value_1


cdef class ReversibleRandomState:
    """ Numpy-compatible reversible random number generator. """

    cdef rng_state *internal_state
    cdef object lock

    def __cinit__(self, seed):
        self.internal_state = <rng_state*> PyMem_Malloc(sizeof(rng_state))

    def __init__(self, seed):
        """
        Reversible random number generator.

        Parameters
        ----------
        seed : int
            Integer seed in range [0, 2**32 - 1].

        Raises
        ------
            ValueError: Seed outside of [0, 2**32 - 1] specified.
            TypeError: Non-integer seed.
        """
        self.lock = Lock()
        self.seed(seed)

    def __dealloc__(self):
        if self.internal_state != NULL:
            PyMem_Free(self.internal_state)
            self.internal_state = NULL

    def seed(self, seed):
        """
        Initialise state using an integer seed.

        Parameters
        ----------
        seed : int
            Integer seed in range [0, 2**32 - 1].

        Raises
        ------
            ValueError: Seed outside of [0, 2**32 - 1] specified.
            TypeError: Non-integer seed.
        """
        try:
            seed = int(seed)
            if seed > int(2**32 - 1) or seed < 0:
                raise ValueError("Seed must be in integer in [0, 2**32 - 1].")
            with self.lock:
                init_state(seed, self.internal_state)
        except TypeError:
            raise TypeError("Seed must be an integer.")

    def get_state(self):
        """
        Get a dictionary representing the internal state of the generator.

        Returns
        -------
        dict
            seed :
                integer seed used to initialise state
            key:
                Mersenne-Twister 624 integer state
            pos:
                current position in key
            reversed:
                whether updating forward (==0) or in reverse (==1)
            n_twists:
                number of twist operations perfomed (initial state defined as
                zero, reverse twists decrement therefore can be negative)
        """
        cdef size_t i
        cdef np.ndarray key = <np.ndarray>np.empty(KEY_LENGTH, np.uint)
        with self.lock:
            seed = self.internal_state.seed
            for i in range(KEY_LENGTH):
                key[i] = self.internal_state.key[i]
            pos = self.internal_state.pos
            reversed = self.internal_state.reversed
            n_twists = self.internal_state.n_twists
        key = <np.ndarray>np.asarray(key, np.uint32)
        return {
            'seed': seed,
            'key': key,
            'pos': pos,
            'reversed': reversed,
            'n_twists': n_twists
        }

    def reverse(self):
        """
        Reverse direction of random number generator updates.
        """
        reverse(self.internal_state)

    def random_int32(self, shape=None):
        """
        Generate array of random integers uniformly distributed on [0, 2**32).

        Parameters
        ----------
        shape : tuple or None
            Shape (dimensions) of generated array or None to return scalar.

        Returns
        -------
        ndarray or int
            Generated samples.
        """
        values = assign_random_ulong_array(
            self.internal_state, random_int32, shape, self.lock
        )
        if shape is None:
            return int(values)
        else:
            return values

    def standard_uniform(self, shape=None):
        """
        Generate array of random double-precision floating point values
        uniformly distributed on [0, 1).

        Parameters
        ----------
        shape : tuple or None
            Shape (dimensions) of generated array or None to return scalar.

        Returns
        -------
        ndarray or float
            Generated samples.
        """
        return assign_random_double_array(
            self.internal_state, random_uniform, shape, self.lock
        )

    def standard_normal(self, shape=None):
        """
        Generate array of random double-precision floating point values
        from zero mean, unit variance normal distribution.

        Note that the normal samples are always generated in pairs - if an
        array of odd overall size (or single scalar value) is specified, one
        normal sample will be discarded (with reversibility maintained).
        Therefore sampling many individual normal values will be relatively
        inefficient.

        Parameters
        ----------
        shape : tuple or None
            Shape (dimensions) of generated array or None to return scalar.

        Returns
        -------
        ndarray or float
            Generated samples.
        """
        return assign_random_double_pair_array(
            self.internal_state, random_normal_pair, shape, self.lock
        )
