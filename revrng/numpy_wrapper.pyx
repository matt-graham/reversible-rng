import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock


cdef extern from "revrand.h":

    ctypedef struct rng_state:
        int reverse
        unsigned long seed
        int n_twists
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

    void init_state(unsigned long seed, rng_state *state)
    void reverse(rng_state *state)
    unsigned long random_int32(rng_state *state) nogil
    double random_double(rng_state *state) nogil
    double random_gauss(rng_state *state) nogil


ctypedef double (* double_rand_func)(rng_state *state) nogil
ctypedef unsigned long (* ulong_rand_func)(rng_state *state) nogil


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
            if state.reverse == 1:
                for i in range(values_size - 1, -1, -1):
                    values_data[i] = func(state)
            else:
                for i in range(values_size):
                    values_data[i] = func(state)
        return values
    else:
        with lock, nogil:
            value = func(state)
        return value


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
            if state.reverse == 1:
                for i in range(values_size - 1, -1, -1):
                    values_data[i] = func(state)
            else:
                for i in range(values_size):
                    values_data[i] = func(state)
        return values
    else:
        with lock, nogil:
            value = func(state)
        return value


cdef class ReversibleRandomState:

    cdef rng_state *internal_state
    cdef object lock

    def __cinit__(self, seed):
        self.internal_state = <rng_state*> PyMem_Malloc(sizeof(rng_state))

    def __init__(self, seed):
        self.lock = Lock()
        self.seed(seed)

    def __dealloc__(self):
        if self.internal_state != NULL:
            PyMem_Free(self.internal_state)
            self.internal_state = NULL

    def seed(self, seed):
        try:
            seed = int(seed)
            if seed > int(2**32 - 1) or seed < 0:
                raise ValueError("Seed must be in integer in [0, 2**32 - 1].")
            with self.lock:
                init_state(seed, self.internal_state)
        except TypeError:
            raise TypeError("Seed must be an integer.")

    def reverse(self):
        reverse(self.internal_state)

    def random_integers(self, shape=None):
        return assign_random_ulong_array(
            self.internal_state, random_int32, shape, self.lock
        )

    def standard_uniform(self, shape=None):
        return assign_random_double_array(
            self.internal_state, random_double, shape, self.lock
        )

    def standard_normal(self, shape=None):
        return assign_random_double_array(
            self.internal_state, random_gauss, shape, self.lock
        )
