import numpy as np
from revrng.numpy_wrapper import ReversibleRandomState


SEED = 12345
N_ITER = 100
IN_RANGE_SAMPLES = 10000
SHAPES = [2, (1,), (5, 4), (3, 2, 1, 2)]


def test_shape():
    state = ReversibleRandomState(SEED)
    for shape in SHAPES:
        # ndarray shape always tuple even if integer specified
        tuple_shape = shape if isinstance(shape, tuple) else (shape,)
        samples = state.random_int32(shape)
        assert samples.shape == tuple_shape, (
            'random_int32 shape mismatch: should be {0} actually {1}'
            .format(tuple_shape, samples.shape)
        )
        samples = state.standard_uniform(shape)
        assert samples.shape == tuple_shape, (
            'standard_uniform shape mismatch: should be {0} actually {1}'
            .format(tuple_shape, samples.shape)
        )
        samples = state.standard_normal(shape)
        assert samples.shape == tuple_shape, (
            'standard_normal shape mismatch: should be {0} actually {1}'
            .format(tuple_shape, samples.shape)
        )


def test_no_shape_calls():
    state = ReversibleRandomState(SEED)
    sample = state.random_int32()
    assert isinstance(sample, int), (
        'random_int32 type mismatch: should be long instance actually {0}'
        .format(type(sample))
    )
    sample = state.standard_uniform()
    assert isinstance(sample, float), (
        'standard_uniform type mismatch: should be float instance actually {0}'
        .format(type(sample))
    )
    sample = state.standard_normal()
    assert isinstance(sample, float), (
        'standard_normal type mismatch: should be float instance actually {0}'
        .format(type(sample))
    )


def test_dtype():
    state = ReversibleRandomState(SEED)
    shape = (5, 4)
    samples = state.random_int32(shape)
    assert samples.dtype == np.uint64, (
        'random_int32 dtype mismatch: should be uint64 actually {0}'
        .format(samples.dtype)
    )
    samples = state.standard_uniform(shape)
    assert samples.dtype == np.float64, (
        'standard_uniform dtype mismatch: should be float64 actually {0}'
        .format(samples.dtype)
    )
    samples = state.standard_normal(shape)
    assert samples.dtype == np.float64, (
        'standard_normal dtype mismatch: should be float64 actually {0}'
        .format(samples.dtype)
    )


def test_reversibility_random_int32():
    state = ReversibleRandomState(SEED)
    samples_fwd = []
    for i in range(N_ITER):
        samples_fwd.append(state.random_int32(i + 1))
    state.reverse()
    for i in range(N_ITER - 1, -1, -1):
        sample_fwd = samples_fwd.pop(-1)
        sample_bwd = state.random_int32(i + 1)
        assert np.all(sample_fwd == sample_bwd), (
            'Incorrect reversed random_int32 samples, expected {0} got {1}'
            .format(sample_fwd, sample_bwd)
        )


def test_reversibility_standard_uniform():
    state = ReversibleRandomState(SEED)
    samples_fwd = []
    for i in range(N_ITER):
        samples_fwd.append(state.standard_uniform(i + 1))
    state.reverse()
    for i in range(N_ITER - 1, -1, -1):
        sample_fwd = samples_fwd.pop(-1)
        sample_bwd = state.standard_uniform(i + 1)
        assert np.all(sample_fwd == sample_bwd), (
            'Incorrect reversed standard_uniform samples, expected {0} got {1}'
            .format(sample_fwd, sample_bwd)
        )


def test_reversibility_standard_normal():
    state = ReversibleRandomState(SEED)
    samples_fwd = []
    for i in range(N_ITER):
        samples_fwd.append(state.standard_normal(i + 1))
    state.reverse()
    for i in range(N_ITER - 1, -1, -1):
        sample_fwd = samples_fwd.pop(-1)
        sample_bwd = state.standard_normal(i + 1)
        assert np.all(sample_fwd == sample_bwd), (
            'Incorrect reversed standard_normal samples, expected {0} got {1}'
            .format(sample_fwd, sample_bwd)
        )


def test_reversibility_mixed():
    state = ReversibleRandomState(SEED)
    samples_fwd = []
    for i in range(N_ITER):
        samples_fwd.append(state.random_int32(i + 1))
        samples_fwd.append(state.standard_uniform(i + 1))
        samples_fwd.append(state.standard_normal(i + 1))
    state.reverse()
    for i in range(N_ITER - 1, -1, -1):
        # sample in reverse order
        sample_fwd = samples_fwd.pop(-1)
        sample_bwd = state.standard_normal(i + 1)
        assert np.all(sample_fwd == sample_bwd), (
            'Incorrect reversed standard_normal samples, expected {0} got {1}'
            .format(sample_fwd, sample_bwd)
        )
        sample_fwd = samples_fwd.pop(-1)
        sample_bwd = state.standard_uniform(i + 1)
        assert np.all(sample_fwd == sample_bwd), (
            'Incorrect reversed standard_uniform samples, expected {0} got {1}'
            .format(sample_fwd, sample_bwd)
        )
        sample_fwd = samples_fwd.pop(-1)
        sample_bwd = state.random_int32(i + 1)
        assert np.all(sample_fwd == sample_bwd), (
            'Incorrect reversed random_int32 samples, expected {0} got {1}'
            .format(sample_fwd, sample_bwd)
        )


def test_random_int32_in_range():
    state = ReversibleRandomState(SEED)
    samples = state.random_int32(IN_RANGE_SAMPLES)
    assert np.all(samples >= 0) and np.all(samples < 2**32), (
        'random_int32 samples out of range [0, 2**32)'
    )


def test_standard_uniform_in_range():
    state = ReversibleRandomState(SEED)
    samples = state.standard_uniform(IN_RANGE_SAMPLES)
    assert np.all(samples >= 0.) and np.all(samples < 1.), (
        'standard_uniform samples out of range [0., 1.)'
    )
