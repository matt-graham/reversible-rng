# Reversible random number generator

Implementation of the Mersenne-Twister pseudo-random number generator of
[Matsumoto and Nishimura (1997)](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.215.1141)
with reversible updates to the generator internal state. 

That is following a sequence of samples from a generator, given only the generator 
internal state the exact reverse of that sequence of samples can be generated.

The reverse updates are based on James Roper's excellent blog post:
 https://jazzy.id.au/2010/09/25/cracking_random_number_generators_part_4.html

Both a C implementation and `numpy` compatible Python wrapper are provided.

## Example usage

```python
import revrng
import numpy as np

seed = 12345
n_iter = 10
rng = revrng.ReversibleRandomState(seed)
us = []
ns = []

# sample set of random uniform and normal ndarrays of increasing length
for i in range(n_iter):
    us.append(rng.standard_uniform(shape=(i,)))
    ns.append(rng.standard_normal(shape=(i,)))
    
# reverse direction of random number generator
rng.reverse()

# sample same random uniform and normal vectors in reverse
for i in range(n_iter - 1, -1, -1):
    assert np.all(ns.pop(-1) == rng.standard_normal(shape=(i,)))
    assert np.all(us.pop(-1) == rng.standard_uniform(shape=(i,)))
```
