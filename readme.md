# Reversible random number generator

Implementation of the Mersenne-Twister pseudo-random number generator of
[Matsumoto and Nishimura (1997)](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.215.1141)
with reversible updates to the generator internal state. 

That is following a sequence of draws from a generator, given only the generator 
internal state the exact reverse of that sequence of draws can be generated.

The reverse updates are based on James Roper's excellent blog post:
 https://jazzy.id.au/2010/09/25/cracking_random_number_generators_part_4.html

Both a lightweight `C` implementation and `python` `numpy` wrapper are provided.
