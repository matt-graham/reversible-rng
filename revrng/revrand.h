/*
 * Reversible Mersenne-Twister pseudo-random number generator.
 *
 * Author: Matt Graham (matt-graham.github.io)
 *
 * Implementation of the Mersenne-Twister pseudo-random number generator of
 * Matsumoto and Nishimura (1997) with reversible updates to the generator
 * internal state. That is following a sequence of draws from a generator,
 * given only the generator internal state the exact reverse of that sequence
 * of draws can be generated.
 *
 * Updates in reverse_twist function based on James Roper's excellent blog post
 * https://jazzy.id.au/2010/09/25/cracking_random_number_generators_part_4.html
 *
 * rng_state structure and overall design and implementation heavily derived
 * from Random kit 1.3 by Jean-Sebastien Roy (js@jeannot.org) though with only
 * a small subset of functions in Random Kit implemented here.
 *
 * The twist, random_int32 and init_state functions algorithms and the
 * original design of the Mersenne Twister RNG:
 *
 *   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   3. The names of its contributors may not be used to endorse or promote
 *   products derived from this software without specific prior written
 *   permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 *   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Constants used in the random_double implementation by Isaku Wada.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

 /* Mersenne-Twister (MT-19937) key/state length */
 #define KEY_LENGTH 624

 /* Internal random number generator state. */
 typedef struct rng_state_
 {
     unsigned long seed; /* integer seed used to initialise state */
     unsigned long key[KEY_LENGTH]; /* Mersenne-Twister state */
     int pos; /* current position in key array */
     int reversed; /* ==0: forward state updates, !=0: reverse state updates */
     int n_twists; /* number of twists performed */
 } rng_state;

 /* Initialise generator state from an integer seed. */
 void init_state(unsigned long seed, rng_state *state);

 /* Optimised implementation of reference Mersenne-Twister from Random Kit. */
 void twist(rng_state *state);

 /* Reverses twist of state: reverse_twist(twist(state)) is identity map. */
 void reverse_twist(rng_state *state);

 /* Reverses direction of random number generation. */
 void reverse(rng_state *state);

 /* Generates a random integer uniformly from range [0, 2^32 - 1]. */
 unsigned long random_int32(rng_state *state);

 /*
  * Generate a random double-precision floating point value from uniform
  * distribution on [0,1).
  */
 double random_uniform(rng_state *state);

 /*
  * Generate a pair of independent random double-precision floating point
  * values from the (zero-mean, unit variance) standard normal distribution.
  */
 void random_normal_pair(rng_state *state, double *ret_1, double *ret_2);
