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
 * The twist, random_int32 and init_state function algorithms and the
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


#include <math.h>
#define PI 3.141592653589793238462643383279502884

/* 32-bit Mersenne-Twister (MT-19937) constants */
#define KEY_LENGTH 624
#define MID_OFFSET 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL
#define TEMPER_SHIFT_A 11
#define TEMPER_SHIFT_B 7
#define TEMPER_SHIFT_C 15
#define TEMPER_SHIFT_D 18
#define TEMPER_MASK_B 0x9d2c5680UL
#define TEMPER_MASK_C 0xefc60000UL

/* State initialisation constants */
#define INIT_MULT 1812433253UL
#define INIT_MASK 0xffffffffUL

/* (int32, int32) -> double constants */
#define RAND_DBL_SHIFT_A 5
#define RAND_DBL_SHIFT_B 6
#define RAND_DBL_MUL 67108864.0
#define RAND_DBL_DIV 9007199254740992.0
/* 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */

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
void init_state(unsigned long seed, rng_state *state)
{
    int pos;
    seed &= INIT_MASK;
    state-> seed = seed;
    for (pos = 0; pos < KEY_LENGTH; pos++) {
        state->key[pos] = seed;
        seed = (INIT_MULT * (seed ^ (seed >> 30)) + pos + 1) & INIT_MASK;
    }
    state->pos = KEY_LENGTH;
    state->reversed = 0;
    state->n_twists = 0;
}

/* Optimised implementation of reference Mersenne-Twister from Random Kit. */
void twist(rng_state *state)
{
    int i;
    unsigned long y;
    for (i = 0; i < KEY_LENGTH - MID_OFFSET; i++) {
        y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
        state->key[i] = state->key[i+MID_OFFSET] ^
                        (y>>1) ^ (-(y & 1) & MATRIX_A);
    }
    for (; i < KEY_LENGTH - 1; i++) {
        y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
        state->key[i] = state->key[i+(MID_OFFSET-KEY_LENGTH)] ^
                        (y>>1) ^ (-(y & 1) & MATRIX_A);
    }
    y = (state->key[KEY_LENGTH - 1] & UPPER_MASK) |
        (state->key[0] & LOWER_MASK);
    state->key[KEY_LENGTH - 1] = state->key[MID_OFFSET - 1] ^
                                 (y >> 1) ^ (-(y & 1) & MATRIX_A);
    state->n_twists++;
}

/* Reverses twist of state i.e. reverse_twist(twist(state)) is identity map. */
void reverse_twist(rng_state *state)
{
    int i, tmp, odd;
    /* set upper bit of last key entry */
    tmp = state->key[KEY_LENGTH - 1] ^ state->key[MID_OFFSET - 1];
    state->key[KEY_LENGTH - 1] = (tmp << 1) & UPPER_MASK;
    /* partition loop over keys to avoid mod ops in index calculations */
    for (i = KEY_LENGTH - 2; i > KEY_LENGTH - MID_OFFSET - 1; i--) {
        tmp = state->key[i] ^ state->key[i + MID_OFFSET - KEY_LENGTH];
        odd = (tmp & UPPER_MASK) == UPPER_MASK;
        tmp ^= (-odd & MATRIX_A);
        tmp = (tmp << 1) | odd;
        state->key[i] = tmp & UPPER_MASK;
        state->key[i + 1] |= tmp & LOWER_MASK;
    }
    for (; i > -1; i--) {
        tmp = state->key[i] ^ state->key[i + MID_OFFSET];
        odd = (tmp & UPPER_MASK) == UPPER_MASK;
        tmp ^= (-odd & MATRIX_A);
        tmp = (tmp << 1) | odd;
        state->key[i] = tmp & UPPER_MASK;
        state->key[i + 1] |= tmp & LOWER_MASK;
    }
    /* set lower bits of first key entry */
    tmp = state->key[KEY_LENGTH - 1] ^ state->key[MID_OFFSET - 1];
    odd = (tmp & UPPER_MASK) == UPPER_MASK;
    tmp ^= (-odd & MATRIX_A);
    tmp = (tmp << 1) | odd;
    state->key[0] |= tmp & LOWER_MASK;
    state->n_twists--;
}

/*
 * Reverses direction of random number generation.
 *
 * After calling the next random value generated will be exactly equal to the
 * last generated before call, the second equal to the penultimate and so on.
 */
void reverse(rng_state *state)
{
    if (state->reversed == 0) {
        state->reversed = 1;
        state->pos--;
    }
    else {
        state->reversed = 0;
        state->pos++;
    }
}

/*
 * Generates a random integer uniformly from range [0, 2^32 - 1].
 *
 * This is the base Mersenne-Twister generator used by all other derived
 * random generator functions.
 */
unsigned long random_int32(rng_state *state)
{
    unsigned long y;
    /* if forward direction and at end of key, twist */
    if (state->reversed == 0) {
        if (state->pos == KEY_LENGTH) {
            twist(state);
            state->pos = 0;
        }
        y = state->key[state->pos++];
    }
    /* if reverse direction and at beginning of key, reverse-twist */
    else {
        if (state->pos == -1) {
            reverse_twist(state);
            state->pos = KEY_LENGTH - 1;
            /*
             * reverse_twist will not correctly recover initial key value as
             * seed when rolling back first twist therefore manually set
             */
            if (state->n_twists == 0) {
                state->key[0] = state->seed;
            }
        }
        y = state->key[state->pos--];
    }
    /* temper */
    y ^= (y >> TEMPER_SHIFT_A);
    y ^= (y << TEMPER_SHIFT_B) & TEMPER_MASK_B;
    y ^= (y << TEMPER_SHIFT_C) & TEMPER_MASK_C;
    y ^= (y >> TEMPER_SHIFT_D);
    return y;
}

/*
 * Generate a random double-precision floating point value from uniform
 * distribution on [0,1).
 */
double random_uniform(rng_state *state)
{
    long a, b;
    if (state->reversed == 0) {
        a = random_int32(state) >> RAND_DBL_SHIFT_A;
        b = random_int32(state) >> RAND_DBL_SHIFT_B;
    }
    /* swap draw order in reverse direction */
    else {
        b = random_int32(state) >> RAND_DBL_SHIFT_B;
        a = random_int32(state) >> RAND_DBL_SHIFT_A;
    }
    return (a * RAND_DBL_MUL + b) / RAND_DBL_DIV;
}

/*
 * Generate a pair of independent random double-precision floating point
 * values from the (zero-mean, unit variance) standard normal distribution.
 *
 * Unlike Random Kit this uses the original non-polar variant of the Box-Muller
 * transform which requires evaluation of the trigonometric sin/cos functions
 * and is generally slower than the polar-method. The polar-method however
 * includes a rejection sampling step which is non-trivial to make reversible.
 * Also unlike Random Kit in the interests of reversibility there is no
 * caching of one of the values in the state hence a pair are returned by
 * writing to the two provided memory locations.
 */
void random_normal_pair(rng_state *state, double *ret_1, double *ret_2)
{
    double r, theta;
    if (state->reversed == 0){
        r = sqrt(-2. * log(random_uniform(state)));
        theta = 2. * PI * random_uniform(state);
    }
    else {
        theta = 2. * PI * random_uniform(state);
        r = sqrt(-2. * log(random_uniform(state)));
    }
    *ret_1 = r * cos(theta);
    *ret_2 = r * sin(theta);
}
