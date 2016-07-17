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
 * The twist, random_int32 and seed functions algorithms and the original
 * design of the Mersenne Twister RNG:
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

/* Internal random number generator state, variant of Random Kit rk_state. */
typedef struct rng_state_
{
    int reverse; /* ==1: apply reversed state updates */
    unsigned long seed; /* integer seed used to initialise state */
    int n_twists; /* number of twist operations performed */
    unsigned long key[KEY_LENGTH]; /* Mersenne-Twister state */
    int pos; /* current position in key array */
    int has_gauss; /* !=0: gauss contains a cached Gaussian sample */
    double gauss; /* has_gauss=1: contains cached Gaussian sample */
} rng_state;

/* Initialise generator state from an integer seed. */
void init_state(unsigned long seed, rng_state *state)
{
    int pos;
    state-> seed = seed;
    seed &= INIT_MASK;
    for (pos = 0; pos < KEY_LENGTH; pos++) {
        state->key[pos] = seed;
        seed = (INIT_MULT * (seed ^ (seed >> 30)) + pos + 1) & INIT_MASK;
    }
    state->n_twists = 0;
    state->pos = KEY_LENGTH;
    state->gauss = 0;
    state->has_gauss = 0;
    state->reverse = 0;
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
    /* parition loop over keys to avoid mod ops in index calculations */
    for (i = KEY_LENGTH - 2; i > KEY_LENGTH - MID_OFFSET; i--) {
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
    if (state->reverse == 0) {
        state->reverse = 1;
        /* need to account for extra draw if Gaussian value is cached */
        if (state->pos > state->has_gauss) {
            state->pos -= 1 + state->has_gauss;
            state->has_gauss = 0;
        }
        else {
            state->pos = KEY_LENGTH - 1 + state->pos - state->has_gauss;
            reverse_twist(state);
            state->has_gauss = 0;
            /*
             * reverse_twist will not correctly recover initial key value as
             * seed when rolling back first twist therefore manually set
             */
            if (state->n_twists == 0) {
                state->key[0] = state->seed;
            }
        }
    }
    else {
        state->reverse = 0;
        if (state->pos < KEY_LENGTH - state->has_gauss - 1) {
            state->pos += 1 + state->has_gauss;
            state->has_gauss = 0;
        }
        else {
            state->pos = KEY_LENGTH - 1 - state->pos + state->has_gauss;
            twist(state);
            state->has_gauss = 0;
        }
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
    if (state->reverse == 0) {
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
double random_double(rng_state *state)
{
    long a, b;
    if (state->reverse == 0) {
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
 * Generate a random double-precision floating point value from zero-mean
 * and unit variance Gaussian distribution.
 *
 * As in Random Kit uses polar variant of Box-Muller transform with rejection
 * sampling to generate initial sample from uniform on unit disk, with one
 * of two generated samples cached in state.
 */
double random_gauss(rng_state *state)
{
    if (state->has_gauss) {
        const double tmp = state->gauss;
        state->gauss = 0;
        state->has_gauss = 0;
        return tmp;
    }
    else {
        double f, x1, x2, r2;
        /* rejection sampling to get uniform on unit disk */
        do {
             /* symmetric in x1 and x2 - therefore no reversing needed */
            x1 = 2.0 * random_double(state) - 1.0;
            x2 = 2.0 * random_double(state) - 1.0;
            r2 = x1 * x1 + x2 * x2;
        }
        while (r2 >= 1.0 || r2 == 0.0);
        /* polar Box-Muller transform */
        f = sqrt(-2.0 * log(r2) / r2);
        /* keep one of samples for next call */
        state->has_gauss = 1;
        state->gauss = f * x1;
        return f * x2;
    }
}
