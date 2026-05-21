#ifndef AESIMULTIPRECISION_GENERATION_H
#define AESIMULTIPRECISION_GENERATION_H

#include <random>
#include <limits>
#include <gmpxx.h>

namespace Generation {
    template <typename F>
    inline void forEachPrecision(F f) {
        f.template operator()<256>();
        f.template operator()<512>();
        f.template operator()<1024>();
    }

    using UnsGmp = mpz_class;

    static std::random_device dev;
    static std::mt19937 rng(dev());

    template <typename Out>
    Out getRandom() {
        std::uniform_int_distribution<Out> dist(std::numeric_limits<Out>::min(), std::numeric_limits<Out>::max());
        return dist(rng);
    }

    inline gmp_randstate_t& getRandstate() {
        static gmp_randstate_t state;
        static bool initialized = false;
        if(!initialized) {
            gmp_randinit_default(state);
            gmp_randseed_ui(state, dev());
            initialized = true;
        }
        return state;
    }

    inline UnsGmp getRandom(std::size_t bitLength) {
        mpz_class result {};
        mpz_urandomb(result.get_mpz_t(), getRandstate(), bitLength);
        return result;
    }
}



#endif //AESIMULTIPRECISION_GENERATION_H
