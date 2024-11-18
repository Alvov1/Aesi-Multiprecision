#ifndef AESIMULTIPRECISION_GENERATION_H
#define AESIMULTIPRECISION_GENERATION_H

#include <random>
#include <limits>
#include <cryptopp/integer.h>
#include <cryptopp/nbtheory.h>
#include <cryptopp/osrng.h>

namespace Generation {
    using UnsPP = CryptoPP::Integer;
    using UnsGmp = mpz_class;

    static std::random_device dev;
    static std::mt19937 rng(dev());

    template <typename Out>
    Out getRandom() {
        std::uniform_int_distribution<Out> dist(std::numeric_limits<Out>::min(),
                                                std::numeric_limits<Out>::max());
        return dist(rng);
    }

    inline UnsPP getRandomWithBits(std::size_t bitLength, bool prime = false) {
        static CryptoPP::AutoSeededRandomPool prng;

        UnsPP value{};
        value.Randomize(prng,
                        UnsPP::Zero(),
                        UnsPP::Power2(bitLength),
                        prime ?
                        UnsPP::RandomNumberType::PRIME
                              :
                        UnsPP::RandomNumberType::ANY);
        return value;
    }

    inline gmp_randstate_t& getRandstate() {
        static gmp_randstate_t state;
        gmp_randinit_default (state);
        return state;
    }

    inline UnsGmp getRandom(std::size_t bitLength) {
        mpz_class result {};
        mpz_rrandomb(result.get_mpz_t(), getRandstate(), bitLength);
        return result;
    }
}



#endif //AESIMULTIPRECISION_GENERATION_H
