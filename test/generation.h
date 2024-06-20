#ifndef AESIMULTIPRECISION_GENERATION_H
#define AESIMULTIPRECISION_GENERATION_H

#include <random>
#include <limits>
#include <cryptopp/integer.h>
#include <cryptopp/nbtheory.h>
#include <cryptopp/osrng.h>

namespace Generation {
    using CryptoPP::Integer;
    using Unsigned = Integer;

    static std::random_device dev;
    static std::mt19937 rng(dev());

    template <typename Out>
    Out getRandom() {
        std::uniform_int_distribution<Out> dist(std::numeric_limits<Out>::min(),
                                                std::numeric_limits<Out>::max());
        return dist(rng);
    }

    inline Unsigned getRandomWithBits(std::size_t bitLength, bool prime = false) {
        static CryptoPP::AutoSeededRandomPool prng;

        Unsigned value{};
        value.Randomize(prng,
                        Unsigned::Zero(),
                        Unsigned::Power2(bitLength),
                        prime ?
                        Unsigned::RandomNumberType::PRIME
                              :
                        Unsigned::RandomNumberType::ANY);
        return value;
    }
}



#endif //AESIMULTIPRECISION_GENERATION_H
