#ifndef AESIMULTIPRECISION_GENERATION_H
#define AESIMULTIPRECISION_GENERATION_H

#include <random>
#include <limits>
#include <gmpxx.h>
#include <gtest/gtest.h>

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

    template<template<std::size_t> class T, std::size_t N, typename Op, typename AssignOp>
    void runCompositeTest(std::size_t lBits, std::size_t rBits, Op op, AssignOp assignOp) {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto l = getRandom(lBits), r = getRandom(rBits);
            T<N> lA = l, rA = r;
            EXPECT_EQ(op(lA, rA), op(l, r));
            assignOp(lA, rA);
            EXPECT_EQ(lA, op(l, r));
        }
    }

    template<template<std::size_t> class T, std::size_t N, typename Op, typename AssignOp>
    void runSignedCompositeTest(std::size_t lBits, std::size_t rBits, Op op, AssignOp assignOp) {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            int first = 0, second = 0;
            switch(i % 4) {
            case 0: first = 1,  second = 1;  break;
            case 1: first = -1, second = -1; break;
            case 2: first = -1, second = 1;  break;
            default: first = 1, second = -1;
            }
            const mpz_class l = first * getRandom(lBits),
                    r = second * getRandom(rBits);
            T<N> lA = l, rA = r;
            EXPECT_EQ(op(lA, rA), op(l, r));
            assignOp(lA, rA);
            EXPECT_EQ(lA, op(l, r));
        }
    }
}



#endif //AESIMULTIPRECISION_GENERATION_H
