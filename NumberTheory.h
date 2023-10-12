#ifndef DIVISORSFINDER_NUMBERTHEORY_H
#define DIVISORSFINDER_NUMBERTHEORY_H

#include "Multiprecision.h"

/* ------------------------------------------- Greatest common divisor -------------------------------------------- */
template <std::size_t bitness>
constexpr Multiprecision<bitness> gcd(const Multiprecision<bitness>& first, const Multiprecision<bitness>& second) noexcept {
    return {};
}
template <std::size_t bFirst, std::size_t bSecond>
constexpr auto gcd(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Multiprecision<bFirst>, Multiprecision<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return gcd(first, second.template precisionCast<bFirst>());
    } else {
        return gcd(first.template precisionCast<bSecond>(), second);
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------------- Power by modulo ------------------------------------------------ */
template <std::size_t bitness>
constexpr Multiprecision<bitness> powm(const Multiprecision<bitness>& base, const Multiprecision<bitness>& power, const Multiprecision<bitness>& mod) {
    return {};
}

template <std::size_t bCommon, std::size_t bDiffer>
constexpr auto powm(const Multiprecision<bCommon>& base, const Multiprecision<bCommon>& power, const Multiprecision<bDiffer>& mod)
-> typename std::conditional<(bCommon > bDiffer), Multiprecision<bCommon>, Multiprecision<bDiffer>>::type {
    if constexpr (bCommon > bDiffer) {
        return powm(base, power, mod.template precisionCast<bCommon>());
    } else {
        return powm(base.template precisionCast<bDiffer>(), power.template precisionCast<bDiffer>(), mod);
    }
}

template <std::size_t bBase, std::size_t bPow, std::size_t bMod>
constexpr auto powm(const Multiprecision<bBase>& base, const Multiprecision<bPow>& power, const Multiprecision<bMod>& mod)
-> typename std::conditional<(bBase > bPow),
        typename std::conditional<(bBase > bMod), Multiprecision<bBase>, Multiprecision<bMod>>::value,
        typename std::conditional<(bPow > bMod), Multiprecision<bPow>, Multiprecision<bMod>>::value>::value {
    if constexpr (bBase > bPow) {
        return powm(base, power.template precisionCast<bBase>(), mod);
    } else {
        return powm(base.template precisionCast<bPow>(), power, mod);
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */

#endif //DIVISORSFINDER_NUMBERTHEORY_H
