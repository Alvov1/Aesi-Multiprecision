#ifndef PRECISION_CAST_H
#define PRECISION_CAST_H

/* ---------------------------------------- Different precision comparison ---------------------------------------- */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr bool operator==(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second) noexcept {
    if constexpr (bFirst > bSecond) {
        return (first == second.template precisionCast<bFirst>());
    } else {
        return (first.template precisionCast<bSecond>() == second);
    }
}


template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto compareTo(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second) noexcept -> AesiCMP {
    if constexpr (bFirst > bSecond) {
        return first.compareTo(second.template precisionCast<bFirst>());
    } else {
        return first.template precisionCast<bSecond>().compareTo(second);
    }
}

#if defined(__CUDACC__) || __cplusplus < 202002L || defined (DEVICE_TESTING)
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator!=(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second) noexcept -> bool { return !first.operator==(second); }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second) noexcept -> bool { return first.compareTo(second) == AesiCMP::less; }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<=(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second) noexcept -> bool { return !first.operator>(second); }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second) noexcept -> bool { return first.compareTo(second) == AesiCMP::greater; }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>=(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second) noexcept -> bool { return !first.operator<(second); }

#else
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<=>(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second) noexcept -> std::strong_ordering {
        if constexpr (bFirst > bSecond) {
            switch(first.compareTo(second.template precisionCast<bFirst>())) {
                case AesiCMP::less: return std::strong_ordering::less;
                case AesiCMP::greater: return std::strong_ordering::greater;
                case AesiCMP::equal: return std::strong_ordering::equal;
                default: return std::strong_ordering::equivalent;
            }
        } else {
            switch(first.template precisionCast<bSecond>().compareTo(second)) {
                case AesiCMP::less: return std::strong_ordering::less;
                case AesiCMP::greater: return std::strong_ordering::greater;
                case AesiCMP::equal: return std::strong_ordering::equal;
                default: return std::strong_ordering::equivalent;
            }
        }
    }
#endif
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision addition ----------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator+(Integral number, const AesiMP<bitness>& value) noexcept {
    return AesiMP<bitness>(number) + value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator+(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second) noexcept
-> typename std::conditional<(bFirst > bSecond), AesiMP<bFirst>, AesiMP<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first + second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() + second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator+=(AesiMP<bFirst>& first, const AesiMP<bSecond>& second) -> AesiMP<bFirst>& {
    return first.operator+=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* --------------------------------------- Different precision subtraction ---------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator-(Integral number, const AesiMP<bitness>& value) noexcept {
    return AesiMP<bitness>(number) - value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator-(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), AesiMP<bFirst>, AesiMP<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first - second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() - second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator-=(AesiMP<bFirst>& first, const AesiMP<bSecond>& second) -> AesiMP<bFirst>& {
    return first.operator-=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------- Different precision multiplication -------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator*(Integral number, const AesiMP<bitness>& value) noexcept {
    return AesiMP<bitness>(number) * value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator*(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), AesiMP<bFirst>, AesiMP<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first * second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() * second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator*=(AesiMP<bFirst>& first, const AesiMP<bSecond>& second) -> AesiMP<bFirst>& {
    return first.operator*=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision division ----------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator/(Integral number, const AesiMP<bitness>& value) noexcept {
    return AesiMP<bitness>(number) / value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator/(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), AesiMP<bFirst>, AesiMP<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first / second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() / second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator/=(AesiMP<bFirst>& first, const AesiMP<bSecond>& second) -> AesiMP<bFirst>& {
    return first.operator/=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------ Different precision modulo ------------------------------------------ */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator%(Integral number, const AesiMP<bitness>& value) noexcept {
    return AesiMP<bitness>(number) % value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator%(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), AesiMP<bFirst>, AesiMP<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first % second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() % second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator%=(AesiMP<bFirst>& first, const AesiMP<bSecond>& second) -> AesiMP<bFirst>& {
    return first.operator%=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Different precision XOR -------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator^(Integral number, const AesiMP<bitness>& value) noexcept {
    return AesiMP<bitness>(number) ^ value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator^(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), AesiMP<bFirst>, AesiMP<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first ^ second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() ^ second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator^=(AesiMP<bFirst>& first, const AesiMP<bSecond>& second) -> AesiMP<bFirst>& {
    return first.operator^=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision AND ------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator&(Integral number, const AesiMP<bitness>& value) noexcept {
    return AesiMP<bitness>(number) & value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator&(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), AesiMP<bFirst>, AesiMP<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first & second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() & second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator&=(AesiMP<bFirst>& first, const AesiMP<bSecond>& second) -> AesiMP<bFirst>& {
    return first.operator&=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision OR -------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator|(Integral number, const AesiMP<bitness>& value) noexcept {
    return AesiMP<bitness>(number) | value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator|(const AesiMP<bFirst>& first, const AesiMP<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), AesiMP<bFirst>, AesiMP<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first | second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() | second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator|=(AesiMP<bFirst>& first, const AesiMP<bSecond>& second) -> AesiMP<bFirst>& {
    return first.operator|=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Greatest common divisor -------------------------------------------- */
template<std::size_t bFirst, std::size_t bSecond>
gpu constexpr auto gcd(const AesiMP<bFirst> &first, const AesiMP<bSecond> &second)
-> typename std::conditional<(bFirst > bSecond), AesiMP<bFirst>, AesiMP<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return AesiMP<bFirst>::gcd(first, second.template precisionCast<bFirst>());
    } else {
        return AesiMP<bSecond>::gcd(first.template precisionCast<bSecond>(), second);
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------------- Power by modulo ------------------------------------------------ */
template<std::size_t bBase, std::size_t bPow, std::size_t bMod>
gpu constexpr auto powm(const AesiMP<bBase> &base, const AesiMP<bPow> &power, const AesiMP<bMod> &mod)
-> typename std::conditional<(bBase > bPow),
        typename std::conditional<(bBase > bMod), AesiMP<bBase>, AesiMP<bMod>>::value,
        typename std::conditional<(bPow > bMod), AesiMP<bPow>, AesiMP<bMod>>::value>::value {
    if constexpr (bBase > bPow) {
        return powm(base, power.template precisionCast<bBase>(), mod);
    } else {
        return powm(base.template precisionCast<bPow>(), power, mod);
    }
}

namespace {
    template<std::size_t bCommon, std::size_t bDiffer>
    gpu constexpr auto powm(const AesiMP<bCommon> &base, const AesiMP<bCommon> &power, const AesiMP<bDiffer> &mod)
    -> typename std::conditional<(bCommon > bDiffer), AesiMP<bCommon>, AesiMP<bDiffer>>::type {
        if constexpr (bCommon > bDiffer) {
            return AesiMP<bCommon>::powm(base, power, mod.template precisionCast<bCommon>());
        } else {
            return AesiMP<bDiffer>::powm(base.template precisionCast<bDiffer>(),
                                         power.template precisionCast<bDiffer>(), mod);
        }
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */

#endif //PRECISION_CAST_H
