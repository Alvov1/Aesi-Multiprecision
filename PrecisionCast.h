#ifndef PRECISION_CAST_H
#define PRECISION_CAST_H

/* ---------------------------------------- Different precision comparison ---------------------------------------- */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr bool operator==(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept {
    if constexpr (bFirst > bSecond) {
        return (first == second.template precisionCast<bFirst>());
    } else {
        return (first.template precisionCast<bSecond>() == second);
    }
}


template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto compareTo(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept -> AesiCMP {
    if constexpr (bFirst > bSecond) {
        return first.compareTo(second.template precisionCast<bFirst>());
    } else {
        return first.template precisionCast<bSecond>().compareTo(second);
    }
}

#if defined(__CUDACC__) || __cplusplus < 202002L || defined (DEVICE_TESTING)
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator!=(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept -> bool { return !first.operator==(second); }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept -> bool { return first.compareTo(second) == AesiCMP::less; }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<=(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept -> bool { return !first.operator>(second); }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept -> bool { return first.compareTo(second) == AesiCMP::greater; }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>=(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept -> bool { return !first.operator<(second); }

#else
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<=>(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept -> std::strong_ordering {
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
gpu constexpr auto operator+(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) + value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator+(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first + second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() + second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator+=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator+=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* --------------------------------------- Different precision subtraction ---------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator-(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) - value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator-(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first - second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() - second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator-=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator-=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------- Different precision multiplication -------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator*(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) * value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator*(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first * second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() * second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator*=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator*=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision division ----------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator/(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) / value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator/(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first / second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() / second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator/=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator/=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------ Different precision modulo ------------------------------------------ */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator%(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) % value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator%(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first % second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() % second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator%=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator%=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Different precision XOR -------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator^(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) ^ value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator^(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first ^ second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() ^ second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator^=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator^=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision AND ------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator&(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) & value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator&(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first & second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() & second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator&=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator&=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision OR -------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator|(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) | value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator|(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first | second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() | second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator|=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator|=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Greatest common divisor -------------------------------------------- */
template<std::size_t bFirst, std::size_t bSecond>
gpu constexpr auto gcd(const Aesi<bFirst> &first, const Aesi<bSecond> &second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return Aesi<bFirst>::gcd(first, second.template precisionCast<bFirst>());
    } else {
        return Aesi<bSecond>::gcd(first.template precisionCast<bSecond>(), second);
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------------- Power by modulo ------------------------------------------------ */
template<std::size_t bBase, std::size_t bPow, std::size_t bMod>
gpu constexpr auto powm(const Aesi<bBase> &base, const Aesi<bPow> &power, const Aesi<bMod> &mod)
-> typename std::conditional<(bBase > bPow),
        typename std::conditional<(bBase > bMod), Aesi<bBase>, Aesi<bMod>>::value,
        typename std::conditional<(bPow > bMod), Aesi<bPow>, Aesi<bMod>>::value>::value {
    if constexpr (bBase > bPow) {
        return powm(base, power.template precisionCast<bBase>(), mod);
    } else {
        return powm(base.template precisionCast<bPow>(), power, mod);
    }
}

namespace {
    template<std::size_t bCommon, std::size_t bDiffer>
    gpu constexpr auto powm(const Aesi<bCommon> &base, const Aesi<bCommon> &power, const Aesi<bDiffer> &mod)
    -> typename std::conditional<(bCommon > bDiffer), Aesi<bCommon>, Aesi<bDiffer>>::type {
        if constexpr (bCommon > bDiffer) {
            return Aesi<bCommon>::powm(base, power, mod.template precisionCast<bCommon>());
        } else {
            return Aesi<bDiffer>::powm(base.template precisionCast<bDiffer>(),
                                       power.template precisionCast<bDiffer>(), mod);
        }
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */

#endif //PRECISION_CAST_H
