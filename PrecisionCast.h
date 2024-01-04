#ifndef PRECISION_CAST_H
#define PRECISION_CAST_H

/* ---------------------------------------- Different precision comparison ---------------------------------------- */
/**
 * @brief: Multiprecision comparison operator.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Bool.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator==(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool {
    if constexpr (bFirst > bSecond) {
        return (left == right.template precisionCast<bFirst>());
    } else {
        return (left.template precisionCast<bSecond>() == right);
    }
}

/**
 * @brief: Multiprecision three-way comparison method.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: AesiCMP.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto compareTo(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> AesiCMP {
    if constexpr (bFirst > bSecond) {
        return left.compareTo(right.template precisionCast<bFirst>());
    } else {
        return left.template precisionCast<bSecond>().compareTo(right);
    }
}

#if defined(__CUDACC__) || __cplusplus < 202002L || defined (DEVICE_TESTING)
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator!=(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool { return !left.operator==(right); }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool { return left.compareTo(right) == AesiCMP::less; }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<=(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool { return !left.operator>(right); }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool { return left.compareTo(right) == AesiCMP::greater; }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>=(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool { return !left.operator<(right); }

#else
    /**
     * @brief: Multiprecision three-way comparison operator.
     * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
     * @return: STD::Strong_ordering.
     * @note: Uses precision cast.
     */
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<=>(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> std::strong_ordering {
        if constexpr (bFirst > bSecond) {
            switch(left.compareTo(right.template precisionCast<bFirst>())) {
                case AesiCMP::less: return std::strong_ordering::less;
                case AesiCMP::greater: return std::strong_ordering::greater;
                case AesiCMP::equal: return std::strong_ordering::equal;
                default: return std::strong_ordering::equivalent;
            }
        } else {
            switch(left.template precisionCast<bSecond>().compareTo(right)) {
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
/**
 * @brief: Integral conversion addition operator.
 * @param: Integral number, Aesi value.
 * @return: Aesi.
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator+(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) + value;
}

/**
 * @brief: Multiprecision addition operator
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from lPrecision, rPrecision
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator+(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left + right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() + right;
    }
}

/**
 * @brief: Multiprecision assignment addition operator.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi<lPrecision>.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator+=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator+=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* --------------------------------------- Different precision subtraction ---------------------------------------- */
/**
 * @brief: Integral conversion subtraction operator.
 * @param: Integral number, Aesi value.
 * @return: Aesi.
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator-(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) - value;
}

/**
 * @brief: Multiprecision subtraction operator
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from lPrecision, rPrecision.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator-(const Aesi<bFirst>& left, const Aesi<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left - right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() - right;
    }
}

/**
 * @brief: Multiprecision assignment subtraction operator.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi<lPrecision>.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator-=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator-=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------- Different precision multiplication -------------------------------------- */
/**
 * @brief: Integral conversion multiplication operator.
 * @param: Integral number, Aesi value.
 * @return: Aesi.
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator*(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) * value;
}

/**
 * @brief: Multiprecision multiplication operator
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from lPrecision, rPrecision.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator*(const Aesi<bFirst>& left, const Aesi<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left * right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() * right;
    }
}

/**
 * @brief: Multiprecision assignment multiplication operator.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi<lPrecision>.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator*=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator*=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision division ----------------------------------------- */
/**
 * @brief: Integral conversion division operator.
 * @param: Integral number, Aesi value.
 * @return: Aesi.
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator/(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) / value;
}

/**
 * @brief: Multiprecision division operator
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from lPrecision, rPrecision.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator/(const Aesi<bFirst>& left, const Aesi<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left / right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() / right;
    }
}

/**
 * @brief: Multiprecision assignment division operator.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi<lPrecision>.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator/=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator/=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------ Different precision modulo ------------------------------------------ */
/**
 * @brief: Integral conversion modulo operator.
 * @param: Integral number, Aesi value.
 * @return: Aesi.
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator%(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) % value;
}

/**
 * @brief: Multiprecision modulo operator
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from lPrecision, rPrecision.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator%(const Aesi<bFirst>& left, const Aesi<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left % right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() % right;
    }
}

/**
 * @brief: Multiprecision assignment modulo operator.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi<lPrecision>.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator%=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator%=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Different precision XOR -------------------------------------------- */
/**
 * @brief: Integral conversion XOR operator.
 * @param: Integral number, Aesi value.
 * @return: Aesi.
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator^(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) ^ value;
}

/**
 * @brief: Multiprecision XOR operator
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from lPrecision, rPrecision.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator^(const Aesi<bFirst>& left, const Aesi<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left ^ right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() ^ right;
    }
}

/**
 * @brief: Multiprecision assignment XOR operator.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi<lPrecision>.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator^=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator^=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision AND ------------------------------------------- */
/**
 * @brief: Integral conversion AND operator.
 * @param: Integral number, Aesi value.
 * @return: Aesi.
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator&(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) & value;
}

/**
 * @brief: Multiprecision AND operator
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from lPrecision, rPrecision.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator&(const Aesi<bFirst>& left, const Aesi<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left & right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() & right;
    }
}

/**
 * @brief: Multiprecision assignment AND operator.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi<lPrecision>.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator&=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator&=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision OR -------------------------------------------- */
/**
 * @brief: Integral conversion OR operator.
 * @param: Integral number, Aesi value.
 * @return: Aesi.
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator|(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) | value;
}

/**
 * @brief: Multiprecision OR operator
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from lPrecision, rPrecision.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator|(const Aesi<bFirst>& left, const Aesi<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left | right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() | right;
    }
}

/**
 * @brief: Multiprecision assignment OR operator.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi<lPrecision>.
 * @note: Uses precision cast.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator|=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator|=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Greatest common divisor -------------------------------------------- */
/**
 * @brief: Multiprecision greatest common divisor function.
 * @param: Aesi<lPrecision> left, Aesi<rPrecision> right.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from lPrecision, rPrecision.
 * @note: Uses precision cast.
 */
template<std::size_t bFirst, std::size_t bSecond>
gpu constexpr auto gcd(const Aesi<bFirst> &left, const Aesi<bSecond> &right)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return Aesi<bFirst>::gcd(left, right.template precisionCast<bFirst>());
    } else {
        return Aesi<bSecond>::gcd(left.template precisionCast<bSecond>(), right);
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------------- Power by modulo ------------------------------------------------ */
/**
 * @brief: Multiprecision power by modulo function.
 * @param: Aesi<bPrecision> base, Aesi<pPrecision> power, Aesi<mPrecision> modulo.
 * @return: Aesi.
 * @details: Returns Aesi with greater precision from bPrecision, pPrecision, mPrecision.
 * @note: Uses precision cast.
 */
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
            return Aesi<bDiffer>::powm(base.template precisionCast<bDiffer>(), power.template precisionCast<bDiffer>(), mod);
        }
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */

#endif //PRECISION_CAST_H
