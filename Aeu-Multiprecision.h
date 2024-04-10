#ifndef AEU_PRECISION_CAST
#define AEU_PRECISION_CAST

/// @cond HIDE_INCLUDES
#include "Aeu.h"
/// @endcond

/**
 * @file Aeu-Multiprecision.h
 * @brief List of operations for two or more integers with different precision
 * @details The library was designed to support operations with numbers of different precision. Each function in this list
 * receives two or more numbers with different precision. It performs precision cast of number with lower precision to
 * greater precision and calls corresponding arithmetic operator. Please note that precision cast operator requires
 * redundant copying and may be slow. Take a look at the main page to find out more.
 */

/* ---------------------------------------- Different precision comparison ---------------------------------------- */
/**
 * @brief Multiprecision comparison operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Bool
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator==(const Aeu<bFirst>& left, const Aeu<bSecond>& right) noexcept -> bool {
    if constexpr (bFirst > bSecond) {
        return (left == right.template precisionCast<bFirst>());
    } else {
        return (left.template precisionCast<bSecond>() == right);
    }
}

/**
 * @brief Multiprecision three-way comparison method
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Comparison
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto compareTo(const Aeu<bFirst>& left, const Aeu<bSecond>& right) noexcept -> Comparison {
    if constexpr (bFirst > bSecond) {
        return left.compareTo(right.template precisionCast<bFirst>());
    } else {
        return left.template precisionCast<bSecond>().compareTo(right);
    }
}

#if (defined(__CUDACC__) || __cplusplus < 202002L || defined (DEVICE_TESTING)) && !defined DOXYGEN_SKIP
/**
     * @brief Oldstyle binary comparison operator(s). Used inside CUDA cause it does not support <=> on preCpp20
     */
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator!=(const Aeu<bFirst>& left, const Aeu<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return !left.operator==(right.template precisionCast<bFirst>());
        } else {
            return !left.template precisionCast<bSecond>().operator==(right);
        };
    }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<(const Aeu<bFirst>& left, const Aeu<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return left.compareTo(right.template precisionCast<bFirst>()) == Comparison::less;
        } else {
            return left.template precisionCast<bSecond>().compareTo(right) == Comparison::less;
        };
    }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<=(const Aeu<bFirst>& left, const Aeu<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return !left.operator>(right.template precisionCast<bFirst>());
        } else {
            return !left.template precisionCast<bSecond>().operator>(right);
        };
    }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>(const Aeu<bFirst>& left, const Aeu<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return left.compareTo(right.template precisionCast<bFirst>()) == Comparison::greater;
        } else {
            return left.template precisionCast<bSecond>().compareTo(right) == Comparison::greater;
        };
    }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>=(const Aeu<bFirst>& left, const Aeu<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return !left.operator<(right.template precisionCast<bFirst>());
        } else {
            return !left.template precisionCast<bSecond>().operator<(right);
        };
    }
#else
/**
 * @brief Multiprecision three-way comparison operator.
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right.
 * @return STD::Strong_ordering.
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator<=>(const Aeu<bFirst>& left, const Aeu<bSecond>& right) noexcept -> std::strong_ordering {
    if constexpr (bFirst > bSecond) {
        switch(left.compareTo(right.template precisionCast<bFirst>())) {
            case Comparison::less: return std::strong_ordering::less;
            case Comparison::greater: return std::strong_ordering::greater;
            case Comparison::equal: return std::strong_ordering::equal;
            default: return std::strong_ordering::equivalent;
        }
    } else {
        switch(left.template precisionCast<bSecond>().compareTo(right)) {
            case Comparison::less: return std::strong_ordering::less;
            case Comparison::greater: return std::strong_ordering::greater;
            case Comparison::equal: return std::strong_ordering::equal;
            default: return std::strong_ordering::equivalent;
        }
    }
}
#endif
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision addition ----------------------------------------- */
/**
 * @brief Multiprecision addition operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu
 * @details Returns Aeu with the highest precision between lPrecision, rPrecision
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator+(const Aeu<bFirst>& left, const Aeu<bSecond>& right) noexcept
-> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left + right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() + right;
    }
}

/**
 * @brief Multiprecision assignment addition operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator+=(Aeu<bFirst>& left, const Aeu<bSecond>& right) -> Aeu<bFirst>& {
    return left.operator+=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* --------------------------------------- Different precision subtraction ---------------------------------------- */
/**
 * @brief Multiprecision subtraction operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu
 * @details Returns Aeu with the highest precision between lPrecision, rPrecision
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator-(const Aeu<bFirst>& left, const Aeu<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left - right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() - right;
    }
}

/**
 * @brief Multiprecision assignment subtraction operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator-=(Aeu<bFirst>& left, const Aeu<bSecond>& right) -> Aeu<bFirst>& {
    return left.operator-=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------- Different precision multiplication -------------------------------------- */
/**
 * @brief Multiprecision multiplication operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu
 * @details Returns Aeu with the highest precision between lPrecision, rPrecision
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator*(const Aeu<bFirst>& left, const Aeu<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left * right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() * right;
    }
}

/**
 * @brief Multiprecision assignment multiplication operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator*=(Aeu<bFirst>& left, const Aeu<bSecond>& right) -> Aeu<bFirst>& {
    return left.operator*=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision division ----------------------------------------- */
/**
 * @brief Multiprecision division operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu
 * @details Returns Aeu with the highest precision between lPrecision, rPrecision
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator/(const Aeu<bFirst>& left, const Aeu<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left / right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() / right;
    }
}

/**
 * @brief Multiprecision assignment division operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator/=(Aeu<bFirst>& left, const Aeu<bSecond>& right) -> Aeu<bFirst>& {
    return left.operator/=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------ Different precision modulo ------------------------------------------ */
/**
 * @brief Multiprecision modulo operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu
 * @details Returns Aeu with the highest precision between lPrecision, rPrecision
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator%(const Aeu<bFirst>& left, const Aeu<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left % right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() % right;
    }
}

/**
 * @brief Multiprecision assignment modulo operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator%=(Aeu<bFirst>& left, const Aeu<bSecond>& right) -> Aeu<bFirst>& {
    return left.operator%=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Different precision XOR -------------------------------------------- */
/**
 * @brief Multiprecision bitwise XOR operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu
 * @details Returns Aeu with the highest precision between lPrecision, rPrecision
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator^(const Aeu<bFirst>& left, const Aeu<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left ^ right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() ^ right;
    }
}

/**
 * @brief Multiprecision assignment bitwise XOR operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator^=(Aeu<bFirst>& left, const Aeu<bSecond>& right) -> Aeu<bFirst>& {
    return left.operator^=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision AND ------------------------------------------- */
/**
 * @brief Multiprecision bitwise AND operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu
 * @details Returns Aeu with the highest precision between lPrecision, rPrecision
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator&(const Aeu<bFirst>& left, const Aeu<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left & right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() & right;
    }
}

/**
 * @brief Multiprecision assignment bitwise AND operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator&=(Aeu<bFirst>& left, const Aeu<bSecond>& right) -> Aeu<bFirst>& {
    return left.operator&=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision OR -------------------------------------------- */
/**
 * @brief Multiprecision bitwise OR operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu
 * @details Returns Aeu with the highest precision between lPrecision, rPrecision
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator|(const Aeu<bFirst>& left, const Aeu<bSecond>& right)
-> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return left | right.template precisionCast<bFirst>();
    } else {
        return left.template precisionCast<bSecond>() | right;
    }
}

/**
 * @brief Multiprecision assignment bitwise OR operator
 * @param Aeu<lPrecision> left
 * @param Aeu<rPrecision> right
 * @return Aeu<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator|=(Aeu<bFirst>& left, const Aeu<bSecond>& right) -> Aeu<bFirst>& {
    return left.operator|=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


namespace AeuMultiprecision {
/* ------------------------------------------- Greatest common divisor -------------------------------------------- */
    /**
     * @brief Multiprecision greatest common divisor
     * @param Aeu<lPrecision> left
     * @param Aeu<rPrecision> right
     * @return Aeu
     * @details Returns Aeu with the highest precision between lPrecision, rPrecision
     */
    template<std::size_t bFirst, std::size_t bSecond>
    gpu constexpr auto gcd(const Aeu<bFirst> &left, const Aeu<bSecond> &right)
    -> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
        if constexpr (bFirst > bSecond) {
            return Aeu<bFirst>::gcd(left, right.template precisionCast<bFirst>());
        } else {
            return Aeu<bSecond>::gcd(left.template precisionCast<bSecond>(), right);
        }
    }
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Least common multiplier -------------------------------------------- */
    /**
     * @brief Multiprecision greatest common divisor
     * @param Aeu<lPrecision> left
     * @param Aeu<rPrecision> right
     * @return Aeu
     * @details Returns Aeu with the highest precision between lPrecision, rPrecision
     */
    template<std::size_t bFirst, std::size_t bSecond>
    gpu constexpr auto lcm(const Aeu<bFirst> &left, const Aeu<bSecond> &right)
    -> typename std::conditional<(bFirst > bSecond), Aeu<bFirst>, Aeu<bSecond>>::type {
        if constexpr (bFirst > bSecond) {
            return Aeu<bFirst>::lcm(left, right.template precisionCast<bFirst>());
        } else {
            return Aeu<bSecond>::lcm(left.template precisionCast<bSecond>(), right);
        }
    }
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------------- Power by modulo ------------------------------------------------ */
    /**
     * @brief Multiprecision power by modulo helper
     * @param Aeu<precFirst> base
     * @param Aeu<precFirst> power
     * @param Aeu<precSecond> modulo
     * @return Aeu
     * @details Returns Aeu with the highest precision between base/power and modulo.
     */
    template<std::size_t bCommon, std::size_t bDiffer> requires (bCommon != bDiffer)
    gpu constexpr auto powm(const Aeu<bCommon>& base, const Aeu<bCommon>& power, const Aeu<bDiffer>& mod) {
        if constexpr (bCommon > bDiffer) {
            return Aeu<bCommon>::powm(base, power, mod.template precisionCast<bCommon>());
        } else {
            return Aeu<bDiffer>::powm(base.template precisionCast<bDiffer>(), power.template precisionCast<bDiffer>(), mod);
        }
    }

    /**
     * @brief Multiprecision power by modulo helper
     * @param Aeu<precFirst> base
     * @param Aeu<precFirst> power
     * @param Aeu<precSecond> modulo
     * @return Aeu
     * @details Returns Aeu with the highest precision between base/power and modulo.
     */
    template<std::size_t bCommon, std::size_t bDiffer> requires (bCommon != bDiffer)
    gpu constexpr auto powm(const Aeu<bCommon>& base, const Aeu<bDiffer>& power, const Aeu<bCommon>& mod) {
        if constexpr (bCommon > bDiffer) {
            return Aeu<bCommon>::powm(base, power.template precisionCast<bCommon>(), mod);
        } else {
            return Aeu<bDiffer>::powm(base.template precisionCast<bDiffer>(),
                                      power, mod.template precisionCast<bDiffer>());
        }
    }

    /* Differ-Common-Common */
    template<std::size_t bCommon, std::size_t bDiffer> requires (bCommon != bDiffer)
    gpu constexpr auto powm(const Aeu<bDiffer>& base, const Aeu<bCommon>& power, const Aeu<bCommon>& mod) {
        if constexpr (bCommon > bDiffer) {
            return Aeu<bCommon>::powm(base.template precisionCast<bCommon>(), power, mod);
        } else {
            return Aeu<bDiffer>::powm(base, power.template precisionCast<bDiffer>(), mod.template precisionCast<bDiffer>());
        }
    }

    /**
     * @brief Multiprecision power by modulo
     * @param Aeu<bPrecision> base
     * @param Aeu<pPrecision> power
     * @param Aeu<mPrecision> modulo
     * @return Aeu
     * @details Returns Aeu with the highest precision between base, power and modulo.
     * @note Be REALLY careful with overflow. The operation is calculated with an precision equal to the highest precision among the function parameters.
     */
    template<std::size_t bBase, std::size_t bPow, std::size_t bMod> requires (bBase != bPow && bPow != bMod && bBase != bMod)
    gpu constexpr auto powm(const Aeu<bBase> &base, const Aeu<bPow> &power, const Aeu<bMod> &mod)
    -> typename std::conditional<(bBase > bPow),
            typename std::conditional<(bBase > bMod), Aeu<bBase>, Aeu<bMod>>::type,
            typename std::conditional<(bPow > bMod), Aeu<bPow>, Aeu<bMod>>::type>::type {
        if constexpr (bBase > bPow) {
            return powm<bBase, bMod>(base, power.template precisionCast<bBase>(), mod);
        } else {
            return powm<bPow, bMod>(base.template precisionCast<bPow>(), power, mod);
        }
    }
/* ---------------------------------------------------------------------------------------------------------------- */
}

#endif //AEU_PRECISION_CAST
