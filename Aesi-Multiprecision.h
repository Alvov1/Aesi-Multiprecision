#ifndef PRECISION_CAST_H
#define PRECISION_CAST_H

/**
 * @file Aesi-Multiprecision.h
 * @brief List of operations for two or more integers with different precision
 * @details The library was designed to support operations with numbers of different precision. Each function in this list
 * receives two or more numbers with different precision. It performs precision cast of number with lower precision to
 * greater precision and calls corresponding arithmetic operator. Please note that precision cast operator requires
 * redundant copying and may be slow. Take a look at the main page to find out more.
 */

/* ---------------------------------------- Different precision comparison ---------------------------------------- */
/**
 * @brief Multiprecision comparison operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Bool
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
 * @brief Multiprecision three-way comparison method
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return AesiCMP
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto compareTo(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> AesiCMP {
    if constexpr (bFirst > bSecond) {
        return left.compareTo(right.template precisionCast<bFirst>());
    } else {
        return left.template precisionCast<bSecond>().compareTo(right);
    }
}

#if (defined(__CUDACC__) || __cplusplus < 202002L || defined (DEVICE_TESTING)) && !defined DOXYGEN_SKIP
    /**
     * @brief Oldstyle binary comparison operator(s). Used inside CUDA cause it does not support <=> on device
     */
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator!=(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return !left.operator==(right.template precisionCast<bFirst>());
        } else {
            return !left.template precisionCast<bSecond>().operator==(right);
        };
    }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return left.compareTo(right.template precisionCast<bFirst>()) == AesiCMP::less;
        } else {
            return left.template precisionCast<bSecond>().compareTo(right) == AesiCMP::less;
        };
    }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator<=(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return !left.operator>(right.template precisionCast<bFirst>());
        } else {
            return !left.template precisionCast<bSecond>().operator>(right);
        };
    }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return left.compareTo(right.template precisionCast<bFirst>()) == AesiCMP::greater;
        } else {
            return left.template precisionCast<bSecond>().compareTo(right) == AesiCMP::greater;
        };
    }
    template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
    gpu constexpr auto operator>=(const Aesi<bFirst>& left, const Aesi<bSecond>& right) noexcept -> bool {
        if constexpr (bFirst > bSecond) {
            return !left.operator<(right.template precisionCast<bFirst>());
        } else {
            return !left.template precisionCast<bSecond>().operator<(right);
        };
    }
#else
    /**
     * @brief Multiprecision three-way comparison operator.
     * @param Aesi<lPrecision> left
     * @param Aesi<rPrecision> right.
     * @return STD::Strong_ordering.
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
 * @brief Multiprecision addition operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
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
 * @brief Multiprecision assignment addition operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator+=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator+=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* --------------------------------------- Different precision subtraction ---------------------------------------- */
/**
 * @brief Multiprecision subtraction operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
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
 * @brief Multiprecision assignment subtraction operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator-=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator-=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------- Different precision multiplication -------------------------------------- */
/**
 * @brief Multiprecision multiplication operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
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
 * @brief Multiprecision assignment multiplication operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator*=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator*=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision division ----------------------------------------- */
/**
 * @brief Multiprecision division operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
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
 * @brief Multiprecision assignment division operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator/=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator/=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------ Different precision modulo ------------------------------------------ */
/**
 * @brief Multiprecision modulo operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
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
 * @brief Multiprecision assignment modulo operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator%=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator%=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Different precision XOR -------------------------------------------- */
/**
 * @brief Multiprecision bitwise XOR operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
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
 * @brief Multiprecision assignment bitwise XOR operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator^=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator^=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision AND ------------------------------------------- */
/**
 * @brief Multiprecision bitwise AND operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
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
 * @brief Multiprecision assignment bitwise AND operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator&=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator&=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision OR -------------------------------------------- */
/**
 * @brief Multiprecision bitwise OR operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
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
 * @brief Multiprecision assignment bitwise OR operator
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi<lPrecision>
 */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator|=(Aesi<bFirst>& left, const Aesi<bSecond>& right) -> Aesi<bFirst>& {
    return left.operator|=(right.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


namespace AesiMultiprecision {
/* ------------------------------------------- Greatest common divisor -------------------------------------------- */
    /**
 * @brief Multiprecision greatest common divisor
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
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


/* ------------------------------------------- Least common multiplier -------------------------------------------- */
/**
 * @brief Multiprecision greatest common divisor
 * @param Aesi<lPrecision> left
 * @param Aesi<rPrecision> right
 * @return Aesi
 * @details Returns Aesi with the highest precision between lPrecision, rPrecision
 */
    template<std::size_t bFirst, std::size_t bSecond>
    gpu constexpr auto lcm(const Aesi<bFirst> &left, const Aesi<bSecond> &right)
    -> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
        if constexpr (bFirst > bSecond) {
            return Aesi<bFirst>::lcm(left, right.template precisionCast<bFirst>());
        } else {
            return Aesi<bSecond>::lcm(left.template precisionCast<bSecond>(), right);
        }
    }
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------------- Power by modulo ------------------------------------------------ */
    /**
     * @brief Multiprecision power by modulo helper
     * @param Aesi<precFirst> base
     * @param Aesi<precFirst> power
     * @param Aesi<precSecond> modulo
     * @return Aesi
     * @details Returns Aesi with the highest precision between base/power and modulo.
     */
    template<std::size_t bCommon, std::size_t bDiffer> requires (bCommon != bDiffer)
    gpu constexpr auto powm(const Aesi<bCommon>& base, const Aesi<bCommon>& power, const Aesi<bDiffer>& mod) {
        if constexpr (bCommon > bDiffer) {
            return Aesi<bCommon>::powm(base, power, mod.template precisionCast<bCommon>());
        } else {
            return Aesi<bDiffer>::powm(base.template precisionCast<bDiffer>(),
                                       power.template precisionCast<bDiffer>(), mod);
        }
    }

    /**
     * @brief Multiprecision power by modulo helper
     * @param Aesi<precFirst> base
     * @param Aesi<precFirst> power
     * @param Aesi<precSecond> modulo
     * @return Aesi
     * @details Returns Aesi with the highest precision between base/power and modulo.
     */
    template<std::size_t bCommon, std::size_t bDiffer> requires (bCommon != bDiffer)
    gpu constexpr auto powm(const Aesi<bCommon>& base, const Aesi<bDiffer>& power, const Aesi<bCommon>& mod) {
        if constexpr (bCommon > bDiffer) {
            return Aesi<bCommon>::powm(base, power.template precisionCast<bCommon>(), mod);
        } else {
            return Aesi<bDiffer>::powm(base.template precisionCast<bDiffer>(),
                                       power, mod.template precisionCast<bDiffer>());
        }
    }

    /* Differ-Common-Common */
    template<std::size_t bCommon, std::size_t bDiffer> requires (bCommon != bDiffer)
    gpu constexpr auto powm(const Aesi<bDiffer>& base, const Aesi<bCommon>& power, const Aesi<bCommon>& mod) {
        if constexpr (bCommon > bDiffer) {
            return Aesi<bCommon>::powm(base.template precisionCast<bCommon>(), power, mod);
        } else {
            return Aesi<bDiffer>::powm(base, power.template precisionCast<bDiffer>(),
                                       mod.template precisionCast<bDiffer>());
        }
    }

    /**
     * @brief Multiprecision power by modulo
     * @param Aesi<bPrecision> base
     * @param Aesi<pPrecision> power
     * @param Aesi<mPrecision> modulo
     * @return Aesi
     * @details Returns Aesi with the highest precision between base, power and modulo.
     * @note Be REALLY careful with overflow. The operation is calculated with an precision equal to the highest precision among the function parameters.
     */
    template<std::size_t bBase, std::size_t bPow, std::size_t bMod> requires (bBase != bPow && bPow != bMod && bBase != bMod)
    gpu constexpr auto powm(const Aesi<bBase> &base, const Aesi<bPow> &power, const Aesi<bMod> &mod)
    -> typename std::conditional<(bBase > bPow),
            typename std::conditional<(bBase > bMod), Aesi<bBase>, Aesi<bMod>>::type,
            typename std::conditional<(bPow > bMod), Aesi<bPow>, Aesi<bMod>>::type>::type {
        if constexpr (bBase > bPow) {
            return powm<bBase, bMod>(base, power.template precisionCast<bBase>(), mod);
        } else {
            return powm<bPow, bMod>(base.template precisionCast<bPow>(), power, mod);
        }
    }
/* ---------------------------------------------------------------------------------------------------------------- */
}
#endif //PRECISION_CAST_H
