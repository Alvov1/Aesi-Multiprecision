#include <iostream>
#include <vector>
#include <primesieve.hpp>
#include <AesiMultiprecision/Aesi.h>

/* Creator: Alexander Lvov
 * Details: Extended Euclidean algorithm for computing Bezout coefficients.
 * Uses: Signed addition, subtraction, multiplication, division, modulo,
 *       unary minus, comparison, squareRoot, power2, isNegative. */

using Aesi512 = Aesi<512>;

/* Returns gcd(a, b) and sets x, y such that a*x + b*y = gcd(a, b). */
Aesi512 extendedGcd(const Aesi512& a, const Aesi512& b, Aesi512& x, Aesi512& y) {
    if (b == 0) {
        x = 1; y = 0;
        return a;
    }
    Aesi512 x1 {}, y1 {};
    const Aesi512 g = extendedGcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - a / b * y1;
    return g;
}

void checkBezout(const Aesi512& a, const Aesi512& b) {
    /* extendedGcd requires non-negative inputs (% follows C++ sign-of-dividend semantics).
     * gcd(a,b) = gcd(|a|,|b|); negate x or y if the original input was negative. */
    const Aesi512 absA = a.isNegative() ? -a : a;
    const Aesi512 absB = b.isNegative() ? -b : b;

    Aesi512 x {}, y {};
    const Aesi512 g = extendedGcd(absA, absB, x, y);

    if (a.isNegative()) x = -x;
    if (b.isNegative()) y = -y;

    if (a * x + b * y != g) {
        std::cerr << "Bezout identity failed: a=" << a << " b=" << b << std::endl;
        std::exit(1);
    }
}

void checkSignedOps() {
    constexpr Aesi512 pos = 42;
    /* Unary minus and isNegative */
    if constexpr (constexpr Aesi512 neg = -pos; !neg.isNegative() || pos.isNegative()) {
        std::cerr << "Sign check failed" << std::endl;
        std::exit(1);
    }

    /* Increment / decrement across zero */
    Aesi512 v = -1;
    ++v;
    if (v != 0) { std::cerr << "Increment across zero failed" << std::endl; std::exit(1); }
    --v;
    if (v != -1) { std::cerr << "Decrement across zero failed" << std::endl; std::exit(1); }

    /* power2 and squareRoot roundtrip */
    for (std::size_t p = 2; p <= 20; p += 2) {
        const Aesi512 powered = Aesi512::power2(p);
        if (const Aesi512 root = powered.squareRoot(); root * root != powered) {
            std::cerr << "squareRoot roundtrip failed at power " << p << std::endl;
            std::exit(1);
        }
    }

    constexpr Aesi512 a = 1000, b = -337;
    /* Mixed-sign arithmetic */
    Aesi512 q {}, r {};
    Aesi512::divide(a, b, q, r);
    if (q * b + r != a) {
        std::cerr << "divide identity failed" << std::endl;
        std::exit(1);
    }
}

/* Multiply a slice of primes[offset], primes[offset+step], ... (count terms) into one Aesi512. */
Aesi512 primeProduct(const std::vector<unsigned int>& primes, const std::size_t offset, const std::size_t step, const std::size_t count) {
    Aesi512 result = 1;
    for (std::size_t k = 0; k < count && offset + k * step < primes.size(); ++k)
        result *= primes[offset + k * step];
    return result;
}

int main() {
    checkSignedOps();

    std::vector<unsigned int> primes {};
    primesieve::generate_primes(10'000u, &primes);

    /* Small primes: breadth check across many distinct pairs (gcd = 1). */
    for (std::size_t i = 0; i + 1 < primes.size(); i += 2) {
        checkBezout(Aesi512(primes[i]), Aesi512(primes[i + 1]));
    }

    /* Large operands: products of 12 primes each (~168 bits), spanning multiple
     * internal 32-bit blocks — exercises multiblock arithmetic and carry propagation.
     * Shared factor ensures non-trivial gcd. */
    constexpr std::size_t groupSize = 12;
    for (std::size_t i = 0; i + 3 * groupSize < primes.size(); i += groupSize) {
        const Aesi512 shared = primes[i];
        const Aesi512 a = shared * primeProduct(primes, i + 1,          1, groupSize);
        const Aesi512 b = shared * primeProduct(primes, i + groupSize + 1, 1, groupSize);
        checkBezout(a, b);
        checkBezout(-a, b);
        checkBezout(a, -b);
    }

    std::cout << "All checks passed." << std::endl;
    return 0;
}
