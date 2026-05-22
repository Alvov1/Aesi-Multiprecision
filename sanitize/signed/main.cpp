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
    Aesi512 x1, y1;
    const Aesi512 g = extendedGcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}

void checkBezout(const Aesi512& a, const Aesi512& b) {
    Aesi512 x, y;
    const Aesi512 g = extendedGcd(a, b, x, y);

    if (a * x + b * y != g) {
        std::cerr << "Bezout identity failed: a=" << a << " b=" << b << std::endl;
        std::exit(1);
    }
}

void checkSignedOps() {
    /* Unary minus and isNegative */
    Aesi512 pos = 42, neg = -pos;
    if (!neg.isNegative() || pos.isNegative()) {
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
        const Aesi512 root = powered.squareRoot();
        if (root * root != powered) {
            std::cerr << "squareRoot roundtrip failed at power " << p << std::endl;
            std::exit(1);
        }
    }

    /* Mixed-sign arithmetic */
    Aesi512 a = 1000, b = -337;
    Aesi512 q, r;
    Aesi512::divide(a, b, q, r);
    if (q * b + r != a) {
        std::cerr << "divide identity failed" << std::endl;
        std::exit(1);
    }
}

int main() {
    checkSignedOps();

    std::vector<unsigned int> primes;
    primesieve::generate_primes(10'000u, &primes);

    /* Pairs of distinct primes: gcd = 1, Bezout coefficients are naturally signed. */
    for (std::size_t i = 0; i + 1 < primes.size(); i += 2) {
        const Aesi512 a = primes[i];
        const Aesi512 b = primes[i + 1];
        checkBezout(a, b);
    }

    /* Products of primes: non-trivial gcd. */
    for (std::size_t i = 0; i + 2 < primes.size(); i += 4) {
        const Aesi512 shared = primes[i];
        const Aesi512 a = shared * primes[i + 1];
        const Aesi512 b = shared * primes[i + 2];
        checkBezout(a, b);
    }

    /* Negative inputs: Bezout identity must still hold. */
    for (std::size_t i = 0; i + 1 < primes.size(); i += 6) {
        checkBezout(-Aesi512(primes[i]), Aesi512(primes[i + 1]));
        checkBezout(Aesi512(primes[i]), -Aesi512(primes[i + 1]));
    }

    std::cout << "All checks passed." << std::endl;
    return 0;
}
