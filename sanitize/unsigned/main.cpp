#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include "../../Aeu.h"

/* Creator: Alexander Lvov
 * Details: Pollard's (P-1) algorithm for integral factoring.
 * Uses: Unsigned multiplication, division, modular exponentiation and GCD. */

using namespace std;
using primeType = unsigned short;

vector<primeType> loadPrimes(const filesystem::path& fromLocation) {
    ifstream input(fromLocation, ios::binary);
    if(input.fail())
        throw std::runtime_error("Failed to open prime table");

    const size_t primesAmount = filesystem::file_size(fromLocation) / sizeof(primeType);
    vector<unsigned short> primes (primesAmount);
    for (auto& prime: primes)
        input.read(reinterpret_cast<char*>(&prime), sizeof(primeType));

    return primes;
}

Aeu512 factorize(const Aeu512& number, const vector<primeType>& primes) {
    Aeu512 base = 2u;
    Aeu<4096> power = 1u;

    for(const unsigned short prime : primes) {
        const auto primeF = static_cast<long double>(prime),
            boarder = static_cast<long double>(numeric_limits<uint64_t>::max());
        const auto primePower = static_cast<long double>(static_cast<uint64_t>(log(boarder) / log(primeF)) - 1);

        power *= static_cast<uint64_t>(pow(primeF, primePower));
        if(power.bitCount() > 3072) {
            base = Aeu512::powm<4096>(base, power, number);
            power = 1u;

            if(auto candidate = Aeu512::gcd(base - 1u, number); candidate > 1u)
                return candidate;
        }
    }

    throw std::runtime_error("Factor is not found.");
}

int main(const int argc, const char * const * const argv) {
    if(argc < 2)
        throw invalid_argument(string("Usage: ") + argv[0] + " <primes location>");

    const filesystem::path fromLocation (argv[1]);
    if(!filesystem::is_regular_file(fromLocation))
        throw invalid_argument("Prime table is not found");

    const auto primes = loadPrimes(fromLocation);

    const array numbers = { Aeu512("0x4c6f0a38f6c296d07052b794a02317ce9758855"),
        // 0xa9ab4314cf -> 2 × 3^2 × 7 × 1181 × 2083 × 2351
        Aeu512("0x14cd01a38ac5c55992acb21ff9665294b30f9ee578393dad147") };
        // 0x2fdd8d6ba69 -> 2^3 × 3 × 5 × 647 × 2203 × 19231

    for(auto& number: numbers) {
        const auto candidate = factorize(number, primes);
        cout << "Completed. N: " << showbase << hex << number << " - 0x" << candidate << std::endl;
    }

    return 0;
}