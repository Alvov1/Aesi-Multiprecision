#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <filesystem>
#include "../../Aeu.h"

/* Creator: Alexander Lvov
 * Details: Pollard's (P-1) algorithm for integral factoring.
 * Uses: Unsigned multiplication, division, modular exponentiation and GCD. */

using primeType = unsigned short;
std::vector<primeType> loadPrimes(const std::filesystem::path& fromLocation) {
    if(!std::filesystem::is_regular_file(fromLocation))
        throw std::invalid_argument("Prime table is not found");

    std::ifstream input(fromLocation, std::ios::binary);
    if(input.fail())
        throw std::runtime_error("Failed to open prime table");

    const std::size_t primesAmount = std::filesystem::file_size(fromLocation) / sizeof(primeType);
    std::vector<unsigned short> primes (primesAmount);
    for (auto& prime: primes)
        input.read(reinterpret_cast<char*>(&prime), sizeof(primeType));

    return primes;
}

int main() {
    const std::filesystem::path primesLocation = "../../primes.bin";
    const auto primes = loadPrimes(primesLocation);

    std::array numbers = { Aeu<512>("0x4c6f0a38f6c296d07052b794a02317ce9758855"),       // 0xa9ab4314cf -> 2 × 3^2 × 7 × 1181 × 2083 × 2351
        Aeu<512>("0x14cd01a38ac5c55992acb21ff9665294b30f9ee578393dad147") }; // 0x2fdd8d6ba69 -> 2^3 × 3 × 5 × 647 × 2203 × 19231

    int factorsFound = 0;

    for(auto& number: numbers) {
        std::cout << "N = " << std::showbase << std::hex << number << std::endl;

        Aeu<512> base = 2u;
        Aeu<4096> power = 1u;

        for(unsigned short prime : primes) {
            const auto primeF = static_cast<long double>(prime),
                boarder = static_cast<long double>(std::numeric_limits<uint64_t>::max());
            const auto primePower = static_cast<long double>(static_cast<uint64_t>(std::log(boarder) / std::log(primeF)) - 1);

            power *= static_cast<uint64_t>(std::pow(primeF, primePower));
            if(power.bitCount() > 3072) {
                base = Aeu<512>::powm<4096>(base, power, number);
                power = 1u;

                const auto candidate = Aeu<512>::gcd(base - 1u, number);
                if(candidate > 1u) {
                    std::cout << "Completed: " << std::showbase << std::hex << candidate << std::endl;
                    ++factorsFound;
                    break;
                }
            }
        }
    }

    return static_cast<int>(numbers.size()) - factorsFound;
}