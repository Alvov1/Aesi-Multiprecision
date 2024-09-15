#include <iostream>
#include <fstream>
#include <vector>
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

    const std::size_t primesAmount = 2000; //std::filesystem::file_size(fromLocation) / sizeof(primeType);
    std::vector<unsigned short> primes (primesAmount);
    for (auto& prime: primes)
        input.read(reinterpret_cast<char*>(&prime), sizeof(primeType));

    return primes;
}

int main() {
    const Aeu<512> n = "0x4c6f0a38f6c296d07052b794a02317ce9758855";
    std::cout << "N = " << std::showbase << std::hex << n << std::endl;

    const std::filesystem::path primesLocation = "../../primes.bin";
    const auto primes = loadPrimes(primesLocation);

    Aeu<512> base = 2u;
    Aeu<2048> power = 1u;

    for(unsigned short prime : primes) {
        const auto primeF = static_cast<long double>(prime),
            boarder = static_cast<long double>(std::numeric_limits<uint64_t>::max());
        const auto primePower = static_cast<long double>(static_cast<uint64_t>(std::log(boarder) / std::log(primeF)) - 1);

        power *= static_cast<uint64_t>(std::pow(primeF, primePower));
        if(power.bitCount() > 1536) {
            base = Aeu<512>::powm<2048>(base, power, n);
            power = 1u;

            const auto candidate = Aeu<512>::gcd(base - 1u, n);
            if(candidate > 1u) {
                std::cout << "Completed: " << std::showbase << std::hex << candidate << std::endl;
                return 0;
            }
        }
    }

    return 1;
}