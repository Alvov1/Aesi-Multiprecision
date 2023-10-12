#include <iostream>
#include <filesystem>

#include "Multiprecision.h"
#include "NumberTheory.h"

std::vector<uint64_t> loadPrimeTable(const std::filesystem::path &fromLocation) {
    if (!std::filesystem::exists(fromLocation))
        throw std::invalid_argument("Prime table location is not found.");
    FILE *input = fopen(fromLocation.string().c_str(), "r");
    if (input == nullptr)
        throw std::runtime_error("Failed to open prime table file.");

    unsigned primeTableSize{};
    if (1 != fscanf(input, "%u\n", &primeTableSize))
        throw std::runtime_error("Failed to read prime table size.");

    std::vector <uint64_t> primes(primeTableSize);
    for (auto &prime: primes)
        if (1 != fscanf(input, "%llu\n", &prime))
            throw std::runtime_error("Failed to read prime number.");

    if (1 == fclose(input))
        throw std::runtime_error("Failed to close the input file.");

    return primes;
}

Multiprecision<512> findDivisor(const Multiprecision<512>& n, const std::vector<uint64_t>& primes) {
    const unsigned threadIndex = 0,
            threadsNumber = 1,
            blockIndex = 1,
            maxIterations = 40'000,
            blockStart = 2 + blockIndex,
            blockShift = threadsNumber,
            blockMax = 2000000000;

    Multiprecision a = threadIndex * maxIterations + 2, e = 1;
    for(unsigned B = blockStart; B < blockMax; B += blockShift) {
        uint64_t prime = primes[0];
        for(unsigned pI = 0; prime < B; ++pI) {
            const auto power = static_cast<unsigned>(log(static_cast<double>(B)) / log(static_cast<double>(prime)));
            e *= static_cast<unsigned>(pow(static_cast<double>(prime), power));
            prime = primes[pI + 1];
        }

        if(e == 1) continue;

        for(unsigned iteration = 0; iteration < maxIterations; ++iteration) {
            Multiprecision d = gcd(a, n);
            if(d > 1)
                return d;

            const Multiprecision b = powm(a, e, n) - 1;
            d = gcd(b, n);

            if(d > 1 && d < n)
                return d;

            a += 1;
        }
    }
    return n;
}

int main() {

}