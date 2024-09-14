#include <iostream>
#include <random>
#include <bitset>
#include <iomanip>
#include <sstream>
#include <map>
#include <set>
#include <cryptopp/integer.h>
#include <cryptopp/nbtheory.h>
#include <cryptopp/osrng.h>

auto& ss = std::cout;

std::random_device dev;
std::mt19937 gen(dev());
std::uniform_int_distribution<unsigned> dist(1u << 30, 1u << 31);

using CryptoPP::Integer;
using Unsigned = Integer;

Unsigned getRandomWithBits(std::size_t bitLength, bool prime = false) {
    CryptoPP::AutoSeededRandomPool prng;
    Unsigned value {};
    value.Randomize(prng, Unsigned::Zero(), Unsigned::Power2(bitLength), prime ? Unsigned::RandomNumberType::PRIME : Unsigned::RandomNumberType::ANY);
    return value;
}

void binaryRead() {
    for (unsigned i = 0; i < 30; ++i) {
        std::array<unsigned, 256 / 32> blocks {};
        for(auto& block: blocks)
            block = dist(gen);



        if(i % 2 == 0) {
            ss << "{\n\tstd::array<unsigned, 8> blocks = { ";
            for(auto block: blocks)
                ss << "0b" << std::bitset<32>(block) << ", ";
            ss << " };\n\tstd::stringstream ss; ";

            ss << "for(auto& block: blocks) ss.write(reinterpret_cast<const char*>(&block), sizeof(unsigned)); Aesi<256> a; a.readBinary(ss, true);\n\t";
            ss << "EXPECT_EQ(a, \"0b";
            for(auto block: blocks)
                std::cout << std::bitset<32>(block);
            ss << "\");\n}\n";
        } else {
            ss << "{\n\tstd::array<unsigned, 8> blocks = { ";
            for(auto it = blocks.rbegin(); it != blocks.rend(); ++it)
                ss << "0b" << std::bitset<32>(*it) << ", ";
            ss << " };\n\tstd::stringstream ss; ";

            ss << "for(auto it = blocks.rbegin(); it != blocks.rend(); ++it) ss.write(reinterpret_cast<const char*>(&*it), sizeof(unsigned)); Aesi<256> a; a.readBinary(ss, false);\n\t";
            ss << "EXPECT_EQ(a, \"0b";
            for(auto it = blocks.rbegin(); it != blocks.rend(); ++it)
                std::cout << std::bitset<32>(*it);
            ss << "\");\n}\n";
        }
    }
}

//{
//    unsigned blocks [] = { 0b01000010010011110100011111111110, 0b11010100010001111100100001010110, 0b10010101011011111001000001000101, 0b10100101101100001111101101111010, 0b10001001011001011010111010101100, 0b10111000110011110011010101010011, 0b10011101011111101100111010111111, 0b00011111011110111001010101000101, };
//    std::stringstream ss; for(auto& block: std::views::reverse(blocks)) ss.write(reinterpret_cast<const char*>(&block), sizeof(unsigned)); Aesi<256> a; a.readBinary(ss, false);
//    EXPECT_EQ(a, "0b0100001001001111010001111111111011010100010001111100100001010110100101010110111110010000010001011010010110110000111110110111101010001001011001011010111010101100101110001100111100110101010100111001110101111110110011101011111100011111011110111001010101000101");
//}

int main() {
    binaryRead();
    return 0;
}
