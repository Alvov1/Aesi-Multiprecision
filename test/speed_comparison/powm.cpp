#include <gtest/gtest.h>
#include <thread>
#include <cryptopp/integer.h>
#include <cryptopp/nbtheory.h>
#include <gmpxx.h>
#include "../../Aesi.h"

constexpr char base[] = "0x5bc934d7d1b1fd4cb5d62afd84e10ad94c030cee0f851155c94d374295228fd11d21119b4ad772673535a6c7f0fc"
                        "15a05458ac2dd14585cd663aeac277d5",
               power[] = "0x2180fdb65bad367705b0dd5420c1a33d4f61797a6cbfa18523ddc7657f24f5937c8aac2bc84ce4095ca0a1372fc"
                         "b5099b5223eea903514e75debcbb63558",
               modulo[] = "0x2417bbd015ab67eff98d57662f7fe82d006fb86ba1ec18d48e7563a19357ed0e7658d5aa115687cbd6c7151a30"
                          "892041ab6f8f7be9e3b828ff39a4688402";

TEST(Powm, CryptoPP) {
    Aesi<512> baseA (base), powerA (power), moduloA (modulo), _ {};
    for (std::size_t i = 0; i < 256 * 5; i += 3)
        _ = Aesi<512>::powm(baseA + 8192 * i, powerA, moduloA + 16384 * i);
    if(_.isZero()) std::cout << '1';
}

TEST(Powm, GMP) {
    mpz_class baseA (base), powerA (power), moduloA (modulo), _ {};
    for (std::size_t i = 0; i < 256 * 5; i += 3) {
        baseA += 8192 * i; moduloA += 16384 * i;
        mpz_powm(_.get_mpz_t(), baseA.get_mpz_t(), powerA.get_mpz_t(), moduloA.get_mpz_t());
    }
    if(_ == 0) std::cout << '1';
}

TEST(Powm, Aesi) {
    CryptoPP::Integer baseA (base), powerA (power), moduloA (modulo), _ {};
    for (long long i = 0; i < 256 * 5; i += 3)
        _ = CryptoPP::ModularExponentiation(baseA + 8192 * i, powerA, moduloA + 16384 * i);
    if(_.IsZero()) std::cout << '1';
}
