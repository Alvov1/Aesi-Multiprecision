#include <benchmark/benchmark.h>
#include <cryptopp/integer.h>
#include <cryptopp/nbtheory.h>
#include <gmpxx.h>
#include "../Aeu.h"

constexpr char base[] = "0x5bc934d7d1b1fd4cb5d62afd84e10ad94c030cee0f851155c94d374295228fd11d21119b4ad772673535a6c7f0fc"
                        "15a05458ac2dd14585cd663aeac277d5",
               power[] = "0x2180fdb65bad367705b0dd5420c1a33d4f61797a6cbfa18523ddc7657f24f5937c8aac2bc84ce4095ca0a1372fc"
                         "b5099b5223eea903514e75debcbb63558",
               modulo[] = "0x2417bbd015ab67eff98d57662f7fe82d006fb86ba1ec18d48e7563a19357ed0e7658d5aa115687cbd6c7151a30"
                          "892041ab6f8f7be9e3b828ff39a4688402";

static void powm_CryptoPP(benchmark::State& state) {
    CryptoPP::Integer baseA (base), powerA (power), moduloA (modulo), result {};
    for (auto _ : state)
        result = CryptoPP::ModularExponentiation(baseA, powerA, moduloA);
}
BENCHMARK(powm_CryptoPP);

static void powm_GMP(benchmark::State& state) {
    mpz_class baseA (base), powerA (power), moduloA (modulo), result {};
    for (auto _ : state)
        mpz_powm(result.get_mpz_t(), baseA.get_mpz_t(), powerA.get_mpz_t(), moduloA.get_mpz_t());
}
BENCHMARK(powm_GMP);

static void powm_Aesi(benchmark::State& state) {
    Aeu<512> baseA (base), powerA (power), moduloA (modulo), result {};
    for (auto _ : state)
        result = Aeu<512>::powm(baseA, powerA, moduloA);
}
BENCHMARK(powm_Aesi);