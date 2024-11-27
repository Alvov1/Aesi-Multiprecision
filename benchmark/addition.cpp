#include <benchmark/benchmark.h>
#include <cryptopp/integer.h>
#include <gmpxx.h>
#include "../Aeu.h"

constexpr char data[] = "0x56612239db6d8ce375c48f335a4ba6f4933c871a672f6e66c7899af62393b55fb0fd38984923f6c7eb2d5f97b66a"
                        "c90bedaf1978972ec071c899f05d006caa686401d48c670c3c49553c15c3b7053eddc1878132dfce005cb4d8151fee"
                        "333b98656b4fc831c569bf7909f929ee6b6f693df50e2c049643195e2f648d593fb543";

static void addition_CryptoPP(benchmark::State& state) {
    CryptoPP::Integer left (data), right (data), result {};
    for (auto _ : state)
        benchmark::DoNotOptimize(result = left + right);
}
BENCHMARK(addition_CryptoPP);

static void addition_GMP(benchmark::State& state) {
    mpz_class left (data), right (data), result {};
    for (auto _ : state)
        benchmark::DoNotOptimize(result = left + right);
}
BENCHMARK(addition_GMP);

static void addition_Aesi(benchmark::State& state) {
    Aeu<2048> left (data), right (data), result {};
    for (auto _ : state)
        benchmark::DoNotOptimize(result = left + right);
}
BENCHMARK(addition_Aesi);