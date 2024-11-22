#include <benchmark/benchmark.h>
#include <cryptopp/integer.h>
#include <gmpxx.h>
#include "../Aeu.h"

constexpr char division[] = "0x1099091f922d948121cf94880af1fd07a60010c9bbf89884aac215f37c6418b2735a3e50e0889fac0c3ea61d"
                            "bc829d3919e94bf714f521969e75e15f570f870ef5e086add27842cfc8cafd321d038354a97e152c0ea74df004"
                            "0a2210f92c0b71aaded40c0a1bef125c2d187a2e8ea2cbfcc4664da71734ea5683da6de60ec5a2be9374608a64"
                            "9ff89756a4f65fd78af3c3744d886a87bfc95a7ea6fdd267b64ca69f4e87d1c7f83f77aee7dc328713778f330e"
                            "b8b330edf206ecfa95b6d642e539ea7a7b1079fff2ed549eb4d4a9b46cab1bacaa87f62bca7ebfefea7b1545af"
                            "3d05af49790b95e8379cd326db1435520c0e817b81567803fb5af42c02b793ce88e684644c52dcca2a69049a51"
                            "d3201e85dd75326b89f216cd309cf66f9c3ed697dfae1983616a8a6cbd33bf29548e5c95bf146bb403c6ca62f5"
                            "44fc0337a7491edcf0fa12dcf685eb0baa11176baee2391838f3fbf8f81d8ab32151f4cd0205340a6736d22db7"
                            "2a8906df17f8ac48b9e39af090733f70f0d8232ab3ff8e10d72e6129cab2507bb7db61e2e0f4f63bf58d315794"
                            "ae840d1bf7d395e3509f36485fb6caeaf775b478391d8b89f1323b1a921cb77ac656f22ec33354252dc017ad31"
                            "e6df7204cbdb7c73a35857d5dd520d4c6db2d1ac33a8f54ccd362837681484de652e54eda7516e72767e6e9ac8"
                            "debb68497b07dbb45c1bdb97ed6b0dbbe5031099091f922d948121cf94880af1fd07a60010c9bbf89884aac215"
                            "f37c6418b2735a3e50e0889fac0c3ea61dbc829d3919e94bf714f521969e75e15f570f870ef5e086add27842cf"
                            "c8cafd321d038354a97e152c0ea74df0040a2210f92c0b71aaded40c0a1bef125c2d187a2e8ea2cbfcc4664da7"
                            "1734ea5683da6de60ec5a2be9374608a649ff89756a4f65fd78af3c3744d886a87bfc95a7ea6fdd267b64ca69f"
                            "4e87d1c7f83f77aee7dc328713778f330eb8b330edf206ecfa95b6d642e539ea7a7b1079fff2ed549eb4d4a9b4"
                            "6cab1bacaa87f62bca7ebfefea7b1545af3d05af49790b95e8379cd326db1435520c0e817b81567803fb5af42c"
                            "02b793ce88e684644c52dcca2a69049a51d3201e85dd75326b89f216cd309cf66f9c3ed697dfae1983616a8a6c"
                            "bd33bf29548e5c95bf146bb403c6ca62f544fc0337a7491edcf0fa12dcf685eb0baa11176baee2391838f3fbf8"
                            "f81d8ab32151f4cd0205340a6736d22db72a8906df17f8ac48b9e39af090733f70f0d8232ab3ff8e10d72e6129"
                            "cab2507bb7db61e2e0f4f63bf58d315794ae840d1bf7d395e3509f36485fb6caeaf775b478391d8b89f1323b1a"
                            "921cb77ac656f22ec33354252dc017ad31e6df7204cbdb7c73a35857d5dd520d4c6db2d1ac33a8f54ccd362837"
                            "681484de652e54eda7516e72767e6e9ac8debb68497b07dbb45c1bdb97ed6b0dbbe503";

constexpr char divisor[] = "0x55c5374ad14e5c9bff62109df3100124f654bb11ef8fbdcc93e892fde002a462";

static void division_CryptoPP(benchmark::State& state) {
    CryptoPP::Integer left (division), right (divisor), result {};
    for(auto _ : state)
        benchmark::DoNotOptimize(result = left / right);
    // if(result.IsEven()) result += 1;
}
BENCHMARK(division_CryptoPP);

static void division_GMP(benchmark::State& state) {
    mpz_class left (division), right (divisor), result {};
    for(auto _ : state)
        benchmark::DoNotOptimize(result = left / right);
    // if(mpz_even_p(result.get_mpz_t())) result += 1;
}
BENCHMARK(division_GMP);

static void division_Aesi(benchmark::State& state) {
    Aeu<8192> left = division, right = divisor, result {};
    for(auto _ : state)
        benchmark::DoNotOptimize(result = left / right);
    // if(result.isEven()) result += 1u;
}
BENCHMARK(division_Aesi);