#include <gtest/gtest.h>
#include <thread>
#include <cryptopp/integer.h>
#include <gmpxx.h>
#include "../../Aesi.h"

constexpr char left[] = "0xa24872afd57464d79dfeec239367995f623429772ec032eb3bc9b376d0775c956dcb330a2c4f8f0d001f62fe6e0b"
                        "2e3b024d8f352522f3aa0ac7fbf05e7e3a2038e4650efb641862969f2437084448a28ff14d9c1e2def670babcf65a8"
                        "bcf32944a0f13525588a09a88a93e209ea0951e2ec64b7ec6904012f1c2f0528f9c68597e4cb39b33bae0dcc84a8c7"
                        "aa93733c0a5bb987b0c231e099aeb793c5fcbbdd9963aaff6ccbd2198b8898f71d60f8dcd8d0719f9b849951960c16"
                        "99b751dda0da6b9cc5a0ea9a04d126a00ce2c3bdaf8ea5e9123d34a00fbd21c31b12fa50d0a504f394bdce92bc1bc8"
                        "1af39a297c232e16af614ae8743a5333e2e1f016e08c09d7520f92da4a54de6819bec543645d4f3bc42c7948164c90"
                        "86cfd3b528ce5839125cfe4a991c66d69b6cf4cef13fb96095f774189943b189373bb931c5e7616f9b416de77d28c1"
                        "6a14a980f664352dc3b03026875f0db91a645a7acff73e79a815329df6c3dfac4c1ba96a1a9ce50b51cc1722f23586";

constexpr char right[] = "0xdb446efadae5960843e38dbdc26afc0c6d6633d0e3f7983b11d7610838b2707f3bcfeb19b071a967738efb939cf"
                         "e36278016a3127f0bbe2cc3ea8bb25d827ce4738db0275c95705462c52b13544a446fa14b4727aa6a472d766acb36"
                         "d6eb1cfc4508ba194b65e816a7343eff12d50669040a7629f04300a5bc4bb98df0191884b6d8236341854a1dcf0e3"
                         "d0f54615b1d9a4eb83fcd1ceecc0b906ad6cae9137baf43136d4e00a2638e7897e3b60b4330df9917e2f62742233c"
                         "24d325d42f14a9ba79f454a6cb5b3617e199335acaa4fd901721287498db25aac318abb51f2c4e9bc2e556168fd9c"
                         "7791285c4e30119bca7a4bd63db4f47fad992880389d21e03b1386a492326981cf8524b538cbfe4397df8bcfebc10"
                         "8bab8b2500ed99ac8b048a5307fc5ac4a03c5672fefde43454aa323be66e77e797b23d5865f7454be74195e7857ed"
                         "f825d5bb62a1a327148f4e8405f2ee2622122d3d8c967f2fdc6e05a8c1d40c93f3c0dfeb5b8b4c9f1755a9c26f351"
                         "ac1cdebf";

TEST(Multiplication, CryptoPP) {
    for(std::size_t i = 0; i < 20; ++i) {
        Aesi<4192> leftA (left), rightA (right), _ {};
        for(std::size_t j = 0; j < 16384 * 16384; j += 16384)
            _ = leftA * (rightA + j);
        if(_.isZero()) std::cout << '1';
    }
}

TEST(Multiplication, GMP) {
    for(std::size_t i = 0; i < 15; ++i) {
        mpz_class leftA (left), rightA (right), _ {};
        for(std::size_t j = 0; j < 16384 * 16384; j += 16384)
            _ = leftA * (rightA + j);
        if(_ == 0) std::cout << '1';
    }
}

TEST(Multiplication, Aesi) {
    for(std::size_t i = 0; i < 15; ++i) {
        CryptoPP::Integer leftA (left), rightA (right), _ {};
        for(long long j = 0; j < 16384 * 16384; j += 16384)
            _ = leftA * (rightA + j);
        if(_.IsZero()) std::cout << '1';
    }
}
