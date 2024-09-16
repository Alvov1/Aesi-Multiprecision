#include <gtest/gtest.h>
#include <thread>
#include <cryptopp/integer.h>
#include <gmpxx.h>
#include "../../Aeu.h"

constexpr char data[] = "0x56612239db6d8ce375c48f335a4ba6f4933c871a672f6e66c7899af62393b55fb0fd38984923f6c7eb2d5f97b66a"
                        "c90bedaf1978972ec071c899f05d006caa686401d48c670c3c49553c15c3b7053eddc1878132dfce005cb4d8151fee"
                        "333b98656b4fc831c569bf7909f929ee6b6f693df50e2c049643195e2f648d593fb543";

TEST(Addition, CryptoPP) {
    for(std::size_t i = 0; i < 10000; ++i) {
        CryptoPP::Integer sum (data), addendum (data);
        for(std::size_t j = 0; j < 2048; ++j)
            sum += addendum;
        if(sum.IsZero()) std::cout << '1';
    }
}

TEST(Addition, GMP) {
    for(std::size_t i = 0; i < 10000; ++i) {
        mpz_class sum (data), addendum (data);
        for(std::size_t j = 0; j < 2048; ++j)
            sum += addendum;
        if(sum == 0) std::cout << '1';
    }
}

TEST(Addition, Aesi) {
    for(std::size_t i = 0; i < 10000; ++i) {
        Aeu<2048> sum (data), addendum (data);
        for(std::size_t j = 0; j < 2048; ++j)
            sum += addendum;
        if(sum.isZero()) std::cout << '1';
    }
}
