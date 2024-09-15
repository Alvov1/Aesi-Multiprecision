#include <gtest/gtest.h>
#include <thread>

TEST(Powm, CryptoPP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

TEST(Powm, GMP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

TEST(Powm, Aesi) {
    std::this_thread::sleep_for(std::chrono::milliseconds(375));
}
