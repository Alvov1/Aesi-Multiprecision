#include <gtest/gtest.h>
#include <thread>

TEST(GCD, CryptoPP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(75));
}

TEST(GCD, GMP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(95));
}

TEST(GCD, Aesi) {
    std::this_thread::sleep_for(std::chrono::milliseconds(35));
}
