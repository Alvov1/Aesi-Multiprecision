#include <gtest/gtest.h>
#include <thread>

TEST(Division, CryptoPP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(125));
}

TEST(Division, GMP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(135));
}

TEST(Division, Aesi) {
    std::this_thread::sleep_for(std::chrono::milliseconds(145));
}
