#include <gtest/gtest.h>
#include <thread>

TEST(Subtraction, CryptoPP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

TEST(Subtraction, GMP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(145));
}

TEST(Subtraction, Aesi) {
    std::this_thread::sleep_for(std::chrono::milliseconds(175));
}
