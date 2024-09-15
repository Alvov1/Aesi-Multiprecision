#include <gtest/gtest.h>
#include <thread>

TEST(Multiplication, CryptoPP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

TEST(Multiplication, GMP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(130));
}

TEST(Multiplication, Aesi) {
    std::this_thread::sleep_for(std::chrono::milliseconds(170));
}
