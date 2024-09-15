#include <gtest/gtest.h>
#include <thread>

TEST(Addition, CryptoPP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

TEST(Addition, GMP) {
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
}

TEST(Addition, Aesi) {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
}
