#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Display, Decimal) {
    Multiprecision m0 = "340282366920938463463374607435256044433";
    std::stringstream ss0 {}; ss0 << std::dec << m0;
    EXPECT_EQ(ss0.str(), "340282366920938463463374607435256044433");

    Multiprecision m1 = "8683317618811886495518194401279999999";
    std::stringstream ss1 {}; ss1 << std::dec << m1;
    EXPECT_EQ(ss1.str(), "8683317618811886495518194401279999999");

    Multiprecision negativeHuge = "-8683317618811886495518194401279999999";
    std::stringstream ss2 {}; ss2 << std::dec << negativeHuge;
    EXPECT_EQ(ss2.str(), "-8683317618811886495518194401279999999");
}

TEST(Display, Octal) {
    Multiprecision m0 = "340282366920938463463374607435256044433";
    std::stringstream ss0 {}; ss0 << std::oct << m0;
    EXPECT_EQ(ss0.str(), "0o4000000000000000000000000000000031771015621");

    Multiprecision m1 = "33116002042533525037";
    std::stringstream ss1 {}; ss1 << std::oct << m1;
    EXPECT_EQ(ss1.str(), "0o489133282872437279");

    Multiprecision m2 = "19175002942688032928599";
    std::stringstream ss2 {}; ss2 << std::oct << m2;
    EXPECT_EQ(ss2.str(), "0o4036752371755534607631527");
}

TEST(Display, Hexadecimal) {
    Multiprecision m0 = "340282366920938463463374607435256044433";
    std::stringstream ss0 {}; ss0 << std::uppercase << std::hex << m0;
    EXPECT_EQ(ss0.str(), "0x1000000000000000000000000CFE41B91");

    Multiprecision m1 = "33116002042533525037";
    std::stringstream ss1 {}; ss1 << std::uppercase << std::hex << m1;
    EXPECT_EQ(ss1.str(), "0x1CB93AC66CE52362D");

    Multiprecision m2 = "-19175002942688032928599";
    std::stringstream ss2 {}; ss2 << std::uppercase << std::hex << m2;
    EXPECT_EQ(ss2.str(), "-0x40F7A9F3EDAE61F3357");

    Multiprecision m3 = "8683317618811886495518194401279999999";
    std::stringstream ss3 {}; ss3 << std::nouppercase << std::hex << m3;
    EXPECT_EQ(ss2.str(), "0x688589cc0e9505e2f2fee557fffffff");
    std::stringstream ss4 {}; ss4 << std::uppercase << std::hex << m3;
    EXPECT_EQ(ss4.str(), "0x688589CC0E9505E2F2FEE557FFFFFFF");

    Multiprecision m5 = "-8683317618811886495518194401279999999";
    std::stringstream ss5 {}; ss5 << std::nouppercase << std::hex << m5;
    EXPECT_EQ(ss5.str(), "-0x688589cc0e9505e2f2fee557fffffff");
    std::stringstream ss6 {}; ss6 << std::uppercase << std::hex << m5;
    EXPECT_EQ(ss6.str(), "-0x688589CC0E9505E2F2FEE557FFFFFFF");
}
