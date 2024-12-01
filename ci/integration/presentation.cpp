#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <regex>
#include "../../Aeu.h"

TEST(Prezentation, Factorial) {
    Aeu<512> f = 1u;
    for(unsigned i = 2; i <= 50; ++i)
        f *= i;

    std::cout << std::showbase << std::hex << f << std::endl;
}

TEST(Prezentation, PrecisionCast) {
    std::stringstream cout {};
    using std::endl;

    /* ---------------------------------------------------------------- */
    Aeu<128> base = "10888869450418352160768000001";
    Aeu<96> power = "99990001";
    Aeu<256> mod = "8683317618811886495518194401279999999";

    cout << Aeu<256>::powm(base.precisionCast<256>(), power.precisionCast<256>(), mod) << endl;

    Aeu<128> m128 = "127958277599458332250117";
    Aeu<192> m192 = "279256103987149586783914830";

    cout << m128.precisionCast<192>() * m192 << endl;
    // Cast number of 128 bits to 256 bits, than multiply by number of 160 bits
    /* ---------------------------------------------------------------- */

    std::smatch sm {}; const auto s = cout.str();
    const std::regex re ("6680141832773294447513292887050873529\n"
                         "35733130075330889632933652650476631619495985535110\n");

    std::regex_search(s, sm, re); EXPECT_TRUE(!sm.empty());
}

TEST(Prezentation, PrecisionCast2) {
    std::stringstream cout {};
    using std::endl;

    /* ---------------------------------------------------------------- */
    Aeu<128> base = "340199290171201906239764863559915798527",
        power = "340282366920937859000464800151540596704",
        modulo = "338953138925230918806032648491249958912";

    cout << Aeu<128>::powm(base, power, modulo) << endl;  // Overflow !!!

    cout << Aeu<256>::powm(base.precisionCast<256>(),         // Fine
        power,
        modulo.precisionCast<256>()) << endl;
    /* ---------------------------------------------------------------- */

    std::smatch sm {}; const auto s = cout.str();
    const std::regex re ("\\w+\n201007033690655614485250957754150944769\n");

    std::regex_search(s, sm, re); EXPECT_TRUE(!sm.empty());
}