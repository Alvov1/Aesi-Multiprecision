#include <gtest/gtest.h>
#include "../../Multiprecision.h"

TEST(Addition, Zero) {
    Multiprecision zero = 0;
    Multiprecision m0 = -26359343;
    EXPECT_EQ(m0 + 0, -26359343);
    EXPECT_EQ(0 + m0, -26359343);
    EXPECT_EQ(m0 - 0, -26359343);
    EXPECT_EQ(m0 + zero, -26359343);
    EXPECT_EQ(m0 - zero, -26359343);

    EXPECT_EQ(m0 + -zero, -26359343);
    EXPECT_EQ(m0 + +zero, -26359343);
    EXPECT_EQ(m0 - +zero, -26359343);
    EXPECT_EQ(m0 - -zero, -26359343);

    EXPECT_EQ(zero + m0, -26359343);
    EXPECT_EQ(+zero + m0, -26359343);
    EXPECT_EQ(-zero + m0, -26359343);

    m0 += 0; EXPECT_EQ(m0, -26359343);
    m0 += zero; EXPECT_EQ(m0, -26359343);
    m0 += -zero; EXPECT_EQ(m0, -26359343);
    m0 += +zero; EXPECT_EQ(m0, -26359343);

    Multiprecision m1 = 14670384;
    EXPECT_EQ(m1 + 0, 14670384);
    EXPECT_EQ(0 + m1, 14670384);
    EXPECT_EQ(m1 - 0, 14670384);
    EXPECT_EQ(m1 + zero, 14670384);
    EXPECT_EQ(m1 - zero, 14670384);
    EXPECT_EQ(m1 + -zero, 14670384);
    EXPECT_EQ(m1 + +zero, 14670384);
    EXPECT_EQ(m1 - +zero, 14670384);
    EXPECT_EQ(m1 + +zero, 14670384);
    EXPECT_EQ(zero + m1, 14670384);
    EXPECT_EQ(+zero + m1, 14670384);
    EXPECT_EQ(-zero + m1, 14670384);
    m1 += 0; EXPECT_EQ(m1, 14670384);
    m1 += zero; EXPECT_EQ(m1, 14670384);
    m1 += -zero; EXPECT_EQ(m1, 14670384);
    m1 += +zero; EXPECT_EQ(m1, 14670384);

    Multiprecision m2 = 55908622;
    EXPECT_EQ(m2 + 0, 55908622);
    EXPECT_EQ(0 + m2, 55908622);
    EXPECT_EQ(m2 - 0, 55908622);
    EXPECT_EQ(m2 + zero, 55908622);
    EXPECT_EQ(m2 - zero, 55908622);
    EXPECT_EQ(m2 + -zero, 55908622);
    EXPECT_EQ(m2 + +zero, 55908622);
    EXPECT_EQ(m2 - +zero, 55908622);
    EXPECT_EQ(m2 + +zero, 55908622);
    EXPECT_EQ(zero + m2, 55908622);
    EXPECT_EQ(+zero + m2, 55908622);
    EXPECT_EQ(-zero + m2, 55908622);
    m2 += 0; EXPECT_EQ(m2, 55908622);
    m2 += zero; EXPECT_EQ(m2, 55908622);
    m2 += -zero; EXPECT_EQ(m2, 55908622);
    m2 += +zero; EXPECT_EQ(m2, 55908622);
}

TEST(Addition, SmallPositive) {
    Multiprecision s1 = 0x24DFBE889, s2 = 0x193E161C, s3 = 0x51CDFC6, s4 = 0x1706808355;
    EXPECT_EQ(s1 + s1, 0x49BF7D112);
    EXPECT_EQ(s1 + s2, 0x26739fea5);
    EXPECT_EQ(s1 + s3, 0x25318c84f);
    EXPECT_EQ(s1 + s4, 0x19547c6bde);

    EXPECT_EQ(s2 + s1, 0x26739fea5);
    EXPECT_EQ(s2 + s2, 0x327c2c38);
    EXPECT_EQ(s2 + s3, 0x1e5af5e2);
    EXPECT_EQ(s2 + s4, 0x171fbe9971);

    EXPECT_EQ(s3 + s1, 0x25318c84f);
    EXPECT_EQ(s3 + s2, 0x1e5af5e2);
    EXPECT_EQ(s3 + s3, 0xa39bf8c);
    EXPECT_EQ(s3 + s4, 0x170b9d631b);

    EXPECT_EQ(s4 + s1, 0x19547c6bde);
    EXPECT_EQ(s4 + s2, 0x171fbe9971);
    EXPECT_EQ(s4 + s3, 0x170b9d631b);
    EXPECT_EQ(s4 + s4, 0x2e0d0106aa);
}

TEST(Addition, SmallNegative) {
    Multiprecision s1 = -0x24DFBE889, s2 = -0x193E161C, s3 = -0x51CDFC6, s4 = -0x1706808355;
    EXPECT_EQ(s1 + s1, -0x49BF7D112);
    EXPECT_EQ(s1 + s2, -0x26739fea5);
    EXPECT_EQ(s1 + s3, -0x25318c84f);
    EXPECT_EQ(s1 + s4, -0x19547c6bde);

    EXPECT_EQ(s2 + s1, -0x26739fea5);
    EXPECT_EQ(s2 + s2, -0x327c2c38);
    EXPECT_EQ(s2 + s3, -0x1e5af5e2);
    EXPECT_EQ(s2 + s4, -0x171fbe9971);

    EXPECT_EQ(s3 + s1, -0x25318c84f);
    EXPECT_EQ(s3 + s2, -0x1e5af5e2);
    EXPECT_EQ(s3 + s3, -0xa39bf8c);
    EXPECT_EQ(s3 + s4, -0x170b9d631b);

    EXPECT_EQ(s4 + s1, -0x19547c6bde);
    EXPECT_EQ(s4 + s2, -0x171fbe9971);
    EXPECT_EQ(s4 + s3, -0x170b9d631b);
    EXPECT_EQ(s4 + s4, -0x2e0d0106aa);
}

TEST(Addition, Increment) {
    Multiprecision m0 = 62492992;
    ++m0; ++m0; m0++; ++m0; m0++; ++m0; m0++; ++m0; m0++; ++m0;
    EXPECT_EQ(m0, 62493002);
    Multiprecision t0 = m0++, u0 = ++m0;
    EXPECT_EQ(t0, 62493002); EXPECT_EQ(u0, 62493004); EXPECT_EQ(m0, 62493004);

    Multiprecision m1 = -10775863;
    m1++; ++m1; m1++; ++m1; m1++; ++m1; m1++; ++m1; ++m1; ++m1; m1++; ++m1; ++m1;
    EXPECT_EQ(m1, -10775850);
    Multiprecision t1 = m1++, u1 = ++m1;
    EXPECT_EQ(t1, -10775850); EXPECT_EQ(u1, -10775848); EXPECT_EQ(m1, -10775848);

    Multiprecision m2 = 77428594;
    m2++; m2++; ++m2; m2++; m2++; m2++; ++m2; m2++; m2++; m2++; ++m2; m2++; ++m2; ++m2;
    EXPECT_EQ(m2, 77428608);
    Multiprecision t2 = m2++, u2 = ++m2;
    EXPECT_EQ(t2, 77428608); EXPECT_EQ(u2, 77428610); EXPECT_EQ(m2, 77428610);

    Multiprecision m3 = 77677795;
    ++m3; ++m3; ++m3; m3++; ++m3; m3++; ++m3; ++m3; m3++; ++m3; m3++; ++m3; m3++; m3++; m3++; m3++; m3++; ++m3;
    EXPECT_EQ(m3, 77677813);
    Multiprecision t3 = m3++, u3 = ++m3;
    EXPECT_EQ(t3, 77677813); EXPECT_EQ(u3, 77677815); EXPECT_EQ(m3, 77677815);

    Multiprecision m4 = -11780979;
    m4++; ++m4; m4++; ++m4; m4++; ++m4; m4++; ++m4; m4++; ++m4; ++m4; ++m4; m4++; ++m4; m4++; ++m4; ++m4; ++m4; ++m4;
    EXPECT_EQ(m4, -11780960);
    Multiprecision t4 = m4++, u4 = ++m4;
    EXPECT_EQ(t4, -11780960); EXPECT_EQ(u4, -11780958); EXPECT_EQ(m4, -11780958);
}

TEST(Addition, MixedAddition) {
    Multiprecision small1 = -8492, small2 = 4243, small3 = -678;
    EXPECT_EQ(small2 + small3, 0xDED);
    EXPECT_EQ(small3 + small2, 0xDED);
    EXPECT_EQ(small2 + small1, -0x1099);
    EXPECT_EQ(small1 + small2, -0x1099);

    Multiprecision m0 = -899842982222, m1 = 54545454545, m2 = -4243242222222;
    EXPECT_EQ(m0 + m0, -0x1A305A4829C);
    EXPECT_EQ(m0 + m1, -0xC4CFA8AB7D);
    EXPECT_EQ(m0 + m2, -0x4AD77C443DC);

    EXPECT_EQ(m1 + m0, -0xC4CFA8AB7D);
    EXPECT_EQ(m1 + m1, 0x1966532BA2);
    EXPECT_EQ(m1 + m2, -0x3CF41C86CBD);

    EXPECT_EQ(m2 + m0, -0x4AD77C443DC);
    EXPECT_EQ(m2 + m1, -0x3CF41C86CBD);
    EXPECT_EQ(m2 + m2, -0x7B7E9E4051C);

    Multiprecision m5 = 44623815129875066, m6 = -67333380797917951;
    EXPECT_EQ(m5 + m6, -22709565668042885);
    Multiprecision m7 = -48014444995445931, m8 = -22286330479143061;
    EXPECT_EQ(m7 + m8, -70300775474588992);
    Multiprecision m9 = -46710497537103336, m10 = 38078385918803582;
    EXPECT_EQ(m9 + m10, -8632111618299754);
    Multiprecision m11 = -85692280903283784, m12 = 67173988903321278;
    EXPECT_EQ(m11 + m12, -18518291999962506);
    Multiprecision m13 = -44612227547418401, m14 = 59491349319431825;
    EXPECT_EQ(m13 + m14, 14879121772013424);
    Multiprecision m15 = 31078689473344356, m16 = -68742297725853412;
    EXPECT_EQ(m15 + m16, -37663608252509056);
    Multiprecision m17 = -45387158811706062, m18 = 79147083749576689;
    EXPECT_EQ(m17 + m18, 33759924937870627);
    Multiprecision m19 = -89958219458984593, m20 = -81023467713372341;
    EXPECT_EQ(m19 + m20, -170981687172356934);
    Multiprecision m21 = -26784201699120505, m22 = -80123928943714897;
    EXPECT_EQ(m21 + m22, -106908130642835402);
    Multiprecision m23 = -45649812857009124, m24 = 83244178464937860;
    EXPECT_EQ(m23 + m24, 37594365607928736);
    Multiprecision m25 = 28170049468723627, m26 = 32541822036943371;
    EXPECT_EQ(m25 + m26, 60711871505666998);
    Multiprecision m27 = -22623135028103130, m28 = 21786244319546654;
    EXPECT_EQ(m27 + m28, -836890708556476);
    Multiprecision m29 = -86015218297363345, m30 = 42612600999881113;
    EXPECT_EQ(m29 + m30, -43402617297482232);
    Multiprecision m31 = -24053032061086904, m32 = 72114036549128840;
    EXPECT_EQ(m31 + m32, 48061004488041936);
    Multiprecision m33 = 36822510221127512, m34 = 94178979038637199;
    EXPECT_EQ(m33 + m34, 131001489259764711);
    Multiprecision m35 = 97484155294477787, m36 = -75334454513432056;
    EXPECT_EQ(m35 + m36, 22149700781045731);
    Multiprecision m37 = -95934355055619379, m38 = 39209954770039503;
    EXPECT_EQ(m37 + m38, -56724400285579876);
    Multiprecision m39 = 49370459189897803, m40 = 75902539607732364;
    EXPECT_EQ(m39 + m40, 125272998797630167);
    Multiprecision m41 = -56291529889007273, m42 = -39790172022000660;
    EXPECT_EQ(m41 + m42, -96081701911007933);
    Multiprecision m43 = -25166174113513561, m44 = -49119563632507651;
    EXPECT_EQ(m43 + m44, -74285737746021212);
}

TEST(Addition, MixedAdditionAssignment) {
    Multiprecision m5 = -75372316459883297, m6 = -75066966863039062;
    m5 += m6; EXPECT_EQ(m5, -150439283322922359);
    Multiprecision m7 = -36248118596253972, m8 = 69183337187190772;
    m7 += m8; EXPECT_EQ(m7, 32935218590936800);
    Multiprecision m9 = 66104089310021402, m10 = -20950498422421752;
    m9 += m10; EXPECT_EQ(m9, 45153590887599650);
    Multiprecision m11 = -43281760133359907, m12 = 60779712756644022;
    m11 += m12; EXPECT_EQ(m11, 17497952623284115);
    Multiprecision m13 = -33624789583363843, m14 = 69106216072981817;
    m13 += m14; EXPECT_EQ(m13, 35481426489617974);
    Multiprecision m15 = 25897371791224934, m16 = -70058626665729639;
    m15 += m16; EXPECT_EQ(m15, -44161254874504705);
    Multiprecision m17 = 59354316783597997, m18 = -94991973626467623;
    m17 += m18; EXPECT_EQ(m17, -35637656842869626);
    Multiprecision m19 = -31274863735638183, m20 = -67383584399340340;
    m19 += m20; EXPECT_EQ(m19, -98658448134978523);
    Multiprecision m21 = 73598557579492490, m22 = -22951646015761999;
    m21 += m22; EXPECT_EQ(m21, 50646911563730491);
    Multiprecision m23 = -87022493315496912, m24 = -79800965327847517;
    m23 += m24; EXPECT_EQ(m23, -166823458643344429);
    Multiprecision m25 = 42148846916246179, m26 = -38317537486308969;
    m25 += m26; EXPECT_EQ(m25, 3831309429937210);
    Multiprecision m27 = -58591164247623666, m28 = 31866245567030322;
    m27 += m28; EXPECT_EQ(m27, -26724918680593344);
    Multiprecision m29 = 94024469360178215, m30 = 33354213359861956;
    m29 += m30; EXPECT_EQ(m29, 127378682720040171);
    Multiprecision m31 = -75956148969625915, m32 = 66500479174544316;
    m31 += m32; EXPECT_EQ(m31, -9455669795081599);
    Multiprecision m33 = 67953678177925403, m34 = -51458681742174591;
    m33 += m34; EXPECT_EQ(m33, 16494996435750812);
    Multiprecision m35 = -82561950395978471, m36 = 20736472398654635;
    m35 += m36; EXPECT_EQ(m35, -61825477997323836);
    Multiprecision m37 = -40406458757268920, m38 = -26529568190778018;
    m37 += m38; EXPECT_EQ(m37, -66936026948046938);
    Multiprecision m39 = -72742681232649893, m40 = 30117909419217961;
    m39 += m40; EXPECT_EQ(m39, -42624771813431932);
    Multiprecision m41 = 89568459482648576, m42 = -89327726642310274;
    m41 += m42; EXPECT_EQ(m41, 240732840338302);
    Multiprecision m43 = 26333862958113887, m44 = 82645320527096753;
    m43 += m44; EXPECT_EQ(m43, 108979183485210640);
}

TEST(Addition, DifferentPrecision) {
    {
        Multiprecision<480> first = -23929412326908601; // Multiprecision<15 blocks>
        Multiprecision<288> second = 23663609923469824; // Multiprecision<9 blocks>
        EXPECT_EQ(first + second, -265802403438777);

        Multiprecision<256> third = 95362148199314213; // Multiprecision<8 blocks>
        Multiprecision<480> forth = 10676274833885487; // Multiprecision<15 blocks>
        forth += third; EXPECT_EQ(forth, 106038423033199700);
    }
    {
        Multiprecision<416> first = 56401005283971309; // Multiprecision<13 blocks>
        Multiprecision<224> second = 98854388331887188; // Multiprecision<7 blocks>
        EXPECT_EQ(first + second, 155255393615858497);

        Multiprecision<256> third = -64101286967077361; // Multiprecision<8 blocks>
        Multiprecision<352> forth = 82812991105896938; // Multiprecision<11 blocks>
        forth += third; EXPECT_EQ(forth, 18711704138819577);
    }
    {
        Multiprecision<480> first = 19289473785919116; // Multiprecision<15 blocks>
        Multiprecision<352> second = 88606354971145352; // Multiprecision<11 blocks>
        EXPECT_EQ(first + second, 107895828757064468);

        Multiprecision<320> third = -33576333235155486; // Multiprecision<10 blocks>
        Multiprecision<448> forth = 9691621099679918; // Multiprecision<14 blocks>
        forth += third; EXPECT_EQ(forth, -23884712135475568);
    }
    {
        Multiprecision<320> first = -70855796405707722; // Multiprecision<10 blocks>
        Multiprecision<320> second = -63148117042169488; // Multiprecision<10 blocks>
        EXPECT_EQ(first + second, -134003913447877210);

        Multiprecision<224> third = 59468727606522938; // Multiprecision<7 blocks>
        Multiprecision<352> forth = -22251957457924020; // Multiprecision<11 blocks>
        forth += third; EXPECT_EQ(forth, 37216770148598918);
    }
    {
        Multiprecision<224> first = -14400024956935781; // Multiprecision<7 blocks>
        Multiprecision<224> second = 47261225548047463; // Multiprecision<7 blocks>
        EXPECT_EQ(first + second, 32861200591111682);

        Multiprecision<320> third = -46320698585836942; // Multiprecision<10 blocks>
        Multiprecision<352> forth = 74308513987943052; // Multiprecision<11 blocks>
        forth += third; EXPECT_EQ(forth, 27987815402106110);
    }
    {
        Multiprecision<288> first = -36746047384322768; // Multiprecision<9 blocks>
        Multiprecision<448> second = 8912025064804399; // Multiprecision<14 blocks>
        EXPECT_EQ(first + second, -27834022319518369);

        Multiprecision<288> third = 40875922588278201; // Multiprecision<9 blocks>
        Multiprecision<448> forth = -12295777138935876; // Multiprecision<14 blocks>
        forth += third; EXPECT_EQ(forth, 28580145449342325);
    }
    {
        Multiprecision<352> first = 95436749490787210; // Multiprecision<11 blocks>
        Multiprecision<192> second = 92713709861975215; // Multiprecision<6 blocks>
        EXPECT_EQ(first + second, 188150459352762425);

        Multiprecision<256> third = 20843439894600469; // Multiprecision<8 blocks>
        Multiprecision<352> forth = 86003846354786140; // Multiprecision<11 blocks>
        forth += third; EXPECT_EQ(forth, 106847286249386609);
    }
    {
        Multiprecision<448> first = -17143274618699834; // Multiprecision<14 blocks>
        Multiprecision<192> second = 50875672361696629; // Multiprecision<6 blocks>
        EXPECT_EQ(first + second, 33732397742996795);

        Multiprecision<256> third = 85083128565169538; // Multiprecision<8 blocks>
        Multiprecision<384> forth = -85598363854112295; // Multiprecision<12 blocks>
        forth += third; EXPECT_EQ(forth, -515235288942757);
    }
    {
        Multiprecision<320> first = 98923931870480728; // Multiprecision<10 blocks>
        Multiprecision<448> second = -38271056516262020; // Multiprecision<14 blocks>
        EXPECT_EQ(first + second, 60652875354218708);

        Multiprecision<224> third = 34102493724613230; // Multiprecision<7 blocks>
        Multiprecision<480> forth = -29854977705540280; // Multiprecision<15 blocks>
        forth += third; EXPECT_EQ(forth, 4247516019072950);
    }
    {
        Multiprecision<480> first = -32420396791982926; // Multiprecision<15 blocks>
        Multiprecision<256> second = 77560072073673480; // Multiprecision<8 blocks>
        EXPECT_EQ(first + second, 45139675281690554);

        Multiprecision<320> third = -17681540872331455; // Multiprecision<10 blocks>
        Multiprecision<384> forth = 65208663927888378; // Multiprecision<12 blocks>
        forth += third; EXPECT_EQ(forth, 47527123055556923);
    }
}

TEST(Addition, Huge) { EXPECT_TRUE(false); }