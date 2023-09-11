#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Modulo, MixedModulo) {
    Multiprecision m0 = -4825285950739, m1 = -26462400;
    EXPECT_EQ(m0 % m1, -26085139);
    Multiprecision m2 = -15187820465943, m3 = -73508543;
    EXPECT_EQ(m2 % m3, -73379627);
    Multiprecision m4 = 29744669116783, m5 = -59738688;
    EXPECT_EQ(m4 % m5, 59497327);
    Multiprecision m6 = 1445481773735, m7 = -10943575;
    EXPECT_EQ(m6 % m7, 10613435);
    Multiprecision m8 = -9608444938718, m9 = -14443665;
    EXPECT_EQ(m8 % m9, -13452443);
    Multiprecision m10 = -1827101039843, m11 = -5476397;
    EXPECT_EQ(m10 % m11, -5232336);
    Multiprecision m12 = -8555190917773, m13 = 26131496;
    EXPECT_EQ(m12 % m13, -442333);
    Multiprecision m14 = 3835023546766, m15 = -66353339;
    EXPECT_EQ(m14 % m15, 65965922);
    Multiprecision m16 = 4152238729728, m17 = 70516764;
    EXPECT_EQ(m16 % m17, 115116);
    Multiprecision m18 = 21152437469773, m19 = 27580050;
    EXPECT_EQ(m18 % m19, 862423);
    Multiprecision m20 = -16788008144183, m21 = 74026418;
    EXPECT_EQ(m20 % m21, -964471);
    Multiprecision m22 = -28451749301697, m23 = 53230588;
    EXPECT_EQ(m22 % m23, -15697);
    Multiprecision m24 = -3270181161316, m25 = -8281687;
    EXPECT_EQ(m24 % m25, -7979000);
    Multiprecision m26 = 49633340100976, m27 = 66209125;
    EXPECT_EQ(m26 % m27, 590351);
    Multiprecision m28 = -46340911398101, m29 = -51209841;
    EXPECT_EQ(m28 % m29, -50870540);
    Multiprecision m30 = 185622644633, m31 = 9521052;
    EXPECT_EQ(m30 % m31, 214841);
    Multiprecision m32 = -10811475069954, m33 = 59784751;
    EXPECT_EQ(m32 % m33, -699114);
    Multiprecision m34 = 12380907605205, m35 = 15918898;
    EXPECT_EQ(m34 % m35, 604603);
    Multiprecision m36 = 1257249026383, m37 = -61467173;
    EXPECT_EQ(m36 % m37, 60937014);
    Multiprecision m38 = 61847752758018, m39 = -72413117;
    EXPECT_EQ(m38 % m39, 71593903);
    Multiprecision m40 = -37469111279132, m41 = 74062214;
    EXPECT_EQ(m40 % m41, -345536);
    Multiprecision m42 = -31033946342900, m43 = 31210566;
    EXPECT_EQ(m42 % m43, -935894);
    Multiprecision m44 = 262286204362, m45 = 1046069;
    EXPECT_EQ(m44 % m45, 93647);
    Multiprecision m46 = -38784099239147, m47 = -66593692;
    EXPECT_EQ(m46 % m47, -66205731);
    Multiprecision m48 = -49611834841836, m49 = 50734228;
    EXPECT_EQ(m48 % m49, -167880);
    Multiprecision m50 = -5264331698408, m51 = -13698889;
    EXPECT_EQ(m50 % m51, -13042376);
    Multiprecision m52 = 8207558345771, m53 = 29861083;
    EXPECT_EQ(m52 % m53, 794557);
    Multiprecision m54 = 13581825744423, m55 = 71047285;
    EXPECT_EQ(m54 % m55, 460113);
    Multiprecision m56 = -10767453796459, m57 = 85680382;
    EXPECT_EQ(m56 % m57, -190519);
    Multiprecision m58 = -7637085665123, m59 = 12026053;
    EXPECT_EQ(m58 % m59, -837738);
    Multiprecision m60 = 11626011187006, m61 = 13548218;
    EXPECT_EQ(m60 % m61, 808628);
    Multiprecision m62 = -23116846973385, m63 = 70653532;
    EXPECT_EQ(m62 % m63, -452433);
    Multiprecision m64 = -10189597866429, m65 = 39212175;
    EXPECT_EQ(m64 % m65, -495279);
    Multiprecision m66 = 35873994210322, m67 = 66136687;
    EXPECT_EQ(m66 % m67, 174408);
    Multiprecision m68 = 4535653103788, m69 = 13603381;
    EXPECT_EQ(m68 % m69, 207387);
    Multiprecision m70 = 9262614882707, m71 = -68228371;
    EXPECT_EQ(m70 % m71, 67692489);
    Multiprecision m72 = -25362332495829, m73 = 31704461;
    EXPECT_EQ(m72 % m73, -169808);
    Multiprecision m74 = 46943441312445, m75 = -56547122;
    EXPECT_EQ(m74 % m75, 56324437);
    Multiprecision m76 = 6397582511119, m77 = 12738858;
    EXPECT_EQ(m76 % m77, 634939);
    Multiprecision m78 = -10713134329789, m79 = -20339953;
    EXPECT_EQ(m78 % m79, -20064830);
}

TEST(Modulo, MixedModuloAssignment) {
    Multiprecision m0 = 347971189167, m1 = -10468158;
    m0 %= m1; EXPECT_EQ(m0, 9617247);
    Multiprecision m2 = 82299524104295, m3 = -85161108;
    m2 %= m3; EXPECT_EQ(m2, 84816419);
    Multiprecision m4 = 15733272660162, m5 = 58445417;
    m4 %= m5; EXPECT_EQ(m4, 185430);
    Multiprecision m6 = 21441802118831, m7 = 32445743;
    m6 %= m7; EXPECT_EQ(m6, 411538);
    Multiprecision m8 = -225797197700, m9 = -791578;
    m8 %= m9; EXPECT_EQ(m8, -364778);
    Multiprecision m10 = 37158981415510, m11 = -85944342;
    m10 %= m11; EXPECT_EQ(m10, 85708390);
    Multiprecision m12 = -84978103484, m13 = -1337355;
    m12 %= m13; EXPECT_EQ(m12, -1229429);
    Multiprecision m14 = -36479221879833, m15 = -57772210;
    m14 %= m15; EXPECT_EQ(m14, -57547323);
    Multiprecision m16 = 19799281452078, m17 = -88989537;
    m16 %= m17; EXPECT_EQ(m16, 88354485);
    Multiprecision m18 = 62544905852708, m19 = 88569191;
    m18 %= m19; EXPECT_EQ(m18, 244238);
    Multiprecision m20 = 10563516105678, m21 = 33470367;
    m20 %= m21; EXPECT_EQ(m20, 517542);
    Multiprecision m22 = 14792471248628, m23 = 16633406;
    m22 %= m23; EXPECT_EQ(m22, 724490);
    Multiprecision m24 = -4474855045896, m25 = 30121937;
    m24 %= m25; EXPECT_EQ(m24, -329050);
    Multiprecision m26 = -67695068629974, m27 = 68710890;
    m26 %= m27; EXPECT_EQ(m26, -427734);
    Multiprecision m28 = -26367812003697, m29 = 89576749;
    m28 %= m29; EXPECT_EQ(m28, -168057);
    Multiprecision m30 = -73884520718086, m31 = 78543985;
    m30 %= m31; EXPECT_EQ(m30, -540241);
    Multiprecision m32 = 27121633606354, m33 = 33856377;
    m32 %= m33; EXPECT_EQ(m32, 975571);
    Multiprecision m34 = 7643784867640, m35 = 8394625;
    m34 %= m35; EXPECT_EQ(m34, 311515);
    Multiprecision m36 = 20823365064094, m37 = -26717860;
    m36 %= m37; EXPECT_EQ(m36, 26055154);
    Multiprecision m38 = 32317724233551, m39 = -42171193;
    m38 %= m39; EXPECT_EQ(m38, 41333966);
    Multiprecision m40 = -79175100071807, m41 = 86660675;
    m40 %= m41; EXPECT_EQ(m40, -856957);
    Multiprecision m42 = 49371377679669, m43 = -72600911;
    m42 %= m43; EXPECT_EQ(m42, 71965962);
    Multiprecision m44 = 4496522629453, m45 = -38748432;
    m44 %= m45; EXPECT_EQ(m44, 38334877);
    Multiprecision m46 = -56997212920295, m47 = 88560132;
    m46 %= m47; EXPECT_EQ(m46, -525227);
    Multiprecision m48 = 21973208639078, m49 = 86842729;
    m48 %= m49; EXPECT_EQ(m48, 819311);
    Multiprecision m50 = 14554092528935, m51 = 25879784;
    m50 %= m51; EXPECT_EQ(m50, 761503);
    Multiprecision m52 = 37798099783123, m53 = 48852995;
    m52 %= m53; EXPECT_EQ(m52, 168678);
    Multiprecision m54 = 57333872564438, m55 = -59196307;
    m54 %= m55; EXPECT_EQ(m54, 58971579);
    Multiprecision m56 = 71400434364443, m57 = 89222326;
    m56 %= m57; EXPECT_EQ(m56, 315965);
    Multiprecision m58 = -15529620519738, m59 = 70288538;
    m58 %= m59; EXPECT_EQ(m58, -645480);
    Multiprecision m60 = 19311829893256, m61 = -36260355;
    m60 %= m61; EXPECT_EQ(m60, 36204871);
    Multiprecision m62 = 21676016617116, m63 = 75122829;
    m62 %= m63; EXPECT_EQ(m62, 414627);
    Multiprecision m64 = -1999725666682, m65 = -4655668;
    m64 %= m65; EXPECT_EQ(m64, -4524650);
    Multiprecision m66 = -13442336259545, m67 = -79617243;
    m66 %= m67; EXPECT_EQ(m66, -79420397);
    Multiprecision m68 = 7141387217361, m69 = 18976547;
    m68 %= m69; EXPECT_EQ(m68, 214492);
    Multiprecision m70 = -70624997799901, m71 = -78849999;
    m70 %= m71; EXPECT_EQ(m70, -78745588);
    Multiprecision m72 = -848542427966, m73 = -1009884;
    m72 %= m73; EXPECT_EQ(m72, -525458);
    Multiprecision m74 = -18916578704681, m75 = 78085076;
    m74 %= m75; EXPECT_EQ(m74, -533225);
    Multiprecision m76 = 36446022312946, m77 = -64759312;
    m76 %= m77; EXPECT_EQ(m76, 64353154);
    Multiprecision m78 = 37344597379618, m79 = 79024810;
    m78 %= m79; EXPECT_EQ(m78, 967538);
}

TEST(Modulo, DifferentPrecision) {
    {
        Multiprecision<480> first = -47787981112424; // Multiprecision<15 blocks>
        Multiprecision<288> second = -54974147; // Multiprecision<9 blocks>
        EXPECT_EQ(first % second, -54608264);

        Multiprecision<256> third = 44015870291461; // Multiprecision<8 blocks>
        Multiprecision<416> forth = 70606483; // Multiprecision<13 blocks>
        forth %= third; EXPECT_EQ(forth, 70606483);
    }
    {
        Multiprecision<224> first = -11153390417922; // Multiprecision<7 blocks>
        Multiprecision<320> second = -26505585; // Multiprecision<10 blocks>
        EXPECT_EQ(first % second, -25789017);

        Multiprecision<192> third = -815051552509; // Multiprecision<6 blocks>
        Multiprecision<384> forth = 1855696; // Multiprecision<12 blocks>
        forth %= third; EXPECT_EQ(forth, 1855696);
    }
    {
        Multiprecision<320> first = 8647828611; // Multiprecision<10 blocks>
        Multiprecision<256> second = -9602; // Multiprecision<8 blocks>
        EXPECT_EQ(first % second, 8157);

        Multiprecision<256> third = -1836109823423; // Multiprecision<8 blocks>
        Multiprecision<480> forth = -62646625; // Multiprecision<15 blocks>
        forth %= third; EXPECT_EQ(forth, -62646625);
    }
    {
        Multiprecision<224> first = -5222275397510; // Multiprecision<7 blocks>
        Multiprecision<192> second = 69368582; // Multiprecision<6 blocks>
        EXPECT_EQ(first % second, -438804);

        Multiprecision<288> third = 40314997110280; // Multiprecision<9 blocks>
        Multiprecision<416> forth = 42485272; // Multiprecision<13 blocks>
        forth %= third; EXPECT_EQ(forth, 42485272);
    }
    {
        Multiprecision<448> first = 22136521343684; // Multiprecision<14 blocks>
        Multiprecision<320> second = -48733741; // Multiprecision<10 blocks>
        EXPECT_EQ(first % second, 47968031);

        Multiprecision<224> third = -72967413440856; // Multiprecision<7 blocks>
        Multiprecision<448> forth = 84594308; // Multiprecision<14 blocks>
        forth %= third; EXPECT_EQ(forth, 84594308);
    }
    {
        Multiprecision<448> first = 65418020788198; // Multiprecision<14 blocks>
        Multiprecision<256> second = -72254288; // Multiprecision<8 blocks>
        EXPECT_EQ(first % second, 72247318);

        Multiprecision<224> third = -17762615822405; // Multiprecision<7 blocks>
        Multiprecision<448> forth = -29064345; // Multiprecision<14 blocks>
        forth %= third; EXPECT_EQ(forth, -29064345);
    }
    {
        Multiprecision<320> first = 2514434986444; // Multiprecision<10 blocks>
        Multiprecision<384> second = -21233554; // Multiprecision<12 blocks>
        EXPECT_EQ(first % second, 21222426);

        Multiprecision<256> third = -26782009416511; // Multiprecision<8 blocks>
        Multiprecision<448> forth = -41313494; // Multiprecision<14 blocks>
        forth %= third; EXPECT_EQ(forth, -41313494);
    }
    {
        Multiprecision<192> first = 2769493530238; // Multiprecision<6 blocks>
        Multiprecision<448> second = 5536682; // Multiprecision<14 blocks>
        EXPECT_EQ(first % second, 900382);

        Multiprecision<288> third = 42554436184507; // Multiprecision<9 blocks>
        Multiprecision<384> forth = -59942665; // Multiprecision<12 blocks>
        forth %= third; EXPECT_EQ(forth, -59942665);
    }
    {
        Multiprecision<448> first = 7524884518456; // Multiprecision<14 blocks>
        Multiprecision<480> second = 47489094; // Multiprecision<15 blocks>
        EXPECT_EQ(first % second, 128686);

        Multiprecision<192> third = 30527365286483; // Multiprecision<6 blocks>
        Multiprecision<480> forth = 36978304; // Multiprecision<15 blocks>
        forth %= third; EXPECT_EQ(forth, 36978304);
    }
    {
        Multiprecision<224> first = 21272610186684; // Multiprecision<7 blocks>
        Multiprecision<320> second = -33061266; // Multiprecision<10 blocks>
        EXPECT_EQ(first % second, 32865570);

        Multiprecision<192> third = -51270397025387; // Multiprecision<6 blocks>
        Multiprecision<416> forth = 65171554; // Multiprecision<13 blocks>
        forth %= third; EXPECT_EQ(forth, 65171554);
    }
}

TEST(Modulo, Huge) { /* PASS */ }