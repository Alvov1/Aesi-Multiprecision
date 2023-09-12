#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Division, SmallPositive) {
    Multiprecision small1 = -8492, small2 = 4243, small3 = -678, small4 = 2323;
    EXPECT_EQ(small1 / small3, 12);
    EXPECT_EQ(small3 / small1, 0);
    EXPECT_EQ(small2 / small4, 1);
    EXPECT_EQ(small4 / small2, 0);
}

TEST(Division, SmallNegative) {
    Multiprecision small1 = -8492, small2 = 4243, small3 = -678, small4 = 2323;
    EXPECT_EQ(small2 / small3, -6);
    EXPECT_EQ(small3 / small2, 0);
    EXPECT_EQ(small1 / small4, -3);
    EXPECT_EQ(small4 / small1, 0);
}

TEST(Division, MixedDivision) {
    Multiprecision m0 = -67602113121365, m1 = -70814915;
    EXPECT_EQ(m0 / m1, 954631);
    Multiprecision m2 = 15113324630360, m3 = 57523730;
    EXPECT_EQ(m2 / m3, 262732);
    Multiprecision m4 = 34122183947714, m5 = -44752342;
    EXPECT_EQ(m4 / m5, -762467);
    Multiprecision m6 = 2220619718304, m7 = -6807624;
    EXPECT_EQ(m6 / m7, -326196);
    Multiprecision m8 = -56544544247220, m9 = -64028010;
    EXPECT_EQ(m8 / m9, 883122);
    Multiprecision m10 = 27311885651556, m11 = 40063554;
    EXPECT_EQ(m10 / m11, 681714);
    Multiprecision m12 = 65477763597661, m13 = -85123723;
    EXPECT_EQ(m12 / m13, -769207);
    Multiprecision m14 = -40861948934173, m15 = -56440949;
    EXPECT_EQ(m14 / m15, 723977);
    Multiprecision m16 = 176062774875, m17 = -14641395;
    EXPECT_EQ(m16 / m17, -12025);
    Multiprecision m18 = -329952273599, m19 = -3832777;
    EXPECT_EQ(m18 / m19, 86087);
    Multiprecision m20 = -1088200002846, m21 = -1883706;
    EXPECT_EQ(m20 / m21, 577691);
    Multiprecision m22 = 6974639128232, m23 = -10309004;
    EXPECT_EQ(m22 / m23, -676558);
    Multiprecision m24 = 31834533607632, m25 = -55868976;
    EXPECT_EQ(m24 / m25, -569807);
    Multiprecision m26 = 31126617188568, m27 = -54848568;
    EXPECT_EQ(m26 / m27, -567501);
    Multiprecision m28 = 6290940249900, m29 = 21494628;
    EXPECT_EQ(m28 / m29, 292675);
    Multiprecision m30 = 42187650007176, m31 = -78019518;
    EXPECT_EQ(m30 / m31, -540732);
    Multiprecision m32 = -20062099158147, m33 = -64832101;
    EXPECT_EQ(m32 / m33, 309447);
    Multiprecision m34 = -9216212918125, m35 = 26711725;
    EXPECT_EQ(m34 / m35, -345025);
    Multiprecision m36 = -5364545160432, m37 = -31267749;
    EXPECT_EQ(m36 / m37, 171568);
    Multiprecision m38 = -19899125279059, m39 = 26608409;
    EXPECT_EQ(m38 / m39, -747851);
    Multiprecision m40 = -5126477678000, m41 = -7908790;
    EXPECT_EQ(m40 / m41, 648200);
    Multiprecision m42 = -54479717879628, m43 = -56742436;
    EXPECT_EQ(m42 / m43, 960123);
    Multiprecision m44 = -11035329100380, m45 = 85511380;
    EXPECT_EQ(m44 / m45, -129051);
    Multiprecision m46 = -77240330776750, m47 = 77342810;
    EXPECT_EQ(m46 / m47, -998675);
    Multiprecision m48 = -25940256135750, m49 = 32187375;
    EXPECT_EQ(m48 / m49, -805914);
    Multiprecision m50 = 68475350540736, m51 = -72959904;
    EXPECT_EQ(m50 / m51, -938534);
    Multiprecision m52 = -20926969579065, m53 = -77395215;
    EXPECT_EQ(m52 / m53, 270391);
    Multiprecision m54 = -1875340132380, m55 = -17179580;
    EXPECT_EQ(m54 / m55, 109161);
    Multiprecision m56 = -287061254477, m57 = -22981447;
    EXPECT_EQ(m56 / m57, 12491);
    Multiprecision m58 = -22187933279568, m59 = -49281768;
    EXPECT_EQ(m58 / m59, 450226);
    Multiprecision m60 = -979862770236, m61 = 2721327;
    EXPECT_EQ(m60 / m61, -360068);
    Multiprecision m62 = -1404488473664, m63 = -39927464;
    EXPECT_EQ(m62 / m63, 35176);
    Multiprecision m64 = -1465619419940, m65 = 26127452;
    EXPECT_EQ(m64 / m65, -56095);
    Multiprecision m66 = 49073567920942, m67 = 77753978;
    EXPECT_EQ(m66 / m67, 631139);
    Multiprecision m68 = 8897687795196, m69 = -49216686;
    EXPECT_EQ(m68 / m69, -180786);
    Multiprecision m70 = 6085886956392, m71 = -16070088;
    EXPECT_EQ(m70 / m71, -378709);
    Multiprecision m72 = 50224482468354, m73 = 75335443;
    EXPECT_EQ(m72 / m73, 666678);
    Multiprecision m74 = 7740677726208, m75 = 15847496;
    EXPECT_EQ(m74 / m75, 488448);
    Multiprecision m76 = 1063246222290, m77 = 8595570;
    EXPECT_EQ(m76 / m77, 123697);
    Multiprecision m78 = -10332555271144, m79 = 29755036;
    EXPECT_EQ(m78 / m79, -347254);
}

TEST(Division, MixedDivisionAssignment) {
    Multiprecision m0 = -3174698500920, m1 = 37446314;
    m0 /= m1; EXPECT_EQ(m0, -84780);
    Multiprecision m2 = 156000466440, m3 = -66666866;
    m2 /= m3; EXPECT_EQ(m2, -2340);
    Multiprecision m4 = -9664086627036, m5 = -51407724;
    m4 /= m5; EXPECT_EQ(m4, 187989);
    Multiprecision m6 = 683516898168, m7 = 1805609;
    m6 /= m7; EXPECT_EQ(m6, 378552);
    Multiprecision m8 = -24751231123115, m9 = -89301431;
    m8 /= m9; EXPECT_EQ(m8, 277165);
    Multiprecision m10 = 26354396493600, m11 = -49311248;
    m10 /= m11; EXPECT_EQ(m10, -534450);
    Multiprecision m12 = -1203507178300, m13 = 67055225;
    m12 /= m13; EXPECT_EQ(m12, -17948);
    Multiprecision m14 = -9860315571000, m15 = -32244750;
    m14 /= m15; EXPECT_EQ(m14, 305796);
    Multiprecision m16 = -13210718718360, m17 = 32872297;
    m16 /= m17; EXPECT_EQ(m16, -401880);
    Multiprecision m18 = -1166273423250, m19 = 5292222;
    m18 /= m19; EXPECT_EQ(m18, -220375);
    Multiprecision m20 = -17154168020562, m21 = -32960958;
    m20 /= m21; EXPECT_EQ(m20, 520439);
    Multiprecision m22 = -24461813157015, m23 = -33125755;
    m22 /= m23; EXPECT_EQ(m22, 738453);
    Multiprecision m24 = -3915356462184, m25 = -67492182;
    m24 /= m25; EXPECT_EQ(m24, 58012);
    Multiprecision m26 = 6239712095136, m27 = 46955376;
    m26 /= m27; EXPECT_EQ(m26, 132886);
    Multiprecision m28 = -41815750742532, m29 = 55578484;
    m28 /= m29; EXPECT_EQ(m28, -752373);
    Multiprecision m30 = -58740024955500, m31 = -66670100;
    m30 /= m31; EXPECT_EQ(m30, 881055);
    Multiprecision m32 = 25823556551045, m33 = 37417039;
    m32 /= m33; EXPECT_EQ(m32, 690155);
    Multiprecision m34 = 5480709811479, m35 = -54757269;
    m34 /= m35; EXPECT_EQ(m34, -100091);
    Multiprecision m36 = -22711656353225, m37 = -79507295;
    m36 /= m37; EXPECT_EQ(m36, 285655);
    Multiprecision m38 = 12409303686885, m39 = 17897733;
    m38 /= m39; EXPECT_EQ(m38, 693345);
    Multiprecision m40 = 13872981907785, m41 = -36991955;
    m40 /= m41; EXPECT_EQ(m40, -375027);
    Multiprecision m42 = -643299920236, m43 = 67038341;
    m42 /= m43; EXPECT_EQ(m42, -9596);
    Multiprecision m44 = 3282278192935, m45 = 4741265;
    m44 /= m45; EXPECT_EQ(m44, 692279);
    Multiprecision m46 = -13386640546113, m47 = -57296749;
    m46 /= m47; EXPECT_EQ(m46, 233637);
    Multiprecision m48 = -368301167920, m49 = 12409069;
    m48 /= m49; EXPECT_EQ(m48, -29680);
    Multiprecision m50 = 4999702954272, m51 = -74294207;
    m50 /= m51; EXPECT_EQ(m50, -67296);
    Multiprecision m52 = -13934020168638, m53 = -18893766;
    m52 /= m53; EXPECT_EQ(m52, 737493);
    Multiprecision m54 = 18163193044176, m55 = 18950076;
    m54 /= m55; EXPECT_EQ(m54, 958476);
    Multiprecision m56 = -80852334667440, m57 = 82227173;
    m56 /= m57; EXPECT_EQ(m56, -983280);
    Multiprecision m58 = -13961168504878, m59 = -14066626;
    m58 /= m59; EXPECT_EQ(m58, 992503);
    Multiprecision m60 = 35163604583850, m61 = 77817450;
    m60 /= m61; EXPECT_EQ(m60, 451873);
    Multiprecision m62 = -71110375898000, m63 = 74848300;
    m62 /= m63; EXPECT_EQ(m62, -950060);
    Multiprecision m64 = -15288429937525, m65 = 45382825;
    m64 /= m65; EXPECT_EQ(m64, -336877);
    Multiprecision m66 = -34335775193105, m67 = 56421811;
    m66 /= m67; EXPECT_EQ(m66, -608555);
    Multiprecision m68 = -20503984546510, m69 = -22501931;
    m68 /= m69; EXPECT_EQ(m68, 911210);
    Multiprecision m70 = -37499367266592, m71 = 43862808;
    m70 /= m71; EXPECT_EQ(m70, -854924);
    Multiprecision m72 = 44304906945178, m73 = -47076238;
    m72 /= m73; EXPECT_EQ(m72, -941131);
    Multiprecision m74 = -39150837269096, m75 = 90134744;
    m74 /= m75; EXPECT_EQ(m74, -434359);
    Multiprecision m76 = -22502676678930, m77 = -89545785;
    m76 /= m77; EXPECT_EQ(m76, 251298);
    Multiprecision m78 = -34136928128166, m79 = -79394298;
    m78 /= m79; EXPECT_EQ(m78, 429967);
}

TEST(Division, DifferentPrecision) {
    {
        Multiprecision<256> first = 11921542526290; // Multiprecision<8 blocks>
        Multiprecision<352> second = 88322770; // Multiprecision<11 blocks>
        EXPECT_EQ(first / second, 134977);

        Multiprecision<320> third = -4102459315740; // Multiprecision<10 blocks>
        Multiprecision<448> forth = 57049914; // Multiprecision<14 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Multiprecision<320> first = 13488766159653; // Multiprecision<10 blocks>
        Multiprecision<352> second = 48109389; // Multiprecision<11 blocks>
        EXPECT_EQ(first / second, 280377);

        Multiprecision<320> third = -991821198108; // Multiprecision<10 blocks>
        Multiprecision<416> forth = 4233378; // Multiprecision<13 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Multiprecision<480> first = -73311512255046; // Multiprecision<15 blocks>
        Multiprecision<384> second = 86942434; // Multiprecision<12 blocks>
        EXPECT_EQ(first / second, -843219);

        Multiprecision<224> third = 721422850272; // Multiprecision<7 blocks>
        Multiprecision<416> forth = 15138768; // Multiprecision<13 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Multiprecision<256> first = 57155718149216; // Multiprecision<8 blocks>
        Multiprecision<256> second = 80590001; // Multiprecision<8 blocks>
        EXPECT_EQ(first / second, 709216);

        Multiprecision<192> third = 657315248892; // Multiprecision<6 blocks>
        Multiprecision<448> forth = 4018458; // Multiprecision<14 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Multiprecision<224> first = -24101170189740; // Multiprecision<7 blocks>
        Multiprecision<224> second = -64694181; // Multiprecision<7 blocks>
        EXPECT_EQ(first / second, 372540);

        Multiprecision<288> third = 27530629582832; // Multiprecision<9 blocks>
        Multiprecision<384> forth = 49146568; // Multiprecision<12 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Multiprecision<448> first = 10440263901354; // Multiprecision<14 blocks>
        Multiprecision<352> second = -25782631; // Multiprecision<11 blocks>
        EXPECT_EQ(first / second, -404934);

        Multiprecision<192> third = 49691966023092; // Multiprecision<6 blocks>
        Multiprecision<416> forth = -57556284; // Multiprecision<13 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Multiprecision<416> first = -19076120667460; // Multiprecision<13 blocks>
        Multiprecision<384> second = 43657645; // Multiprecision<12 blocks>
        EXPECT_EQ(first / second, -436948);

        Multiprecision<224> third = 3594292228818; // Multiprecision<7 blocks>
        Multiprecision<384> forth = 8218869; // Multiprecision<12 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Multiprecision<192> first = -80928002017452; // Multiprecision<6 blocks>
        Multiprecision<480> second = -83298854; // Multiprecision<15 blocks>
        EXPECT_EQ(first / second, 971538);

        Multiprecision<256> third = 17240509748020; // Multiprecision<8 blocks>
        Multiprecision<352> forth = 35593090; // Multiprecision<11 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Multiprecision<192> first = -34515574677948; // Multiprecision<6 blocks>
        Multiprecision<224> second = 72218589; // Multiprecision<7 blocks>
        EXPECT_EQ(first / second, -477932);

        Multiprecision<320> third = -14100957360246; // Multiprecision<10 blocks>
        Multiprecision<448> forth = -35406678; // Multiprecision<14 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Multiprecision<480> first = -1093297575920; // Multiprecision<15 blocks>
        Multiprecision<448> second = -22640248; // Multiprecision<14 blocks>
        EXPECT_EQ(first / second, 48290);

        Multiprecision<320> third = 6343088533611; // Multiprecision<10 blocks>
        Multiprecision<352> forth = 8322199; // Multiprecision<11 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
}

TEST(Division, Huge) { EXPECT_TRUE(false); }