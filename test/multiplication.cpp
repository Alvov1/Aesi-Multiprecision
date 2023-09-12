#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Multiplication, ZeroOne) {
    Multiprecision zero = 0, one = 1;
    Multiprecision m0 = 10919396;
    EXPECT_EQ(m0 * 0, 0);
    EXPECT_EQ(0 * m0, 0);
    EXPECT_EQ(m0 * 1, 10919396);
    EXPECT_EQ(1 * m0, 10919396);
    EXPECT_EQ(m0 * -1, -10919396);
    EXPECT_EQ(-1 * m0, -10919396);

    EXPECT_EQ(m0 * zero, 0);
    EXPECT_EQ(m0 * -zero, 0);
    EXPECT_EQ(m0 * +zero, 0);
    EXPECT_EQ(m0 * one, 10919396);
    EXPECT_EQ(m0 * -one, -10919396);
    EXPECT_EQ(m0 * +one, 10919396);

    EXPECT_EQ(zero * m0, 0);
    EXPECT_EQ(zero * -m0, 0);
    EXPECT_EQ(zero * +m0, 0);
    EXPECT_EQ(one * m0, 10919396);
    EXPECT_EQ(one * -m0, -10919396);
    EXPECT_EQ(one * +m0, 10919396);

    EXPECT_EQ(+zero * m0, 0);
    EXPECT_EQ(+zero * -m0, 0);
    EXPECT_EQ(+zero * +m0, 0);
    EXPECT_EQ(+one * m0, 10919396);
    EXPECT_EQ(+one * -m0, -10919396);
    EXPECT_EQ(+one * +m0, 10919396);

    EXPECT_EQ(-zero * m0, 0);
    EXPECT_EQ(-zero * -m0, 0);
    EXPECT_EQ(-zero * +m0, 0);
    EXPECT_EQ(-one * m0, -10919396);
    EXPECT_EQ(-one * -m0, 10919396);
    EXPECT_EQ(-one * +m0, -10919396);

    m0 *= 0; EXPECT_EQ(m0, 0); m0 = 10919396;
    m0 *= zero; EXPECT_EQ(m0, 0); m0 = 10919396;
    m0 *= -zero; EXPECT_EQ(m0, 0); m0 = 10919396;
    m0 *= +zero; EXPECT_EQ(m0, 0); m0 = 10919396;
    zero *= m0; EXPECT_EQ(zero, 0);
    zero *= +m0; EXPECT_EQ(zero, 0);
    zero *= -m0; EXPECT_EQ(zero, 0);

    m0 *= 1; EXPECT_EQ(m0, 10919396);
    m0 *= -1; EXPECT_EQ(m0, -10919396); m0 *= -1;

    m0 *= one; EXPECT_EQ(m0, 10919396);
    m0 *= -one; EXPECT_EQ(m0, -10919396); m0 *= -1;
    m0 *= +one; EXPECT_EQ(m0, 10919396);
    one *= m0; EXPECT_EQ(one, 10919396); one = 1;
    one *= +m0; EXPECT_EQ(one, 10919396); one = 1;
    one *= -m0; EXPECT_EQ(one, -10919396); one = 1;



    Multiprecision m1 = -14144694;
    EXPECT_EQ(m1 * 0, 0);
    EXPECT_EQ(0 * m1, 0);
    EXPECT_EQ(m1 * 1, -14144694);
    EXPECT_EQ(1 * m1, -14144694);
    EXPECT_EQ(m1 * -1, 14144694);
    EXPECT_EQ(-1 * m1, 14144694);

    EXPECT_EQ(m1 * zero, 0);
    EXPECT_EQ(m1 * -zero, 0);
    EXPECT_EQ(m1 * +zero, 0);
    EXPECT_EQ(m1 * one, -14144694);
    EXPECT_EQ(m1 * -one, 14144694);
    EXPECT_EQ(m1 * +one, -14144694);

    EXPECT_EQ(zero * m1, 0);
    EXPECT_EQ(zero * -m1, 0);
    EXPECT_EQ(zero * +m1, 0);
    EXPECT_EQ(one * m1, -14144694);
    EXPECT_EQ(one * -m1, 14144694);
    EXPECT_EQ(one * +m1, -14144694);

    EXPECT_EQ(+zero * m1, 0);
    EXPECT_EQ(+zero * -m1, 0);
    EXPECT_EQ(+zero * +m1, 0);
    EXPECT_EQ(+one * m1, -14144694);
    EXPECT_EQ(+one * -m1, 14144694);
    EXPECT_EQ(+one * +m1, -14144694);

    EXPECT_EQ(-zero * m1, 0);
    EXPECT_EQ(-zero * -m1, 0);
    EXPECT_EQ(-zero * +m1, 0);
    EXPECT_EQ(-one * m1, 14144694);
    EXPECT_EQ(-one * -m1, -14144694);
    EXPECT_EQ(-one * +m1, 14144694);

    m1 *= 0; EXPECT_EQ(m1, 0); m1 = -14144694;
    m1 *= zero; EXPECT_EQ(m1, 0); m1 = -14144694;
    m1 *= -zero; EXPECT_EQ(m1, 0); m1 = -14144694;
    m1 *= +zero; EXPECT_EQ(m1, 0); m1 = -14144694;
    zero *= m1; EXPECT_EQ(zero, 0);
    zero *= +m1; EXPECT_EQ(zero, 0);
    zero *= -m1; EXPECT_EQ(zero, 0);

    m1 *= 1; EXPECT_EQ(m1, -14144694);
    m1 *= -1; EXPECT_EQ(m1, 14144694); m1 *= -1;

    m1 *= one; EXPECT_EQ(m1, -14144694);
    m1 *= -one; EXPECT_EQ(m1, 14144694); m1 *= -1;
    m1 *= +one; EXPECT_EQ(m1, -14144694);
    one *= m1; EXPECT_EQ(one, -14144694); one = 1;
    one *= +m1; EXPECT_EQ(one, -14144694); one = 1;
    one *= -m1; EXPECT_EQ(one, 14144694); one = 1;



    Multiprecision m2 = -49285963;
    EXPECT_EQ(m2 * 0, 0);
    EXPECT_EQ(0 * m2, 0);
    EXPECT_EQ(m2 * 1, -49285963);
    EXPECT_EQ(1 * m2, -49285963);
    EXPECT_EQ(m2 * -1, 49285963);
    EXPECT_EQ(-1 * m2, 49285963);

    EXPECT_EQ(m2 * zero, 0);
    EXPECT_EQ(m2 * -zero, 0);
    EXPECT_EQ(m2 * +zero, 0);
    EXPECT_EQ(m2 * one, -49285963);
    EXPECT_EQ(m2 * -one, 49285963);
    EXPECT_EQ(m2 * +one, -49285963);

    EXPECT_EQ(zero * m2, 0);
    EXPECT_EQ(zero * -m2, 0);
    EXPECT_EQ(zero * +m2, 0);
    EXPECT_EQ(one * m2, -49285963);
    EXPECT_EQ(one * -m2, 49285963);
    EXPECT_EQ(one * +m2, -49285963);

    EXPECT_EQ(+zero * m2, 0);
    EXPECT_EQ(+zero * -m2, 0);
    EXPECT_EQ(+zero * +m2, 0);
    EXPECT_EQ(+one * m2, -49285963);
    EXPECT_EQ(+one * -m2, 49285963);
    EXPECT_EQ(+one * +m2, -49285963);

    EXPECT_EQ(-zero * m2, 0);
    EXPECT_EQ(-zero * -m2, 0);
    EXPECT_EQ(-zero * +m2, 0);
    EXPECT_EQ(-one * m2, 49285963);
    EXPECT_EQ(-one * -m2, -49285963);
    EXPECT_EQ(-one * +m2, 49285963);

    m2 *= 0; EXPECT_EQ(m2, 0); m2 = -49285963;
    m2 *= zero; EXPECT_EQ(m2, 0); m2 = -49285963;
    m2 *= -zero; EXPECT_EQ(m2, 0); m2 = -49285963;
    m2 *= +zero; EXPECT_EQ(m2, 0); m2 = -49285963;
    zero *= m2; EXPECT_EQ(zero, 0);
    zero *= +m2; EXPECT_EQ(zero, 0);
    zero *= -m2; EXPECT_EQ(zero, 0);

    m2 *= 1; EXPECT_EQ(m2, -49285963);
    m2 *= -1; EXPECT_EQ(m2, 49285963); m2 *= -1;

    m2 *= one; EXPECT_EQ(m2, -49285963);
    m2 *= -one; EXPECT_EQ(m2, 49285963); m2 *= -1;
    m2 *= +one; EXPECT_EQ(m2, -49285963);
    one *= m2; EXPECT_EQ(one, -49285963); one = 1;
    one *= +m2; EXPECT_EQ(one, -49285963); one = 1;
    one *= -m2; EXPECT_EQ(one, 49285963); one = 1;
}

TEST(Multiplication, SmallPositive) {
    Multiprecision small1 = -8492, small2 = 4243, small3 = -678, small4 = 2323;
    EXPECT_EQ(small1 * small3, 5757576); //5766068
    EXPECT_EQ(small3 * small1, 5757576);
    EXPECT_EQ(small2 * small4, 9856489);
    EXPECT_EQ(small4 * small2, 9856489);
}

TEST(Multiplication, SmallNegative) {
    Multiprecision small1 = -8492, small2 = 4243, small3 = -678, small4 = 2323;
    EXPECT_EQ(small2 * small1, -36031556);
    EXPECT_EQ(small1 * small2, -36031556);
    EXPECT_EQ(small3 * small2, -2876754);
    EXPECT_EQ(small2 * small3, -2876754);
}

TEST(Multiplication, MixedMultiplication) {
    Multiprecision m0 = -632224826, m1 = 854946415;
    EXPECT_EQ(m0 * m1, -540518348462698790);
    Multiprecision m2 = 15124283, m3 = 855984917;
    EXPECT_EQ(m2 * m3, 12946158128439511);
    Multiprecision m4 = -255413170, m5 = -976424326;
    EXPECT_EQ(m4 * m5, 249391632368773420);
    Multiprecision m6 = -436691566, m7 = -215084769;
    EXPECT_EQ(m6 * m7, 93925704597358254);
    Multiprecision m8 = -508779305, m9 = -969810292;
    EXPECT_EQ(m8 * m9, 493419406345607060);
    Multiprecision m10 = 389511413, m11 = 558722925;
    EXPECT_EQ(m10 * m11, 217628955992243025);
    Multiprecision m12 = 33861785, m13 = -471602607;
    EXPECT_EQ(m12 * m13, -15969306083673495);
    Multiprecision m14 = 9170436, m15 = -469382092;
    EXPECT_EQ(m14 * m15, -4304438434232112);
    Multiprecision m16 = 459781677, m17 = -589479067;
    EXPECT_EQ(m16 * m17, -271031673981655359);
    Multiprecision m18 = -605361046, m19 = 568168204;
    EXPECT_EQ(m18 * m19, -343946898277381384);
    Multiprecision m20 = 814926234, m21 = -397421695;
    EXPECT_EQ(m20 * m21, -323869365216246630);
    Multiprecision m22 = 855291014, m23 = -737527571;
    EXPECT_EQ(m22 * m23, -630800704053546994);
    Multiprecision m24 = -683591332, m25 = -662674041;
    EXPECT_EQ(m24 * m25, 452998230369012612);
    Multiprecision m26 = -115005179, m27 = -86969330;
    EXPECT_EQ(m26 * m27, 10001923364160070);
    Multiprecision m28 = -698162382, m29 = -709316011;
    EXPECT_EQ(m28 * m29, 495217755830498202);
    Multiprecision m30 = 264231125, m31 = 576272843;
    EXPECT_EQ(m30 * m31, 152269221612838375);
    Multiprecision m32 = -416242575, m33 = -775215804;
    EXPECT_EQ(m32 * m33, 322677822437655300);
    Multiprecision m34 = -631448808, m35 = 169513175;
    EXPECT_EQ(m34 * m35, -107038892294045400);
    Multiprecision m36 = 882601391, m37 = -396333897;
    EXPECT_EQ(m36 * m37, -349804848792650727);
    Multiprecision m38 = -364424759, m39 = -814894879;
    EXPECT_EQ(m38 * m39, 296967869889909161);
    Multiprecision m40 = 37054480, m41 = 755633477;
    EXPECT_EQ(m40 * m41, 27999605560826960);
    Multiprecision m42 = -479511582, m43 = 753322217;
    EXPECT_EQ(m42 * m43, -361226728029417294);
    Multiprecision m44 = -409793824, m45 = -424625374;
    EXPECT_EQ(m44 * m45, 174008855778890176);
    Multiprecision m46 = 183576582, m47 = -987453846;
    EXPECT_EQ(m46 * m47, -181273401931434372);
    Multiprecision m48 = -562035517, m49 = 439637934;
    EXPECT_EQ(m48 * m49, -247092133528501878);
    Multiprecision m50 = -499481757, m51 = -930190942;
    EXPECT_EQ(m50 * m51, 464613406055645094);
    Multiprecision m52 = 220311749, m53 = -446455934;
    EXPECT_EQ(m52 * m53, -98359487670968566);
    Multiprecision m54 = -893464199, m55 = 503713658;
    EXPECT_EQ(m54 * m55, -450050119970329942);
    Multiprecision m56 = -510392458, m57 = -499319405;
    EXPECT_EQ(m56 * m57, 254848858445047490);
    Multiprecision m58 = 978772862, m59 = 38529994;
    EXPECT_EQ(m58 * m59, 37712112500222828);
    Multiprecision m60 = 302637761, m61 = -377843880;
    EXPECT_EQ(m60 * m61, -114349825850752680);
    Multiprecision m62 = -606373345, m63 = -957840592;
    EXPECT_EQ(m62 * m63, 580809003747820240);
    Multiprecision m64 = -482327274, m65 = 72347072;
    EXPECT_EQ(m64 * m65, -34894966019641728);
    Multiprecision m66 = -124692430, m67 = -448472121;
    EXPECT_EQ(m66 * m67, 55921078554744030);
    Multiprecision m68 = 100886786, m69 = 486116510;
    EXPECT_EQ(m68 * m69, 49042732315436860);
    Multiprecision m70 = 417006369, m71 = -824160488;
    EXPECT_EQ(m70 * m71, -343680172574148072);
    Multiprecision m72 = 602004950, m73 = -699239102;
    EXPECT_EQ(m72 * m73, -420945400637554900);
    Multiprecision m74 = 333271488, m75 = -742295106;
    EXPECT_EQ(m74 * m75, -247385794511737728);
    Multiprecision m76 = -321447732, m77 = -339153497;
    EXPECT_EQ(m76 * m77, 109020122410518804);
    Multiprecision m78 = -293108259, m79 = 82767790;
    EXPECT_EQ(m78 * m79, -24259922828177610);
}

TEST(Multiplication, MixedMultiplicationAssignment) {
    Multiprecision m0 = -78990477, m1 = -924464422;
    m0 *= m1; EXPECT_EQ(m0, 73023885663309294);
    Multiprecision m2 = -130189339, m3 = 753383660;
    m2 *= m3; EXPECT_EQ(m2, -98082520708800740);
    Multiprecision m4 = -790677253, m5 = 866863439;
    m4 *= m5; EXPECT_EQ(m4, -685409202674653067);
    Multiprecision m6 = 166592228, m7 = 593311507;
    m6 *= m7; EXPECT_EQ(m6, 98841085849167596);
    Multiprecision m8 = -437879590, m9 = 808169924;
    m8 *= m9; EXPECT_EQ(m8, -353881114971451160);
    Multiprecision m10 = -787058027, m11 = 414778979;
    m10 *= m11; EXPECT_EQ(m10, -326455124852814433);
    Multiprecision m12 = -918620953, m13 = -322386314;
    m12 *= m13; EXPECT_EQ(m12, 296150823000837242);
    Multiprecision m14 = 474122456, m15 = -84479346;
    m14 *= m15; EXPECT_EQ(m14, -40053555006793776);
    Multiprecision m16 = 727097635, m17 = -47548650;
    m16 *= m17; EXPECT_EQ(m16, -34572510962442750);
    Multiprecision m18 = 897765113, m19 = 975202363;
    m18 *= m19; EXPECT_EQ(m18, 875502659616562019);
    Multiprecision m20 = -454181230, m21 = -785982875;
    m20 *= m21; EXPECT_EQ(m20, 356978668926436250);
    Multiprecision m22 = -766757515, m23 = 755940205;
    m22 *= m23; EXPECT_EQ(m22, -579622833074390575);
    Multiprecision m24 = -676345196, m25 = -868942436;
    m24 *= m25; EXPECT_EQ(m24, 587705042189137456);
    Multiprecision m26 = -764919638, m27 = -644748000;
    m26 *= m27; EXPECT_EQ(m26, 493180406761224000);
    Multiprecision m28 = -36106476, m29 = 5898500;
    m28 *= m29; EXPECT_EQ(m28, -212974048686000);
    Multiprecision m30 = -508311768, m31 = -400158082;
    m30 *= m31; EXPECT_EQ(m30, 203405062140908976);
    Multiprecision m32 = 699063090, m33 = -567008183;
    m32 *= m33; EXPECT_EQ(m32, -396374492463265470);
    Multiprecision m34 = -217982726, m35 = -203957548;
    m34 *= m35; EXPECT_EQ(m34, 44459222301315848);
    Multiprecision m36 = -94926525, m37 = -919379377;
    m36 *= m37; EXPECT_EQ(m36, 87273489415274925);
    Multiprecision m38 = 228581475, m39 = -75340786;
    m38 *= m39; EXPECT_EQ(m38, -17221507991539350);
    Multiprecision m40 = -548442627, m41 = 276540841;
    m40 *= m41; EXPECT_EQ(m40, -151666785310829307);
    Multiprecision m42 = 311095923, m43 = -635342106;
    m42 *= m43; EXPECT_EQ(m42, -197652338886833838);
    Multiprecision m44 = -48816753, m45 = 342286992;
    m44 *= m45; EXPECT_EQ(m44, -16709339543576976);
    Multiprecision m46 = -315839248, m47 = -928470470;
    m46 *= m47; EXPECT_EQ(m46, 293247415035006560);
    Multiprecision m48 = -249050979, m49 = -516669820;
    m48 *= m49; EXPECT_EQ(m48, 128677124490753780);
    Multiprecision m50 = 171420914, m51 = 717817322;
    m50 *= m51; EXPECT_EQ(m50, 123048901422272308);
    Multiprecision m52 = -41298512, m53 = 224451575;
    m52 *= m53; EXPECT_EQ(m52, -9269516063556400);
    Multiprecision m54 = -948716932, m55 = -386328174;
    m54 *= m55; EXPECT_EQ(m54, 366516079982442168);
    Multiprecision m56 = 648648935, m57 = -707788376;
    m56 *= m57; EXPECT_EQ(m56, -459106176297779560);
    Multiprecision m58 = 103993409, m59 = -766917779;
    m58 *= m59; EXPECT_EQ(m58, -79754394260918611);
    Multiprecision m60 = 32082502, m61 = -164829182;
    m60 *= m61; EXPECT_EQ(m60, -5288132561173364);
    Multiprecision m62 = -501474190, m63 = -188061417;
    m62 *= m63; EXPECT_EQ(m62, 94307946760327230);
    Multiprecision m64 = -520504227, m65 = -366873757;
    m64 *= m65; EXPECT_EQ(m64, 190959341293870839);
    Multiprecision m66 = 151343708, m67 = -769245438;
    m66 *= m67; EXPECT_EQ(m66, -116420456949004104);
    Multiprecision m68 = 582249656, m69 = -197127361;
    m68 *= m69; EXPECT_EQ(m68, -114777338130437816);
    Multiprecision m70 = -69879403, m71 = -573718443;
    m70 *= m71; EXPECT_EQ(m70, 40091102286929529);
    Multiprecision m72 = 316032836, m73 = -595187191;
    m72 *= m73; EXPECT_EQ(m72, -188098695922603676);
    Multiprecision m74 = 594592468, m75 = 445255386;
    m74 *= m75; EXPECT_EQ(m74, 264745498852032648);
    Multiprecision m76 = -933643542, m77 = 838661995;
    m76 *= m77; EXPECT_EQ(m76, -783011355552586290);
    Multiprecision m78 = -887038650, m79 = 333513342;
    m78 *= m79; EXPECT_EQ(m78, -295839224644668300);
}

TEST(Multiplication, DifferentPrecision) {
    {
        Multiprecision<320> first = 80176598; // Multiprecision<10 blocks>
        Multiprecision<480> second = -426614748; // Multiprecision<15 blocks>
        EXPECT_EQ(first * second, -34204519151267304);

        Multiprecision<320> third = 638380867; // Multiprecision<10 blocks>
        Multiprecision<352> forth = 534963661; // Multiprecision<11 blocks>
        forth *= third; EXPECT_EQ(forth, 341510565722674087);
    }
    {
        Multiprecision<352> first = 118510918; // Multiprecision<11 blocks>
        Multiprecision<256> second = 392821062; // Multiprecision<8 blocks>
        EXPECT_EQ(first * second, 46553584667354916);

        Multiprecision<320> third = 490743664; // Multiprecision<10 blocks>
        Multiprecision<416> forth = -755654452; // Multiprecision<13 blocks>
        forth *= third; EXPECT_EQ(forth, -370832634492392128);
    }
    {
        Multiprecision<480> first = 368054998; // Multiprecision<15 blocks>
        Multiprecision<352> second = -412866557; // Multiprecision<11 blocks>
        EXPECT_EQ(first * second, -151957599810901886);

        Multiprecision<256> third = -806407399; // Multiprecision<8 blocks>
        Multiprecision<416> forth = -755378465; // Multiprecision<13 blocks>
        forth *= third; EXPECT_EQ(forth, 609142783221262535);
    }
    {
        Multiprecision<288> first = 929534487; // Multiprecision<9 blocks>
        Multiprecision<480> second = -405487320; // Multiprecision<15 blocks>
        EXPECT_EQ(first * second, -376914447981204840);

        Multiprecision<320> third = -797426321; // Multiprecision<10 blocks>
        Multiprecision<352> forth = 271542653; // Multiprecision<11 blocks>
        forth *= third; EXPECT_EQ(forth, -216535258776369613);
    }
    {
        Multiprecision<256> first = -143407538; // Multiprecision<8 blocks>
        Multiprecision<320> second = 342454657; // Multiprecision<10 blocks>
        EXPECT_EQ(first * second, -49110579237004466);

        Multiprecision<224> third = -549448186; // Multiprecision<7 blocks>
        Multiprecision<416> forth = -271976909; // Multiprecision<13 blocks>
        forth *= third; EXPECT_EQ(forth, 149437219283937074);
    }
    {
        Multiprecision<352> first = -219679513; // Multiprecision<11 blocks>
        Multiprecision<416> second = -854551178; // Multiprecision<13 blocks>
        EXPECT_EQ(first * second, 187727386616616314);

        Multiprecision<224> third = 666244848; // Multiprecision<7 blocks>
        Multiprecision<480> forth = 296272756; // Multiprecision<15 blocks>
        forth *= third; EXPECT_EQ(forth, 197390197287761088);
    }
    {
        Multiprecision<448> first = -268356793; // Multiprecision<14 blocks>
        Multiprecision<416> second = 842723607; // Multiprecision<13 blocks>
        EXPECT_EQ(first * second, -226150604559912351);

        Multiprecision<192> third = -616257946; // Multiprecision<6 blocks>
        Multiprecision<448> forth = 491688630; // Multiprecision<14 blocks>
        forth *= third; EXPECT_EQ(forth, -303007025195353980);
    }
    {
        Multiprecision<256> first = 456747220; // Multiprecision<8 blocks>
        Multiprecision<384> second = -110801174; // Multiprecision<12 blocks>
        EXPECT_EQ(first * second, -50608128197236280);

        Multiprecision<224> third = -396227181; // Multiprecision<7 blocks>
        Multiprecision<416> forth = -244030095; // Multiprecision<13 blocks>
        forth *= third; EXPECT_EQ(forth, 96691356621012195);
    }
    {
        Multiprecision<352> first = 623180065; // Multiprecision<11 blocks>
        Multiprecision<288> second = -2907435; // Multiprecision<9 blocks>
        EXPECT_EQ(first * second, -1811855532283275);

        Multiprecision<256> third = 541805924; // Multiprecision<8 blocks>
        Multiprecision<448> forth = 257383120; // Multiprecision<14 blocks>
        forth *= third; EXPECT_EQ(forth, 139451699153602880);
    }
    {
        Multiprecision<288> first = -84363176; // Multiprecision<9 blocks>
        Multiprecision<352> second = -347420155; // Multiprecision<11 blocks>
        EXPECT_EQ(first * second, 29309467682212280);

        Multiprecision<224> third = 289550886; // Multiprecision<7 blocks>
        Multiprecision<480> forth = -717975170; // Multiprecision<15 blocks>
        forth *= third; EXPECT_EQ(forth, -207890346599500620);
    }
}

TEST(Multiplication, Huge) { EXPECT_TRUE(false); }