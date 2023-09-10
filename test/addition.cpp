#include <gtest/gtest.h>
#include "../Multiprecision.h"

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
        Multiprecision<8> first = -49650840693364918;
        Multiprecision<9> second = -87362490924180900;
        EXPECT_EQ(first + second, -137013331617545818);

        Multiprecision<7> third = -38902231494252614;
        Multiprecision<13> forth = 48627023061381137;
        forth += third; EXPECT_EQ(forth, 9724791567128523);
    }
    {
        Multiprecision<7> first = 29631140543823370;
        Multiprecision<9> second = 47234036361872764;
        EXPECT_EQ(first + second, 76865176905696134);

        Multiprecision<8> third = -54176090545093679;
        Multiprecision<13> forth = -53270350607178462;
        forth += third; EXPECT_EQ(forth, -107446441152272141);
    }
    {
        Multiprecision<11> first = -56865373832774749;
        Multiprecision<14> second = -67510568228469125;
        EXPECT_EQ(first + second, -124375942061243874);

        Multiprecision<6> third = -85466160995345838;
        Multiprecision<13> forth = 96161369383460700;
        forth += third; EXPECT_EQ(forth, 10695208388114862);
    }
    {
        Multiprecision<12> first = -70290964716006492;
        Multiprecision<13> second = -57526970498717665;
        EXPECT_EQ(first + second, -127817935214724157);

        Multiprecision<7> third = 37992284657659695;
        Multiprecision<11> forth = -62588589946074542;
        forth += third; EXPECT_EQ(forth, -24596305288414847);
    }
    {
        Multiprecision<12> first = -25694805674181358;
        Multiprecision<10> second = -38748713653933972;
        EXPECT_EQ(first + second, -64443519328115330);

        Multiprecision<8> third = 84553652870886634;
        Multiprecision<11> forth = 76865334394841545;
        forth += third; EXPECT_EQ(forth, 161418987265728179);
    }
    {
        Multiprecision<6> first = -23090573497749045;
        Multiprecision<13> second = 34016031597779492;
        EXPECT_EQ(first + second, 10925458100030447);

        Multiprecision<6> third = 51943436946341375;
        Multiprecision<13> forth = -31004408006990685;
        forth += third; EXPECT_EQ(forth, 20939028939350690);
    }
    {
        Multiprecision<11> first = 29696117782117406;
        Multiprecision<13> second = -67930593148625732;
        EXPECT_EQ(first + second, -38234475366508326);

        Multiprecision<9> third = -90369034361091933;
        Multiprecision<12> forth = 94996113224887252;
        forth += third; EXPECT_EQ(forth, 4627078863795319);
    }
    {
        Multiprecision<12> first = -79093307512948808;
        Multiprecision<10> second = -82409636200625616;
        EXPECT_EQ(first + second, -161502943713574424);

        Multiprecision<8> third = 28036001795911104;
        Multiprecision<15> forth = 76319358320902920;
        forth += third; EXPECT_EQ(forth, 104355360116814024);
    }
    {
        Multiprecision<15> first = 63382449378511788;
        Multiprecision<11> second = 53422101447704395;
        EXPECT_EQ(first + second, 116804550826216183);

        Multiprecision<10> third = 22251008569120548;
        Multiprecision<13> forth = 85667186737817198;
        forth += third; EXPECT_EQ(forth, 107918195306937746);
    }
    {
        Multiprecision<11> first = 44779626386448550;
        Multiprecision<11> second = 43872286989081712;
        EXPECT_EQ(first + second, 88651913375530262);

        Multiprecision<7> third = 98220469236206120;
        Multiprecision<15> forth = 98673809166167706;
        forth += third; EXPECT_EQ(forth, 196894278402373826);
    }
}

TEST(Addition, Huge) {
    Multiprecision huge = "8683317618811886495518194401279999999", negativeHuge = "-8683317618811886495518194401279999999";
    EXPECT_EQ(huge + negativeHuge, 0);
    EXPECT_EQ(huge + huge + huge, "26049952856435659486554583203839999997");

    Multiprecision huge2 = "26049952856435659486554583203839999997";
    huge += huge2;
    EXPECT_EQ(huge, "34733270475247545982072777605119999996");

    Multiprecision huge3 = "-489133282872437279";
    huge2 += huge3;
    EXPECT_EQ(huge2, "26049952856435659486065449920967562718");
}