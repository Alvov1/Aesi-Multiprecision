#include <gtest/gtest.h>
#include "../../Multiprecision.h"

TEST(Subtraction, Zero) {
    Multiprecision zero = 0;
    Multiprecision m0 = 14377898;
    EXPECT_EQ(m0 - 0, 14377898);
    EXPECT_EQ(0 - m0, -14377898);
    EXPECT_EQ(m0 + 0, 14377898);
    EXPECT_EQ(m0 + zero, 14377898);
    EXPECT_EQ(m0 - zero, 14377898);
    EXPECT_EQ(m0 + -zero, 14377898);
    EXPECT_EQ(m0 + +zero, 14377898);
    EXPECT_EQ(m0 - +zero, 14377898);
    EXPECT_EQ(m0 - -zero, 14377898);
    EXPECT_EQ(zero - m0, -14377898);
    EXPECT_EQ(+zero - m0, -14377898);
    EXPECT_EQ(-zero - m0, -14377898);
    m0 -= 0; EXPECT_EQ(m0, 14377898);
    m0 -= zero; EXPECT_EQ(m0, 14377898);
    m0 -= -zero; EXPECT_EQ(m0, 14377898);
    m0 -= +zero; EXPECT_EQ(m0, 14377898);

    Multiprecision m1 = 42824647;
    EXPECT_EQ(m1 - 0, 42824647);
    EXPECT_EQ(0 - m1, -42824647);
    EXPECT_EQ(m1 + 0, 42824647);
    EXPECT_EQ(m1 + zero, 42824647);
    EXPECT_EQ(m1 - zero, 42824647);
    EXPECT_EQ(m1 + -zero, 42824647);
    EXPECT_EQ(m1 + +zero, 42824647);
    EXPECT_EQ(m1 - +zero, 42824647);
    EXPECT_EQ(m1 - -zero, 42824647);
    EXPECT_EQ(zero - m1, -42824647);
    EXPECT_EQ(+zero - m1, -42824647);
    EXPECT_EQ(-zero - m1, -42824647);
    m1 -= 0; EXPECT_EQ(m1, 42824647);
    m1 -= zero; EXPECT_EQ(m1, 42824647);
    m1 -= -zero; EXPECT_EQ(m1, 42824647);
    m1 -= +zero; EXPECT_EQ(m1, 42824647);

    Multiprecision m2 = 56407773;
    EXPECT_EQ(m2 - 0, 56407773);
    EXPECT_EQ(0 - m2, -56407773);
    EXPECT_EQ(m2 + 0, 56407773);
    EXPECT_EQ(m2 + zero, 56407773);
    EXPECT_EQ(m2 - zero, 56407773);
    EXPECT_EQ(m2 + -zero, 56407773);
    EXPECT_EQ(m2 + +zero, 56407773);
    EXPECT_EQ(m2 - +zero, 56407773);
    EXPECT_EQ(m2 - -zero, 56407773);
    EXPECT_EQ(zero - m2, -56407773);
    EXPECT_EQ(+zero - m2, -56407773);
    EXPECT_EQ(-zero - m2, -56407773);
    m2 -= 0; EXPECT_EQ(m2, 56407773);
    m2 -= zero; EXPECT_EQ(m2, 56407773);
    m2 -= -zero; EXPECT_EQ(m2, 56407773);
    m2 -= +zero; EXPECT_EQ(m2, 56407773);
}

TEST(Subtraction, SmallPositive) {
    Multiprecision small1 = -8492, small2 = 4243, small3 = -678, small4 = 2323;
    EXPECT_EQ(small2 - small4, 1920);
    EXPECT_EQ(small3 - small1, 7814);
    EXPECT_EQ(small2 - small1, 12735);
    EXPECT_EQ(small2 - small3, 4921);
}

TEST(Subtraction, SmallNegative) {
    Multiprecision small1 = -8492, small2 = 4243, small3 = -678, small4 = 2323;
    EXPECT_EQ(small1 - small2, -12735);
    EXPECT_EQ(small3 - small2, -4921);
    EXPECT_EQ(small4 - small2, -1920);
    EXPECT_EQ(small1 - small3, -7814);
}

TEST(Subtraction, Decrement) {
    Multiprecision m0 = -33924914;
    --m0; m0--; m0--; --m0; m0--; m0--;
    EXPECT_EQ(m0, -33924920);
    Multiprecision t0 = m0--, u0 = --m0;
    EXPECT_EQ(t0, -33924920); EXPECT_EQ(u0, -33924922); EXPECT_EQ(m0, -33924922);

    Multiprecision m1 = 28310193;
    --m1; --m1; --m1; m1--; --m1; m1--; --m1; m1--; --m1; --m1; --m1;
    EXPECT_EQ(m1, 28310182);
    Multiprecision t1 = m1--, u1 = --m1;
    EXPECT_EQ(t1, 28310182); EXPECT_EQ(u1, 28310180); EXPECT_EQ(m1, 28310180);

    Multiprecision m2 = 57075809;
    --m2; m2--; m2--; m2--; m2--; --m2; --m2; --m2; --m2;
    EXPECT_EQ(m2, 57075800);
    Multiprecision t2 = m2--, u2 = --m2;
    EXPECT_EQ(t2, 57075800); EXPECT_EQ(u2, 57075798); EXPECT_EQ(m2, 57075798);

    Multiprecision m3 = -2900339;
    m3--; --m3; m3--; m3--; m3--; m3--; --m3; m3--; m3--; m3--; m3--; --m3; --m3; m3--;
    EXPECT_EQ(m3, -2900353);
    Multiprecision t3 = m3--, u3 = --m3;
    EXPECT_EQ(t3, -2900353); EXPECT_EQ(u3, -2900355); EXPECT_EQ(m3, -2900355);

    Multiprecision m4 = -12687358;
    --m4; --m4; --m4; --m4; --m4; m4--; m4--; --m4; --m4; m4--; m4--; m4--; m4--;
    EXPECT_EQ(m4, -12687371);
    Multiprecision t4 = m4--, u4 = --m4;
    EXPECT_EQ(t4, -12687371); EXPECT_EQ(u4, -12687373); EXPECT_EQ(m4, -12687373);
}

TEST(Subtraction, MixedSubtraction) {
    Multiprecision m0 = -78795448765594211, m1 = 48701305384651798;
    EXPECT_EQ(m0 - m1, -127496754150246009);
    Multiprecision m2 = 67574064564830699, m3 = -91803406528648461;
    EXPECT_EQ(m2 - m3, 159377471093479160);
    Multiprecision m4 = -96016808488224510, m5 = 250370865835895;
    EXPECT_EQ(m4 - m5, -96267179354060405);
    Multiprecision m6 = -41074518047002314, m7 = 95579003191734993;
    EXPECT_EQ(m6 - m7, -136653521238737307);
    Multiprecision m8 = -29705744161350641, m9 = 80392893348344206;
    EXPECT_EQ(m8 - m9, -110098637509694847);
    Multiprecision m10 = 46806496601606081, m11 = -91563580387805963;
    EXPECT_EQ(m10 - m11, 138370076989412044);
    Multiprecision m12 = -36224949170869920, m13 = 94125420376985567;
    EXPECT_EQ(m12 - m13, -130350369547855487);
    Multiprecision m14 = 56826716379651150, m15 = -3945619004625386;
    EXPECT_EQ(m14 - m15, 60772335384276536);
    Multiprecision m16 = 70075148666243867, m17 = 41028685512594215;
    EXPECT_EQ(m16 - m17, 29046463153649652);
    Multiprecision m18 = 9684075024490575, m19 = -94275562315467831;
    EXPECT_EQ(m18 - m19, 103959637339958406);
    Multiprecision m20 = -19394083460764198, m21 = 4598882134566940;
    EXPECT_EQ(m20 - m21, -23992965595331138);
    Multiprecision m22 = -59638329813822567, m23 = 39620622567810820;
    EXPECT_EQ(m22 - m23, -99258952381633387);
    Multiprecision m24 = 10481374612533748, m25 = -45690246001184389;
    EXPECT_EQ(m24 - m25, 56171620613718137);
    Multiprecision m26 = -39861901704867459, m27 = -71363004984618855;
    EXPECT_EQ(m26 - m27, 31501103279751396);
    Multiprecision m28 = 57092377547587484, m29 = 47047741340310546;
    EXPECT_EQ(m28 - m29, 10044636207276938);
    Multiprecision m30 = -60473701642492004, m31 = 36953318815595088;
    EXPECT_EQ(m30 - m31, -97427020458087092);
    Multiprecision m32 = -78684568025867499, m33 = 39474612379303263;
    EXPECT_EQ(m32 - m33, -118159180405170762);
    Multiprecision m34 = -59708830071153936, m35 = -24987753064694161;
    EXPECT_EQ(m34 - m35, -34721077006459775);
    Multiprecision m36 = -61431982293008665, m37 = -93223158394351966;
    EXPECT_EQ(m36 - m37, 31791176101343301);
    Multiprecision m38 = -77616429233373210, m39 = -56564296394666424;
    EXPECT_EQ(m38 - m39, -21052132838706786);
    Multiprecision m40 = -62011231426006983, m41 = -32675741172931378;
    EXPECT_EQ(m40 - m41, -29335490253075605);
    Multiprecision m42 = -36932046954643746, m43 = -6581413634276615;
    EXPECT_EQ(m42 - m43, -30350633320367131);
    Multiprecision m44 = -63308462008430376, m45 = -30568052971566789;
    EXPECT_EQ(m44 - m45, -32740409036863587);
    Multiprecision m46 = -3761885636924116, m47 = -51334343669010520;
    EXPECT_EQ(m46 - m47, 47572458032086404);
    Multiprecision m48 = 29643032223799847, m49 = -8815829513288304;
    EXPECT_EQ(m48 - m49, 38458861737088151);
    Multiprecision m50 = -63790459687888788, m51 = 56482071741526173;
    EXPECT_EQ(m50 - m51, -120272531429414961);
    Multiprecision m52 = 44052929832619500, m53 = 94503788809467804;
    EXPECT_EQ(m52 - m53, -50450858976848304);
    Multiprecision m54 = 75262871827916165, m55 = 75116929302625443;
    EXPECT_EQ(m54 - m55, 145942525290722);
    Multiprecision m56 = 42012563671870421, m57 = -39551471801647037;
    EXPECT_EQ(m56 - m57, 81564035473517458);
    Multiprecision m58 = -58094893884533034, m59 = -89603604140659081;
    EXPECT_EQ(m58 - m59, 31508710256126047);
    Multiprecision m60 = -15491398282947126, m61 = 75124171238430754;
    EXPECT_EQ(m60 - m61, -90615569521377880);
    Multiprecision m62 = -15279473212308778, m63 = -25214196246095585;
    EXPECT_EQ(m62 - m63, 9934723033786807);
    Multiprecision m64 = -15459700611573614, m65 = 61544035971577530;
    EXPECT_EQ(m64 - m65, -77003736583151144);
    Multiprecision m66 = -88407578525560434, m67 = 70572399006127154;
    EXPECT_EQ(m66 - m67, -158979977531687588);
    Multiprecision m68 = -76673284026197188, m69 = -42838723021411352;
    EXPECT_EQ(m68 - m69, -33834561004785836);
    Multiprecision m70 = -77534335981039599, m71 = -2575361264407743;
    EXPECT_EQ(m70 - m71, -74958974716631856);
    Multiprecision m72 = 14336673228002761, m73 = -63399735148346295;
    EXPECT_EQ(m72 - m73, 77736408376349056);
    Multiprecision m74 = -45220785989975054, m75 = -88657223135591669;
    EXPECT_EQ(m74 - m75, 43436437145616615);
    Multiprecision m76 = -3750032932034014, m77 = -3079274694736935;
    EXPECT_EQ(m76 - m77, -670758237297079);
    Multiprecision m78 = -57879783190606075, m79 = 30582080660739954;
    EXPECT_EQ(m78 - m79, -88461863851346029);
}

TEST(Subtraction, MixedSubtractionAssignment) {
    Multiprecision m0 = 76497282207537392, m1 = -81483767496064665;
    m0 -= m1; EXPECT_EQ(m0, 157981049703602057);
    Multiprecision m2 = 52766273087594876, m3 = -45464959271515713;
    m2 -= m3; EXPECT_EQ(m2, 98231232359110589);
    Multiprecision m4 = -39866035793358885, m5 = -74561994540740578;
    m4 -= m5; EXPECT_EQ(m4, 34695958747381693);
    Multiprecision m6 = -32310239597891199, m7 = 38858227484763400;
    m6 -= m7; EXPECT_EQ(m6, -71168467082654599);
    Multiprecision m8 = 83600373085209769, m9 = -4676747027315379;
    m8 -= m9; EXPECT_EQ(m8, 88277120112525148);
    Multiprecision m10 = 26800952209504502, m11 = -81901349566193950;
    m10 -= m11; EXPECT_EQ(m10, 108702301775698452);
    Multiprecision m12 = -22980648057789418, m13 = -62890011667727509;
    m12 -= m13; EXPECT_EQ(m12, 39909363609938091);
    Multiprecision m14 = -30243185699710800, m15 = 85196747975255772;
    m14 -= m15; EXPECT_EQ(m14, -115439933674966572);
    Multiprecision m16 = 88575636175295575, m17 = 5058788936387072;
    m16 -= m17; EXPECT_EQ(m16, 83516847238908503);
    Multiprecision m18 = -8627149227965893, m19 = -51145930004297782;
    m18 -= m19; EXPECT_EQ(m18, 42518780776331889);
    Multiprecision m20 = -61641748360877573, m21 = 1347982179977643;
    m20 -= m21; EXPECT_EQ(m20, -62989730540855216);
    Multiprecision m22 = -76812276267570764, m23 = -66388565315430388;
    m22 -= m23; EXPECT_EQ(m22, -10423710952140376);
    Multiprecision m24 = -22413217680403760, m25 = 20776822720647242;
    m24 -= m25; EXPECT_EQ(m24, -43190040401051002);
    Multiprecision m26 = 60501229183601977, m27 = -97208132801489513;
    m26 -= m27; EXPECT_EQ(m26, 157709361985091490);
    Multiprecision m28 = -90717506923312883, m29 = 80581288039243653;
    m28 -= m29; EXPECT_EQ(m28, -171298794962556536);
    Multiprecision m30 = -1434671774297789, m31 = -48788207733204515;
    m30 -= m31; EXPECT_EQ(m30, 47353535958906726);
    Multiprecision m32 = 98636409838290088, m33 = -24858988765114797;
    m32 -= m33; EXPECT_EQ(m32, 123495398603404885);
    Multiprecision m34 = 32698890626094304, m35 = -95452487538418616;
    m34 -= m35; EXPECT_EQ(m34, 128151378164512920);
    Multiprecision m36 = 54592571715348901, m37 = 27499032134617746;
    m36 -= m37; EXPECT_EQ(m36, 27093539580731155);
    Multiprecision m38 = -34229390668287600, m39 = 70527679356389759;
    m38 -= m39; EXPECT_EQ(m38, -104757070024677359);
    Multiprecision m40 = 38082671719287318, m41 = 90080041129782021;
    m40 -= m41; EXPECT_EQ(m40, -51997369410494703);
    Multiprecision m42 = -26471635935191754, m43 = -11535551530392665;
    m42 -= m43; EXPECT_EQ(m42, -14936084404799089);
    Multiprecision m44 = 45091999474176387, m45 = -33228319472241331;
    m44 -= m45; EXPECT_EQ(m44, 78320318946417718);
    Multiprecision m46 = -91842884207028021, m47 = 51904091210877915;
    m46 -= m47; EXPECT_EQ(m46, -143746975417905936);
    Multiprecision m48 = -5357775061339009, m49 = -25273398031638359;
    m48 -= m49; EXPECT_EQ(m48, 19915622970299350);
    Multiprecision m50 = -14905275244784635, m51 = -94021456037974671;
    m50 -= m51; EXPECT_EQ(m50, 79116180793190036);
    Multiprecision m52 = -13576279815222827, m53 = -42123086873941691;
    m52 -= m53; EXPECT_EQ(m52, 28546807058718864);
    Multiprecision m54 = -52554401815218966, m55 = 99184114388707534;
    m54 -= m55; EXPECT_EQ(m54, -151738516203926500);
    Multiprecision m56 = -3883942059250462, m57 = -34609928360199472;
    m56 -= m57; EXPECT_EQ(m56, 30725986300949010);
    Multiprecision m58 = 14175824081074887, m59 = 1435953275620283;
    m58 -= m59; EXPECT_EQ(m58, 12739870805454604);
    Multiprecision m60 = -60149276957042841, m61 = -76721648621066592;
    m60 -= m61; EXPECT_EQ(m60, 16572371664023751);
    Multiprecision m62 = 61326832747483284, m63 = 46379655301271614;
    m62 -= m63; EXPECT_EQ(m62, 14947177446211670);
    Multiprecision m64 = 21854645686000102, m65 = 45002361665513639;
    m64 -= m65; EXPECT_EQ(m64, -23147715979513537);
    Multiprecision m66 = 33091273075346782, m67 = 321895524121470;
    m66 -= m67; EXPECT_EQ(m66, 32769377551225312);
    Multiprecision m68 = -51174023236882378, m69 = 66903075876437798;
    m68 -= m69; EXPECT_EQ(m68, -118077099113320176);
    Multiprecision m70 = -24801419283802435, m71 = -14207322058971261;
    m70 -= m71; EXPECT_EQ(m70, -10594097224831174);
    Multiprecision m72 = -92599999116779629, m73 = 90215441005269707;
    m72 -= m73; EXPECT_EQ(m72, -182815440122049336);
    Multiprecision m74 = 73877826268355011, m75 = 65420173307826179;
    m74 -= m75; EXPECT_EQ(m74, 8457652960528832);
    Multiprecision m76 = 19586642362468978, m77 = -65885160582015983;
    m76 -= m77; EXPECT_EQ(m76, 85471802944484961);
    Multiprecision m78 = 7573797785526040, m79 = -17722295466379071;
    m78 -= m79; EXPECT_EQ(m78, 25296093251905111);

}

TEST(Subtraction, DifferentPrecision) {
    {
        Multiprecision<448> first = 62809965279956307; // Multiprecision<14 blocks>
        Multiprecision<448> second = 39080556778759957; // Multiprecision<14 blocks>
        EXPECT_EQ(first - second, 23729408501196350);

        Multiprecision<192> third = 45398660613732434; // Multiprecision<6 blocks>
        Multiprecision<352> forth = -53004642661804844; // Multiprecision<11 blocks>
        forth -= third; EXPECT_EQ(forth, -98403303275537278);
    }
    {
        Multiprecision<320> first = 18374689440060232; // Multiprecision<10 blocks>
        Multiprecision<288> second = -53557241983505081; // Multiprecision<9 blocks>
        EXPECT_EQ(first - second, 71931931423565313);

        Multiprecision<256> third = 71440960238152095; // Multiprecision<8 blocks>
        Multiprecision<448> forth = 27627026332533578; // Multiprecision<14 blocks>
        forth -= third; EXPECT_EQ(forth, -43813933905618517);
    }
    {
        Multiprecision<320> first = -58860382348449740; // Multiprecision<10 blocks>
        Multiprecision<448> second = -51106271527231455; // Multiprecision<14 blocks>
        EXPECT_EQ(first - second, -7754110821218285);

        Multiprecision<320> third = -26435995134335414; // Multiprecision<10 blocks>
        Multiprecision<416> forth = 6966067345525782; // Multiprecision<13 blocks>
        forth -= third; EXPECT_EQ(forth, 33402062479861196);
    }
    {
        Multiprecision<224> first = 26949994938035307; // Multiprecision<7 blocks>
        Multiprecision<320> second = -57345460147331841; // Multiprecision<10 blocks>
        EXPECT_EQ(first - second, 84295455085367148);

        Multiprecision<320> third = 40115210446714256; // Multiprecision<10 blocks>
        Multiprecision<416> forth = 19929757054766530; // Multiprecision<13 blocks>
        forth -= third; EXPECT_EQ(forth, -20185453391947726);
    }
    {
        Multiprecision<480> first = 43936342115046018; // Multiprecision<15 blocks>
        Multiprecision<480> second = -10279600786446307; // Multiprecision<15 blocks>
        EXPECT_EQ(first - second, 54215942901492325);

        Multiprecision<256> third = -22316375566079840; // Multiprecision<8 blocks>
        Multiprecision<480> forth = 76005949847059604; // Multiprecision<15 blocks>
        forth -= third; EXPECT_EQ(forth, 98322325413139444);
    }
    {
        Multiprecision<192> first = -93657206214492841; // Multiprecision<6 blocks>
        Multiprecision<256> second = -71486982985505092; // Multiprecision<8 blocks>
        EXPECT_EQ(first - second, -22170223228987749);

        Multiprecision<320> third = -66846726585288343; // Multiprecision<10 blocks>
        Multiprecision<416> forth = 94957726168621779; // Multiprecision<13 blocks>
        forth -= third; EXPECT_EQ(forth, 161804452753910122);
    }
    {
        Multiprecision<192> first = -62005162619173997; // Multiprecision<6 blocks>
        Multiprecision<224> second = 10671564912426240; // Multiprecision<7 blocks>
        EXPECT_EQ(first - second, -72676727531600237);

        Multiprecision<224> third = -30028516325007256; // Multiprecision<7 blocks>
        Multiprecision<416> forth = -38572290747742216; // Multiprecision<13 blocks>
        forth -= third; EXPECT_EQ(forth, -8543774422734960);
    }
    {
        Multiprecision<320> first = -58940891305080103; // Multiprecision<10 blocks>
        Multiprecision<320> second = -57959309993606272; // Multiprecision<10 blocks>
        EXPECT_EQ(first - second, -981581311473831);

        Multiprecision<224> third = 30782639550732749; // Multiprecision<7 blocks>
        Multiprecision<416> forth = -14578762817130093; // Multiprecision<13 blocks>
        forth -= third; EXPECT_EQ(forth, -45361402367862842);
    }
    {
        Multiprecision<480> first = 51699864098074373; // Multiprecision<15 blocks>
        Multiprecision<448> second = -48065639687773956; // Multiprecision<14 blocks>
        EXPECT_EQ(first - second, 99765503785848329);

        Multiprecision<320> third = 10097273671316052; // Multiprecision<10 blocks>
        Multiprecision<384> forth = 90189328318453908; // Multiprecision<12 blocks>
        forth -= third; EXPECT_EQ(forth, 80092054647137856);
    }
    {
        Multiprecision<416> first = -10287965013816677; // Multiprecision<13 blocks>
        Multiprecision<448> second = -86849365490800050; // Multiprecision<14 blocks>
        EXPECT_EQ(first - second, 76561400476983373);

        Multiprecision<288> third = -92327403515448960; // Multiprecision<9 blocks>
        Multiprecision<480> forth = -33125917700155649; // Multiprecision<15 blocks>
        forth -= third; EXPECT_EQ(forth, 59201485815293311);
    }
}

TEST(Subtraction, Huge) { EXPECT_TRUE(false); }