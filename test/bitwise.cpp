#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Bitwise, LeftShift) {
    Multiprecision m0 = 14183932482008727;
    EXPECT_EQ(m0 << 0, 14183932482008727);
    Multiprecision m1 = -7533293377377441;
    EXPECT_EQ(m1 << 1, -15066586754754882);
    Multiprecision m2 = -12962879017404857;
    EXPECT_EQ(m2 << 2, -51851516069619428);
    Multiprecision m3 = 4933582305544806;
    EXPECT_EQ(m3 << 3, 39468658444358448);
    Multiprecision m4 = -9000593612784804;
    EXPECT_EQ(m4 << 4, -144009497804556864);
    Multiprecision m5 = -7896449064952662;
    EXPECT_EQ(m5 << 5, -252686370078485184);
    Multiprecision m6 = -9977134616321182;
    EXPECT_EQ(m6 << 6, -638536615444555648);
    Multiprecision m7 = -5188703837105184;
    EXPECT_EQ(m7 << 7, -664154091149463552);
    Multiprecision m8 = -12405341280171558;
    EXPECT_EQ(m8 << 8, -3175767367723918848);
    Multiprecision m9 = 4617283807189650;
    EXPECT_EQ(m9 << 9, 2364049309281100800);
    Multiprecision m10 = 13774933746320759;
    EXPECT_EQ(m10 << 10, -4341211917477094400);
    Multiprecision m11 = -9532907860164647;
    EXPECT_EQ(m11 << 11, -1076651223907645440);
    Multiprecision m12 = 6494511260504350;
    EXPECT_EQ(m12 << 12, 8154774049316265984);
    Multiprecision m13 = -7309855157811124;
    EXPECT_EQ(m13 << 13, -4542101231660072960);
    Multiprecision m14 = -16953264463001869;
    EXPECT_EQ(m14 << 14, -1061123856179347456);
    Multiprecision m15 = -2588503079613101;
    EXPECT_EQ(m15 << 15, 7413651455785664512);
    Multiprecision m16 = 2024262343836439;
    EXPECT_EQ(m16 << 16, 3534848449698004992);
    Multiprecision m17 = 9765228865927115;
    EXPECT_EQ(m17 << 17, 7122736828839755776);
    Multiprecision m18 = 6715255533152866;
    EXPECT_EQ(m18 << 18, 7923259480417501184);
    Multiprecision m19 = -15662963279947392;
    EXPECT_EQ(m19 << 19, -3102579316307787776);
    Multiprecision m20 = -11272421836405961;
    EXPECT_EQ(m20 << 20, 4371951716605624320);
    Multiprecision m21 = 291596602048657;
    EXPECT_EQ(m21 << 21, 2779842747129921536);
    Multiprecision m22 = -14285185479111361;
    EXPECT_EQ(m22 << 22, -1385844370074238976);
    Multiprecision m23 = 14889113361230920;
    EXPECT_EQ(m23 << 23, -3968668158788632576);
    Multiprecision m24 = -9757803458185449;
    EXPECT_EQ(m24 << 24, 6077350648024662016);
    Multiprecision m25 = 8930692502047541;
    EXPECT_EQ(m25 << 25, -3043204547590750208);
    Multiprecision m26 = 10146051943402445;
    EXPECT_EQ(m26 << 26, 2249502037119074304);
    Multiprecision m27 = 1381502026004279;
    EXPECT_EQ(m27 << 27, -4608271237167185920);
    Multiprecision m28 = -2374880268253134;
    EXPECT_EQ(m28 << 28, -1039310603954421760);
    Multiprecision m29 = -15357701832410866;
    EXPECT_EQ(m29 << 29, 916147318078570496);
    Multiprecision m30 = -15708248868342747;
    EXPECT_EQ(m30 << 30, -7815384685596377088);
    Multiprecision m31 = 1387149039262648;
    EXPECT_EQ(m31 << 31, 7412412459614470144);
    Multiprecision m32 = 3095107995689681;
    EXPECT_EQ(m32 << 32, -243226467578675200);
    Multiprecision m33 = 11068042612693158;
    EXPECT_EQ(m33 << 33, -534888990751326208);
    Multiprecision m34 = 16463563885386761;
    EXPECT_EQ(m34 << 34, -5440225049942425600);
    Multiprecision m35 = -13203286722271379;
    EXPECT_EQ(m35 << 35, 606080942720679936);
    Multiprecision m36 = -13891410158518308;
    EXPECT_EQ(m36 << 36, -9140057917649584128);
    Multiprecision m37 = 17074850743639188;
    EXPECT_EQ(m37 << 37, -6531465756117303296);
    Multiprecision m38 = -5407701702443945;
    EXPECT_EQ(m38 << 38, 4494334867516948480);
    Multiprecision m39 = -3438764334683483;
    EXPECT_EQ(m39 << 39, -5857685030616170496);
    Multiprecision m40 = 13628884299929670;
    EXPECT_EQ(m40 << 40, 7050462182462455808);
    Multiprecision m41 = -377748155790375;
    EXPECT_EQ(m41 << 41, -5181477133196722176);
    Multiprecision m42 = -17491383313456234;
    EXPECT_EQ(m42 << 42, -2121661617421680640);
    Multiprecision m43 = -7781656164555196;
    EXPECT_EQ(m43 << 43, 7003695554886631424);
    Multiprecision m44 = 1950012978625198;
    EXPECT_EQ(m44 << 44, 6893568470248587264);
    Multiprecision m45 = -14015136770581528;
    EXPECT_EQ(m45 << 45, -901564350404231168);
    Multiprecision m46 = 7130095049721998;
    EXPECT_EQ(m46 << 46, -3304656964071456768);
    Multiprecision m47 = -15611743499611488;
    EXPECT_EQ(m47 << 47, -3940649673949184000);
    Multiprecision m48 = -1794292739008361;
    EXPECT_EQ(m48 << 48, 1771884978393579520);
    Multiprecision m49 = -7205245416778920;
    EXPECT_EQ(m49 << 49, -6435643867512438784);
    Multiprecision m50 = -4095254900875778;
    EXPECT_EQ(m50 << 50, 8644659484737667072);
    Multiprecision m51 = -3270971943834592;
    EXPECT_EQ(m51 << 51, 6989586621679009792);
    Multiprecision m52 = -10770710027817230;
    EXPECT_EQ(m52 << 52, -1215971899390033920);
    Multiprecision m53 = 12596347119710265;
    EXPECT_EQ(m53 << 53, 513410357520236544);
    Multiprecision m54 = -17793673717264978;
    EXPECT_EQ(m54 << 54, 7746191359077253120);
    Multiprecision m55 = -2726345845347465;
    EXPECT_EQ(m55 << 55, -4935945191598063616);
    Multiprecision m56 = -910787265089945;
    EXPECT_EQ(m56 << 56, 7421932185906577408);
    Multiprecision m57 = -14280424552757831;
    EXPECT_EQ(m57 << 57, 8214565720323784704);
    Multiprecision m58 = 10152159557838908;
    EXPECT_EQ(m58 << 58, -1152921504606846976);
    Multiprecision m59 = -15141503220449987;
    EXPECT_EQ(m59 << 59, -1729382256910270464);
    Multiprecision m60 = 14657618016388230;
    EXPECT_EQ(m60 << 60, 6917529027641081856);
    Multiprecision m61 = -15621887041472499;
    EXPECT_EQ(m61 << 61, -6917529027641081856);
    Multiprecision m62 = 14670186568604283;
    EXPECT_EQ(m62 << 62, -4611686018427387904);
    Multiprecision m63 = -9747273255042196;
    EXPECT_EQ(m63 << 63, 0);
    Multiprecision m64 = -17018677234954975;
    EXPECT_EQ(m64 << 64, -17018677234954975);

    Multiprecision<320> r0 = -8277188590249714821;
    EXPECT_EQ(r0 << 437, -8277188590249714821);
    Multiprecision<448> r1 = -7698131438092406145;
    EXPECT_EQ(r1 << 613, -7698131438092406145);
    Multiprecision<448> r2 = 7585167919090284413;
    EXPECT_EQ(r2 << 577, 7585167919090284413);
    Multiprecision<544> r3 = -9009624143360082356;
    EXPECT_EQ(r3 << 764, -9009624143360082356);
    Multiprecision<352> r4 = 7395180038703830670;
    EXPECT_EQ(r4 << 384, 7395180038703830670);
    Multiprecision<384> r5 = 8038774720557599729;
    EXPECT_EQ(r5 << 446, 8038774720557599729);
    Multiprecision<640> r6 = 7565028275438670183;
    EXPECT_EQ(r6 << 735, 7565028275438670183);
    Multiprecision<416> r7 = -8122567541833228077;
    EXPECT_EQ(r7 << 611, -8122567541833228077);
    Multiprecision<640> r8 = -8123356254105794039;
    EXPECT_EQ(r8 << 773, -8123356254105794039);
    Multiprecision<416> r9 = -8434996685836347372;
    EXPECT_EQ(r9 << 620, -8434996685836347372);
    Multiprecision<384> r10 = 8665745177991139082;
    EXPECT_EQ(r10 << 473, 8665745177991139082);
}

TEST(Bitwise, RightShift) {
    Multiprecision m0 = 8127453382154628290;
    EXPECT_EQ(m0 >> 0, 8127453382154628290);
    Multiprecision m1 = -8144722340665737661;
    EXPECT_EQ(m1 >> 1, -4072361170332868831);
    Multiprecision m2 = 7218476478305563666;
    EXPECT_EQ(m2 >> 2, 1804619119576390916);
    Multiprecision m3 = 7750622717597323170;
    EXPECT_EQ(m3 >> 3, 968827839699665396);
    Multiprecision m4 = 8139984050252575721;
    EXPECT_EQ(m4 >> 4, 508749003140785982);
    Multiprecision m5 = 9073606883707592999;
    EXPECT_EQ(m5 >> 5, 283550215115862281);
    Multiprecision m6 = 8616984612426039063;
    EXPECT_EQ(m6 >> 6, 134640384569156860);
    Multiprecision m7 = 8267705293996351586;
    EXPECT_EQ(m7 >> 7, 64591447609346496);
    Multiprecision m8 = -7220363035220360305;
    EXPECT_EQ(m8 >> 8, -28204543106329533);
    Multiprecision m9 = 9205584151102696752;
    EXPECT_EQ(m9 >> 9, 17979656545122454);
    Multiprecision m10 = -7968847425047808558;
    EXPECT_EQ(m10 >> 10, -7782077563523251);
    Multiprecision m11 = 8723675218965904029;
    EXPECT_EQ(m11 >> 11, 4259607040510695);
    Multiprecision m12 = 8946089651846449589;
    EXPECT_EQ(m12 >> 12, 2184103918907824);
    Multiprecision m13 = -8012503653905290945;
    EXPECT_EQ(m13 >> 13, -978088824939611);
    Multiprecision m14 = -8978507834985276594;
    EXPECT_EQ(m14 >> 14, -548004628600176);
    Multiprecision m15 = 7207610213605958824;
    EXPECT_EQ(m15 >> 15, 219958807788267);
    Multiprecision m16 = 7155423308404327926;
    EXPECT_EQ(m16 >> 16, 109183094915837);
    Multiprecision m17 = -8058775352000943671;
    EXPECT_EQ(m17 >> 17, -61483576599129);
    Multiprecision m18 = 7426601645777368347;
    EXPECT_EQ(m18 >> 18, 28330236991033);
    Multiprecision m19 = 8847151517221453270;
    EXPECT_EQ(m19 >> 19, 16874602350657);
    Multiprecision m20 = -7247278646723952484;
    EXPECT_EQ(m20 >> 20, -6911543509221);
    Multiprecision m21 = -7518196135879204065;
    EXPECT_EQ(m21 >> 21, -3584955280247);
    Multiprecision m22 = -9041534879021751521;
    EXPECT_EQ(m22 >> 22, -2155669898754);
    Multiprecision m23 = -7217195916222406198;
    EXPECT_EQ(m23 >> 23, -860356797722);
    Multiprecision m24 = -7651626799614001937;
    EXPECT_EQ(m24 >> 24, -456072497345);
    Multiprecision m25 = 7444671774464937498;
    EXPECT_EQ(m25 >> 25, 221868508293);
    Multiprecision m26 = 9163356457293897033;
    EXPECT_EQ(m26 >> 26, 136544651646);
    Multiprecision m27 = -8685282192677654389;
    EXPECT_EQ(m27 >> 27, -64710394984);
    Multiprecision m28 = 7952855855589160876;
    EXPECT_EQ(m28 >> 28, 29626696763);
    Multiprecision m29 = -7591404361865154537;
    EXPECT_EQ(m29 >> 29, -14140092511);
    Multiprecision m30 = 7842048178821644141;
    EXPECT_EQ(m30 >> 30, 7303476500);
    Multiprecision m31 = -7827287102773679631;
    EXPECT_EQ(m31 >> 31, -3644864589);
    Multiprecision m32 = -7560054105921234268;
    EXPECT_EQ(m32 >> 32, -1760212264);
    Multiprecision m33 = 7800426419228256144;
    EXPECT_EQ(m33 >> 33, 908089151);
    Multiprecision m34 = -8653696694987692297;
    EXPECT_EQ(m34 >> 34, -503711443);
    Multiprecision m35 = 7303917434842851937;
    EXPECT_EQ(m35 >> 35, 212571974);
    Multiprecision m36 = 8037396469532149207;
    EXPECT_EQ(m36 >> 36, 116959512);
    Multiprecision m37 = -8091551554983113795;
    EXPECT_EQ(m37 >> 37, -58873787);
    Multiprecision m38 = -7240468296232555615;
    EXPECT_EQ(m38 >> 38, -26340671);
    Multiprecision m39 = -8567757256730308669;
    EXPECT_EQ(m39 >> 39, -15584660);
    Multiprecision m40 = 7434515926690727299;
    EXPECT_EQ(m40 >> 40, 6761652);
    Multiprecision m41 = 8836352719667454881;
    EXPECT_EQ(m41 >> 41, 4018307);
    Multiprecision m42 = 8928056884127673923;
    EXPECT_EQ(m42 >> 42, 2030005);
    Multiprecision m43 = -9137746494950078834;
    EXPECT_EQ(m43 >> 43, -1038842);
    Multiprecision m44 = -8190632727576959362;
    EXPECT_EQ(m44 >> 44, -465584);
    Multiprecision m45 = 8066953901535948115;
    EXPECT_EQ(m45 >> 45, 229276);
    Multiprecision m46 = -8398462365818992550;
    EXPECT_EQ(m46 >> 46, -119350);
    Multiprecision m47 = -7026716507353674952;
    EXPECT_EQ(m47 >> 47, -49928);
    Multiprecision m48 = 9166632574670105381;
    EXPECT_EQ(m48 >> 48, 32566);
    Multiprecision m49 = 8817658003157363952;
    EXPECT_EQ(m49 >> 49, 15663);
    Multiprecision m50 = 7690071574521976350;
    EXPECT_EQ(m50 >> 50, 6830);
    Multiprecision m51 = -9083406980093258649;
    EXPECT_EQ(m51 >> 51, -4034);
    Multiprecision m52 = 7748140986306674816;
    EXPECT_EQ(m52 >> 52, 1720);
    Multiprecision m53 = 7183299866343108040;
    EXPECT_EQ(m53 >> 53, 797);
    Multiprecision m54 = -7951030804115409903;
    EXPECT_EQ(m54 >> 54, -442);
    Multiprecision m55 = 8913285552037539786;
    EXPECT_EQ(m55 >> 55, 247);
    Multiprecision m56 = -9153834404704512943;
    EXPECT_EQ(m56 >> 56, -128);
    Multiprecision m57 = 7595616646652081296;
    EXPECT_EQ(m57 >> 57, 52);
    Multiprecision m58 = -9138106715904641691;
    EXPECT_EQ(m58 >> 58, -32);
    Multiprecision m59 = 7164400798038049350;
    EXPECT_EQ(m59 >> 59, 12);
    Multiprecision m60 = -8182214843427186538;
    EXPECT_EQ(m60 >> 60, -8);
    Multiprecision m61 = 7076500928276120366;
    EXPECT_EQ(m61 >> 61, 3);
    Multiprecision m62 = -7528476277073922331;
    EXPECT_EQ(m62 >> 62, -2);
    Multiprecision m63 = -8364867615549303686;
    EXPECT_EQ(m63 >> 63, -1);
    Multiprecision m64 = 8786527288323906543;
    EXPECT_EQ(m64 >> 64, 8786527288323906543);

    Multiprecision<384> r0 = -9181196393426359111;
    EXPECT_EQ(r0 >> 546, -9181196393426359111);
    Multiprecision<448> r1 = 9145862889354412168;
    EXPECT_EQ(r1 >> 459, 9145862889354412168);
    Multiprecision<480> r2 = -7163388349126125485;
    EXPECT_EQ(r2 >> 513, -7163388349126125485);
    Multiprecision<352> r3 = -8391867237189050543;
    EXPECT_EQ(r3 >> 389, -8391867237189050543);
    Multiprecision<640> r4 = -8925994839841689779;
    EXPECT_EQ(r4 >> 903, -8925994839841689779);
    Multiprecision<640> r5 = 9066700264884018589;
    EXPECT_EQ(r5 >> 837, 9066700264884018589);
    Multiprecision<384> r6 = 7866074859540430653;
    EXPECT_EQ(r6 >> 501, 7866074859540430653);
    Multiprecision<640> r7 = 8032775485025027428;
    EXPECT_EQ(r7 >> 912, 8032775485025027428);
    Multiprecision<416> r8 = 9109723032170972228;
    EXPECT_EQ(r8 >> 441, 9109723032170972228);
    Multiprecision<352> r9 = 7468813884241587944;
    EXPECT_EQ(r9 >> 381, 7468813884241587944);
    Multiprecision<384> r10 = -7338864925845722961;
    EXPECT_EQ(r10 >> 486, -7338864925845722961);
}