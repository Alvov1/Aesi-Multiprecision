#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Bitwise, LeftShift) {
    Multiprecision<128> t = 14183932482008727;
    EXPECT_EQ(t << 48, "3992422065038923586049945894912");

    Multiprecision<128> t2 = -51851516069619428;
    EXPECT_EQ(t2 << 40, "-57011344836360679021114032128");

    Multiprecision<128> t3 = 664154091149463552;
    EXPECT_EQ(t3 << 32, "2852520100991549003783995392");
    EXPECT_EQ(t3 << 64, "12251480544941320143633640456854700032");

    Multiprecision m0 = "7611378567734697137.";
    EXPECT_EQ(m0 << 0, "7611378567734697137.");
    Multiprecision m1 = "-14850715800956435.";
    EXPECT_EQ(m1 << 1, "-29701431601912870.");
    Multiprecision m2 = "3028070178233437411.";
    EXPECT_EQ(m2 << 2, "12112280712933749644.");
    Multiprecision m3 = "9209955177909808159.";
    EXPECT_EQ(m3 << 3, "73679641423278465272.");
    Multiprecision m4 = "7165047339148406389.";
    EXPECT_EQ(m4 << 4, "114640757426374502224.");
    Multiprecision m5 = "-6626468019145325155.";
    EXPECT_EQ(m5 << 5, "-212046976612650404960.");
    Multiprecision m6 = "-7243061664548612285.";
    EXPECT_EQ(m6 << 6, "-463555946531111186240.");
    Multiprecision m7 = "-4116700124356305042.";
    EXPECT_EQ(m7 << 7, "-526937615917607045376.");
    Multiprecision m8 = "6549960156157611707.";
    EXPECT_EQ(m8 << 8, "1676789799976348596992.");
    Multiprecision m9 = "-2220785136167096747.";
    EXPECT_EQ(m9 << 9, "-1137041989717553534464.");
    Multiprecision m10 = "-7483545574330362604.";
    EXPECT_EQ(m10 << 10, "-7663150668114291306496.");
    Multiprecision m11 = "5261993495534712122.";
    EXPECT_EQ(m11 << 11, "10776562678855090425856.");
    Multiprecision m12 = "4481097010014354026.";
    EXPECT_EQ(m12 << 12, "18354573353018794090496.");
    Multiprecision m13 = "6106312505204834006.";
    EXPECT_EQ(m13 << 13, "50022912042638000177152.");
    Multiprecision m14 = "5040797595058699798.";
    EXPECT_EQ(m14 << 14, "82588427797441737490432.");
    Multiprecision m15 = "2583471234648375770.";
    EXPECT_EQ(m15 << 15, "84655185416957977231360.");
    Multiprecision m16 = "-6788631554508561346.";
    EXPECT_EQ(m16 << 16, "-444899757556273076371456.");
    Multiprecision m17 = "-4757003174447655866.";
    EXPECT_EQ(m17 << 17, "-623509920081203149668352.");
    Multiprecision m18 = "662716474672706556.";
    EXPECT_EQ(m18 << 18, "173727147536601987416064.");
    Multiprecision m19 = "5540167296155729817.";
    EXPECT_EQ(m19 << 19, "2904643231366895274295296.");
    Multiprecision m20 = "-4474265061727718627.";
    EXPECT_EQ(m20 << 20, "-4691606961366204287025152.");
    Multiprecision m21 = "5969471790508062653.";
    EXPECT_EQ(m21 << 21, "12518889704407564608864256.");
    Multiprecision m22 = "-5265695256006552653.";
    EXPECT_EQ(m22 << 22, "-22085926675049307818688512.");
    Multiprecision m23 = "6873645034989426292.";
    EXPECT_EQ(m23 << 23, "57660313729672581308481536.");
    Multiprecision m24 = "-7905873368535063591.";
    EXPECT_EQ(m24 << 24, "-132638545172560365439942656.");
    Multiprecision m25 = "-8693228561668035435.";
    EXPECT_EQ(m25 << 25, "-291696346632947901577297920.");
    Multiprecision m26 = "-7043023364600938157.";
    EXPECT_EQ(m26 << 26, "-472649297123826773050523648.");
    Multiprecision m27 = "9015025164445810640.";
    EXPECT_EQ(m27 << 27, "1209976195434743083219025920.");
    Multiprecision m28 = "9196295832952317037.";
    EXPECT_EQ(m28 << 28, "2468611865429455050083663872.");
    Multiprecision m29 = "8916222897565776331.";
    EXPECT_EQ(m29 << 29, "4786860718611420918811983872.");
    Multiprecision m30 = "-3572865449040353898.";
    EXPECT_EQ(m30 << 30, "-3836335064159168644044029952.");
    Multiprecision m31 = "-2302817096154652646.";
    EXPECT_EQ(m31 << 31, "-4945262058326960236404932608.");
    Multiprecision m32 = "8766205683538466197.";
    EXPECT_EQ(m32 << 32, "37650566720807037874116493312.");
    Multiprecision m33 = "1266493289020195681.";
    EXPECT_EQ(m33 << 33, "10879094513890432666830897152.");
    Multiprecision m34 = "-3803022461808819812.";
    EXPECT_EQ(m34 << 34, "-65335428397689160387587473408.");
    Multiprecision m35 = "2819148680530451735.";
    EXPECT_EQ(m35 << 35, "96865211083518737071451668480.");
    Multiprecision m36 = "-115014577408934116.";
    EXPECT_EQ(m36 << 36, "-7903741576554119143018725376.");
    Multiprecision m37 = "5094258238779213589.";
    EXPECT_EQ(m37 << 37, "700149521053929002539321131008.");
    Multiprecision m38 = "6926579382724352037.";
    EXPECT_EQ(m38 << 38, "1903963643004733400429182844928.");
    Multiprecision m39 = "-4288188454945410159.";
    EXPECT_EQ(m39 << 39, "-2357456534153639180571028488192.");
    Multiprecision m40 = "3988053960996393546.";
    EXPECT_EQ(m40 << 40, "4384911702313669082627960733696.");
    Multiprecision m41 = "-2258729486426273464.";
    EXPECT_EQ(m41 << 41, "-4966998668652400866832708272128.");
    Multiprecision m42 = "-3570856372627071555.";
    EXPECT_EQ(m42 << 42, "-15704792411285977019368310046720.");
    Multiprecision m43 = "-5982264275839351161.";
    EXPECT_EQ(m43 << 43, "-52620553053714710909654283583488.");
    Multiprecision m44 = "3885006050906365667.";
    EXPECT_EQ(m44 << 44, "68345749231226682154935199465472.");
    Multiprecision m45 = "2053736025208931047.";
    EXPECT_EQ(m45 << 45, "72259412483189886271357246767104.");
    Multiprecision m46 = "8299788207491102700.";
    EXPECT_EQ(m46 << 46, "584045673101733860269708070092800.");
    Multiprecision m47 = "-3983710329745327356.";
    EXPECT_EQ(m47 << 47, "-560657386143532875868869406752768.");
    Multiprecision m48 = "-983082642483429910.";
    EXPECT_EQ(m48 << 48, "-276713163897673592691636526120960.");
    Multiprecision m49 = "1193608687650818412.";
    EXPECT_EQ(m49 << 49, "671941955116301568858476242796544.");
    Multiprecision m50 = "-1409637001423648041.";
    EXPECT_EQ(m50 << 50, "-1587110168584801164226584152899584.");
    Multiprecision m51 = "-2318294309199358223.";
    EXPECT_EQ(m51 << 51, "-5220334693522685565061654122594304.");
    Multiprecision m52 = "7759206841436993107.";
    EXPECT_EQ(m52 << 52, "34944361039786245398611811287171072.");
    Multiprecision m53 = "2481279744575149350.";
    EXPECT_EQ(m53 << 53, "22349381066141204212750813967155200.");
    Multiprecision m54 = "6109086707179308773.";
    EXPECT_EQ(m54 << 54, "110051522472107141603454931416645632.");
    Multiprecision m55 = "-3627467853719473164.";
    EXPECT_EQ(m55 << 55, "-130693302994475777871668478058954752.");
    Multiprecision m56 = "5409416572247721086.";
    EXPECT_EQ(m56 << 56, "389789543345065958989778025495658496.");
    Multiprecision m57 = "-4523803313772409553.";
    EXPECT_EQ(m57 << 57, "-651948765382490837066286700183945216.");
    Multiprecision m58 = "353876324472099082.";
    EXPECT_EQ(m58 << 58, "101997906113778314260086863071019008.");
    Multiprecision m59 = "-6294738255972270889.";
    EXPECT_EQ(m59 << 59, "-3628669550590915205494791792421240832.");
    Multiprecision m60 = "4970526475298684237.";
    EXPECT_EQ(m60 << 60, "5730626862589526840406077674002317312.");
    Multiprecision m61 = "-5578860993609043904.";
    EXPECT_EQ(m61 << 61, "-12863977621488376418595430470787268608.");
    Multiprecision m62 = "-9076278761796422038.";
    EXPECT_EQ(m62 << 62, "-41856947865126003831195552773520228352.");
    Multiprecision m63 = "5117330721373399756.";
    EXPECT_EQ(m63 << 63, "47199045078853293325904793862141902848.");
    Multiprecision m64 = "-6514032524115553954.";
    EXPECT_EQ(m64 << 64, "-120162690860179866772223964920795889664.");

    Multiprecision<448> r0 = 6072323214095036032;
    EXPECT_EQ(r0 << 587, 6072323214095036032);
    Multiprecision<640> r1 = -8572474764272951220;
    EXPECT_EQ(r1 << 771, -8572474764272951220);
    Multiprecision<320> r2 = -9038684410537786378;
    EXPECT_EQ(r2 << 419, -9038684410537786378);
    Multiprecision<640> r3 = 1858310578097034903;
    EXPECT_EQ(r3 << 885, 1858310578097034903);
    Multiprecision<320> r4 = -1747591025141945043;
    EXPECT_EQ(r4 << 373, -1747591025141945043);
    Multiprecision<576> r5 = 6464175228791360326;
    EXPECT_EQ(r5 << 801, 6464175228791360326);
    Multiprecision<448> r6 = 8210083021775260927;
    EXPECT_EQ(r6 << 526, 8210083021775260927);
    Multiprecision<352> r7 = 9110354989523270182;
    EXPECT_EQ(r7 << 524, 9110354989523270182);
    Multiprecision<512> r8 = -5765383716230255672;
    EXPECT_EQ(r8 << 533, -5765383716230255672);
    Multiprecision<448> r9 = 5003573553825361921;
    EXPECT_EQ(r9 << 461, 5003573553825361921);
}

TEST(Bitwise, RightShift) {

}