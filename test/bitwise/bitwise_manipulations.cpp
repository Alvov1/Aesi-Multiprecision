#include <gtest/gtest.h>
#include "../../Aesi.h"
#include "../benchmarks/benchmarks.h"

TEST(Bitwise, GetSetBit) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi512 m0 = "72705387953193747615171521307121358171.";
    m0.setBit(126, true); EXPECT_EQ(m0, "157775979683428363481015173165063411035.");
    Aesi512 m1 = "245785293645359766049968031280770229918.";
    m1.setBit(73, false); EXPECT_EQ(m1, "245785293645359766049968031280770229918.");
    Aesi512 m2 = "40858198912689051933487804650601349382.";
    m2.setBit(50, true); EXPECT_EQ(m2, "40858198912689051933487804650601349382.");
    Aesi512 m3 = "54847862086374059183251323410557990283.";
    m3.setBit(121, false); EXPECT_EQ(m3, "54847862086374059183251323410557990283.");
    Aesi512 m4 = "110865429160911119237136997258727998569.";
    m4.setBit(31, true); EXPECT_EQ(m4, "110865429160911119237136997258727998569.");
    Aesi512 m5 = "187782640071050878679583809146823278321.";
    m5.setBit(128, true); EXPECT_EQ(m5, "528065006991989342142958416578591489777.");
    Aesi512 m6 = "295314379921377398364225673808278353050.";
    m6.setBit(15, true); EXPECT_EQ(m6, "295314379921377398364225673808278385818.");
    Aesi512 m7 = "3178747094108352161326739593823524879.";
    m7.setBit(71, false); EXPECT_EQ(m7, "3178747094108352161326739593823524879.");
    Aesi512 m8 = "86494165169548119678449182549238377356.";
    m8.setBit(76, true); EXPECT_EQ(m8, "86494165169548119678449182549238377356.");
    Aesi512 m9 = "32567167322927313851856437531751981177.";
    m9.setBit(40, true); EXPECT_EQ(m9, "32567167322927313851856437531751981177.");
    Aesi512 m10 = "172530118786257976509377267953319712204.";
    m10.setBit(20, true); EXPECT_EQ(m10, "172530118786257976509377267953319712204.");
    Aesi512 m11 = "172559365867210208688264944029949150686.";
    m11.setBit(99, true); EXPECT_EQ(m11, "172559365867210208688264944029949150686.");
    Aesi512 m12 = "78529106901055941434084997135871639427.";
    m12.setBit(10, true); EXPECT_EQ(m12, "78529106901055941434084997135871639427.");
    Aesi512 m13 = "207303494278178530702609902345571888268.";
    m13.setBit(69, true); EXPECT_EQ(m13, "207303494278178530702609902345571888268.");
    Aesi512 m14 = "75647073125004035230985916463405072523.";
    m14.setBit(3, false); EXPECT_EQ(m14, "75647073125004035230985916463405072515.");
    Aesi512 m15 = "222723315237755082168376048193872639645.";
    m15.setBit(107, true); EXPECT_EQ(m15, "222723477497031911381739439771882927773.");
    Aesi512 m16 = "193675621579885645481372969626118961567.";
    m16.setBit(33, false); EXPECT_EQ(m16, "193675621579885645481372969617529026975.");
    Aesi512 m17 = "184830840876535640520166369784004708864.";
    m17.setBit(94, true); EXPECT_EQ(m17, "184830840896342681148732454182390696448.");
    Aesi512 m18 = "184781425874546653782780922198149710463.";
    m18.setBit(123, true); EXPECT_EQ(m18, "184781425874546653782780922198149710463.");
    Aesi512 m19 = "143111532682479361522938570020545208955.";
    m19.setBit(47, false); EXPECT_EQ(m19, "143111532682479361522938429283056853627.");
    Aesi512 m20 = "271575289760478049691481842232695651446.";
    m20.setBit(11, false); EXPECT_EQ(m20, "271575289760478049691481842232695649398.");
    Aesi512 m21 = "187912626417229430413829117802893310299.";
    m21.setBit(68, false); EXPECT_EQ(m21, "187912626417229430413829117802893310299.");
    Aesi512 m22 = "158932010648066253535888104656448665002.";
    m22.setBit(79, true); EXPECT_EQ(m22, "158932010648066253535888104656448665002.");
    Aesi512 m23 = "109487819811077581867635815837691548334.";
    m23.setBit(113, true); EXPECT_EQ(m23, "109487819811077581867635815837691548334.");
    Aesi512 m24 = "47379343172205350263905530302298557228.";
    m24.setBit(61, false); EXPECT_EQ(m24, "47379343172205350263905530302298557228.");
    Aesi512 m25 = "158911198835109948520770291433578090947.";
    m25.setBit(50, false); EXPECT_EQ(m25, "158911198835109948520770291433578090947.");
    Aesi512 m26 = "248326728370153648748027863303163864728.";
    m26.setBit(91, false); EXPECT_EQ(m26, "248326728367677768669457102753365616280.");
    Aesi512 m27 = "176392199758157475536173991226043128119.";
    m27.setBit(42, true); EXPECT_EQ(m27, "176392199758157475536173991226043128119.");
    Aesi512 m28 = "238588105982679173784425043462049184726.";
    m28.setBit(105, false); EXPECT_EQ(m28, "238588105982679173784425043462049184726.");
    Aesi512 m29 = "150346354478535467451721603451021909402.";
    m29.setBit(107, true); EXPECT_EQ(m29, "150346516737812296665084995029032197530.");
    Aesi512 m30 = "273819485712316995631595319491380561223.";
    m30.setBit(114, false); EXPECT_EQ(m30, "273798716524882856321081197506063680839.");
    Aesi512 m31 = "9465809177493504473582186962123883962.";
    m31.setBit(52, false); EXPECT_EQ(m31, "9465809177493504473582186962123883962.");
    Aesi512 m32 = "328749017563933875762256624294091022769.";
    m32.setBit(82, false); EXPECT_EQ(m32, "328749017563929040058978165777392198065.");
    Aesi512 m33 = "187349705948370703926994252284122213259.";
    m33.setBit(21, false); EXPECT_EQ(m33, "187349705948370703926994252284122213259.");
    Aesi512 m34 = "95199700526301209891562846305547693713.";
    m34.setBit(46, false); EXPECT_EQ(m34, "95199700526301209891562846305547693713.");
    Aesi512 m35 = "103655044468308507172810437786291934566.";
    m35.setBit(51, true); EXPECT_EQ(m35, "103655044468308507172810437786291934566.");
    Aesi512 m36 = "62548752160102009726032985107282388037.";
    m36.setBit(71, true); EXPECT_EQ(m36, "62548752160102009726032985107282388037.");
    Aesi512 m37 = "114713660827724670972157050548957340052.";
    m37.setBit(82, true); EXPECT_EQ(m37, "114713660827724670972157050548957340052.");
    Aesi512 m38 = "298233052050341589489614113949082900217.";
    m38.setBit(77, false); EXPECT_EQ(m38, "298233052050341589489614113949082900217.");
    Aesi512 m39 = "244677446963714695792063603125629139705.";
    m39.setBit(52, true); EXPECT_EQ(m39, "244677446963714695792063603125629139705.");
    Aesi512 u0 = "183266889657794562429184270028014133994.";
    EXPECT_EQ(u0.getBit(0), false);
    EXPECT_EQ(u0.getBit(1), true); EXPECT_EQ(u0.getBit(2), false); EXPECT_EQ(u0.getBit(3), true); EXPECT_EQ(u0.getBit(4), false); EXPECT_EQ(u0.getBit(5), true); EXPECT_EQ(u0.getBit(6), true); EXPECT_EQ(u0.getBit(7), true); EXPECT_EQ(u0.getBit(8), false); EXPECT_EQ(u0.getBit(9), true);
    EXPECT_EQ(u0.getBit(10), false); EXPECT_EQ(u0.getBit(11), false); EXPECT_EQ(u0.getBit(12), false); EXPECT_EQ(u0.getBit(13), true); EXPECT_EQ(u0.getBit(14), false); EXPECT_EQ(u0.getBit(15), true); EXPECT_EQ(u0.getBit(16), false); EXPECT_EQ(u0.getBit(17), false); EXPECT_EQ(u0.getBit(18), true);
    EXPECT_EQ(u0.getBit(19), true); EXPECT_EQ(u0.getBit(20), true); EXPECT_EQ(u0.getBit(21), false); EXPECT_EQ(u0.getBit(22), true); EXPECT_EQ(u0.getBit(23), false); EXPECT_EQ(u0.getBit(24), false); EXPECT_EQ(u0.getBit(25), true); EXPECT_EQ(u0.getBit(26), false); EXPECT_EQ(u0.getBit(27), true);
    EXPECT_EQ(u0.getBit(28), true); EXPECT_EQ(u0.getBit(29), true); EXPECT_EQ(u0.getBit(30), true); EXPECT_EQ(u0.getBit(31), false); EXPECT_EQ(u0.getBit(32), false); EXPECT_EQ(u0.getBit(33), false); EXPECT_EQ(u0.getBit(34), false); EXPECT_EQ(u0.getBit(35), false); EXPECT_EQ(u0.getBit(36), true);
    EXPECT_EQ(u0.getBit(37), true); EXPECT_EQ(u0.getBit(38), false); EXPECT_EQ(u0.getBit(39), true); EXPECT_EQ(u0.getBit(40), false); EXPECT_EQ(u0.getBit(41), false); EXPECT_EQ(u0.getBit(42), true); EXPECT_EQ(u0.getBit(43), false); EXPECT_EQ(u0.getBit(44), false); EXPECT_EQ(u0.getBit(45), false);
    EXPECT_EQ(u0.getBit(46), false); EXPECT_EQ(u0.getBit(47), true); EXPECT_EQ(u0.getBit(48), false); EXPECT_EQ(u0.getBit(49), true); EXPECT_EQ(u0.getBit(50), true); EXPECT_EQ(u0.getBit(51), false); EXPECT_EQ(u0.getBit(52), false); EXPECT_EQ(u0.getBit(53), true); EXPECT_EQ(u0.getBit(54), true);
    EXPECT_EQ(u0.getBit(55), false); EXPECT_EQ(u0.getBit(56), false); EXPECT_EQ(u0.getBit(57), true); EXPECT_EQ(u0.getBit(58), true); EXPECT_EQ(u0.getBit(59), true); EXPECT_EQ(u0.getBit(60), false); EXPECT_EQ(u0.getBit(61), false); EXPECT_EQ(u0.getBit(62), false); EXPECT_EQ(u0.getBit(63), true);
    EXPECT_EQ(u0.getBit(64), false); EXPECT_EQ(u0.getBit(65), true); EXPECT_EQ(u0.getBit(66), false); EXPECT_EQ(u0.getBit(67), true); EXPECT_EQ(u0.getBit(68), true); EXPECT_EQ(u0.getBit(69), true); EXPECT_EQ(u0.getBit(70), true); EXPECT_EQ(u0.getBit(71), false); EXPECT_EQ(u0.getBit(72), true);
    EXPECT_EQ(u0.getBit(73), true); EXPECT_EQ(u0.getBit(74), false); EXPECT_EQ(u0.getBit(75), false); EXPECT_EQ(u0.getBit(76), false); EXPECT_EQ(u0.getBit(77), true); EXPECT_EQ(u0.getBit(78), false); EXPECT_EQ(u0.getBit(79), true); EXPECT_EQ(u0.getBit(80), false); EXPECT_EQ(u0.getBit(81), true);
    EXPECT_EQ(u0.getBit(82), true); EXPECT_EQ(u0.getBit(83), true); EXPECT_EQ(u0.getBit(84), false); EXPECT_EQ(u0.getBit(85), false); EXPECT_EQ(u0.getBit(86), true); EXPECT_EQ(u0.getBit(87), false); EXPECT_EQ(u0.getBit(88), true); EXPECT_EQ(u0.getBit(89), true); EXPECT_EQ(u0.getBit(90), true);
    EXPECT_EQ(u0.getBit(91), true); EXPECT_EQ(u0.getBit(92), false); EXPECT_EQ(u0.getBit(93), false); EXPECT_EQ(u0.getBit(94), false); EXPECT_EQ(u0.getBit(95), true); EXPECT_EQ(u0.getBit(96), true); EXPECT_EQ(u0.getBit(97), true); EXPECT_EQ(u0.getBit(98), true); EXPECT_EQ(u0.getBit(99), false);
    EXPECT_EQ(u0.getBit(100), false); EXPECT_EQ(u0.getBit(101), false); EXPECT_EQ(u0.getBit(102), true); EXPECT_EQ(u0.getBit(103), false); EXPECT_EQ(u0.getBit(104), true); EXPECT_EQ(u0.getBit(105), true); EXPECT_EQ(u0.getBit(106), false); EXPECT_EQ(u0.getBit(107), true); EXPECT_EQ(u0.getBit(108), false);
    EXPECT_EQ(u0.getBit(109), true); EXPECT_EQ(u0.getBit(110), true); EXPECT_EQ(u0.getBit(111), true); EXPECT_EQ(u0.getBit(112), true); EXPECT_EQ(u0.getBit(113), true); EXPECT_EQ(u0.getBit(114), true); EXPECT_EQ(u0.getBit(115), true); EXPECT_EQ(u0.getBit(116), true); EXPECT_EQ(u0.getBit(117), false);
    EXPECT_EQ(u0.getBit(118), true); EXPECT_EQ(u0.getBit(119), true); EXPECT_EQ(u0.getBit(120), true); EXPECT_EQ(u0.getBit(121), false); EXPECT_EQ(u0.getBit(122), false); EXPECT_EQ(u0.getBit(123), true); EXPECT_EQ(u0.getBit(124), false); EXPECT_EQ(u0.getBit(125), false); EXPECT_EQ(u0.getBit(126), false);
    EXPECT_EQ(u0.getBit(127), true);

    Aesi512 u1 = "28532736677951985108511975536651973171.";
    EXPECT_EQ(u1.getBit(0), true);
    EXPECT_EQ(u1.getBit(1), true); EXPECT_EQ(u1.getBit(2), false); EXPECT_EQ(u1.getBit(3), false); EXPECT_EQ(u1.getBit(4), true); EXPECT_EQ(u1.getBit(5), true); EXPECT_EQ(u1.getBit(6), false); EXPECT_EQ(u1.getBit(7), false); EXPECT_EQ(u1.getBit(8), false); EXPECT_EQ(u1.getBit(9), true);
    EXPECT_EQ(u1.getBit(10), false); EXPECT_EQ(u1.getBit(11), false); EXPECT_EQ(u1.getBit(12), true); EXPECT_EQ(u1.getBit(13), true); EXPECT_EQ(u1.getBit(14), false); EXPECT_EQ(u1.getBit(15), true); EXPECT_EQ(u1.getBit(16), false); EXPECT_EQ(u1.getBit(17), false); EXPECT_EQ(u1.getBit(18), true);
    EXPECT_EQ(u1.getBit(19), false); EXPECT_EQ(u1.getBit(20), false); EXPECT_EQ(u1.getBit(21), true); EXPECT_EQ(u1.getBit(22), false); EXPECT_EQ(u1.getBit(23), true); EXPECT_EQ(u1.getBit(24), false); EXPECT_EQ(u1.getBit(25), true); EXPECT_EQ(u1.getBit(26), true); EXPECT_EQ(u1.getBit(27), true);
    EXPECT_EQ(u1.getBit(28), false); EXPECT_EQ(u1.getBit(29), false); EXPECT_EQ(u1.getBit(30), false); EXPECT_EQ(u1.getBit(31), true); EXPECT_EQ(u1.getBit(32), false); EXPECT_EQ(u1.getBit(33), false); EXPECT_EQ(u1.getBit(34), true); EXPECT_EQ(u1.getBit(35), true); EXPECT_EQ(u1.getBit(36), true);
    EXPECT_EQ(u1.getBit(37), true); EXPECT_EQ(u1.getBit(38), true); EXPECT_EQ(u1.getBit(39), true); EXPECT_EQ(u1.getBit(40), false); EXPECT_EQ(u1.getBit(41), false); EXPECT_EQ(u1.getBit(42), false); EXPECT_EQ(u1.getBit(43), true); EXPECT_EQ(u1.getBit(44), true); EXPECT_EQ(u1.getBit(45), false);
    EXPECT_EQ(u1.getBit(46), false); EXPECT_EQ(u1.getBit(47), true); EXPECT_EQ(u1.getBit(48), false); EXPECT_EQ(u1.getBit(49), true); EXPECT_EQ(u1.getBit(50), true); EXPECT_EQ(u1.getBit(51), false); EXPECT_EQ(u1.getBit(52), false); EXPECT_EQ(u1.getBit(53), false); EXPECT_EQ(u1.getBit(54), true);
    EXPECT_EQ(u1.getBit(55), true); EXPECT_EQ(u1.getBit(56), false); EXPECT_EQ(u1.getBit(57), false); EXPECT_EQ(u1.getBit(58), true); EXPECT_EQ(u1.getBit(59), false); EXPECT_EQ(u1.getBit(60), false); EXPECT_EQ(u1.getBit(61), true); EXPECT_EQ(u1.getBit(62), true); EXPECT_EQ(u1.getBit(63), true);
    EXPECT_EQ(u1.getBit(64), false); EXPECT_EQ(u1.getBit(65), true); EXPECT_EQ(u1.getBit(66), true); EXPECT_EQ(u1.getBit(67), true); EXPECT_EQ(u1.getBit(68), true); EXPECT_EQ(u1.getBit(69), false); EXPECT_EQ(u1.getBit(70), false); EXPECT_EQ(u1.getBit(71), false); EXPECT_EQ(u1.getBit(72), false);
    EXPECT_EQ(u1.getBit(73), false); EXPECT_EQ(u1.getBit(74), false); EXPECT_EQ(u1.getBit(75), false); EXPECT_EQ(u1.getBit(76), true); EXPECT_EQ(u1.getBit(77), false); EXPECT_EQ(u1.getBit(78), false); EXPECT_EQ(u1.getBit(79), true); EXPECT_EQ(u1.getBit(80), false); EXPECT_EQ(u1.getBit(81), true);
    EXPECT_EQ(u1.getBit(82), false); EXPECT_EQ(u1.getBit(83), true); EXPECT_EQ(u1.getBit(84), false); EXPECT_EQ(u1.getBit(85), false); EXPECT_EQ(u1.getBit(86), true); EXPECT_EQ(u1.getBit(87), false); EXPECT_EQ(u1.getBit(88), true); EXPECT_EQ(u1.getBit(89), true); EXPECT_EQ(u1.getBit(90), true);
    EXPECT_EQ(u1.getBit(91), true); EXPECT_EQ(u1.getBit(92), true); EXPECT_EQ(u1.getBit(93), true); EXPECT_EQ(u1.getBit(94), false); EXPECT_EQ(u1.getBit(95), true); EXPECT_EQ(u1.getBit(96), true); EXPECT_EQ(u1.getBit(97), true); EXPECT_EQ(u1.getBit(98), true); EXPECT_EQ(u1.getBit(99), false);
    EXPECT_EQ(u1.getBit(100), false); EXPECT_EQ(u1.getBit(101), false); EXPECT_EQ(u1.getBit(102), false); EXPECT_EQ(u1.getBit(103), true); EXPECT_EQ(u1.getBit(104), false); EXPECT_EQ(u1.getBit(105), false); EXPECT_EQ(u1.getBit(106), true); EXPECT_EQ(u1.getBit(107), false); EXPECT_EQ(u1.getBit(108), true);
    EXPECT_EQ(u1.getBit(109), true); EXPECT_EQ(u1.getBit(110), false); EXPECT_EQ(u1.getBit(111), false); EXPECT_EQ(u1.getBit(112), true); EXPECT_EQ(u1.getBit(113), true); EXPECT_EQ(u1.getBit(114), true); EXPECT_EQ(u1.getBit(115), false); EXPECT_EQ(u1.getBit(116), true); EXPECT_EQ(u1.getBit(117), true);
    EXPECT_EQ(u1.getBit(118), true); EXPECT_EQ(u1.getBit(119), false); EXPECT_EQ(u1.getBit(120), true); EXPECT_EQ(u1.getBit(121), false); EXPECT_EQ(u1.getBit(122), true); EXPECT_EQ(u1.getBit(123), false); EXPECT_EQ(u1.getBit(124), true);

    Aesi512 u2 = "202963489620510457840756855970746955719.";
    EXPECT_EQ(u2.getBit(0), true);
    EXPECT_EQ(u2.getBit(1), true); EXPECT_EQ(u2.getBit(2), true); EXPECT_EQ(u2.getBit(3), false); EXPECT_EQ(u2.getBit(4), false); EXPECT_EQ(u2.getBit(5), false); EXPECT_EQ(u2.getBit(6), true); EXPECT_EQ(u2.getBit(7), true); EXPECT_EQ(u2.getBit(8), true); EXPECT_EQ(u2.getBit(9), true);
    EXPECT_EQ(u2.getBit(10), true); EXPECT_EQ(u2.getBit(11), false); EXPECT_EQ(u2.getBit(12), true); EXPECT_EQ(u2.getBit(13), true); EXPECT_EQ(u2.getBit(14), true); EXPECT_EQ(u2.getBit(15), false); EXPECT_EQ(u2.getBit(16), true); EXPECT_EQ(u2.getBit(17), false); EXPECT_EQ(u2.getBit(18), false);
    EXPECT_EQ(u2.getBit(19), false); EXPECT_EQ(u2.getBit(20), false); EXPECT_EQ(u2.getBit(21), false); EXPECT_EQ(u2.getBit(22), true); EXPECT_EQ(u2.getBit(23), false); EXPECT_EQ(u2.getBit(24), false); EXPECT_EQ(u2.getBit(25), false); EXPECT_EQ(u2.getBit(26), true); EXPECT_EQ(u2.getBit(27), true);
    EXPECT_EQ(u2.getBit(28), true); EXPECT_EQ(u2.getBit(29), false); EXPECT_EQ(u2.getBit(30), true); EXPECT_EQ(u2.getBit(31), false); EXPECT_EQ(u2.getBit(32), true); EXPECT_EQ(u2.getBit(33), false); EXPECT_EQ(u2.getBit(34), false); EXPECT_EQ(u2.getBit(35), false); EXPECT_EQ(u2.getBit(36), true);
    EXPECT_EQ(u2.getBit(37), false); EXPECT_EQ(u2.getBit(38), true); EXPECT_EQ(u2.getBit(39), true); EXPECT_EQ(u2.getBit(40), false); EXPECT_EQ(u2.getBit(41), true); EXPECT_EQ(u2.getBit(42), false); EXPECT_EQ(u2.getBit(43), true); EXPECT_EQ(u2.getBit(44), true); EXPECT_EQ(u2.getBit(45), false);
    EXPECT_EQ(u2.getBit(46), true); EXPECT_EQ(u2.getBit(47), true); EXPECT_EQ(u2.getBit(48), true); EXPECT_EQ(u2.getBit(49), false); EXPECT_EQ(u2.getBit(50), false); EXPECT_EQ(u2.getBit(51), false); EXPECT_EQ(u2.getBit(52), true); EXPECT_EQ(u2.getBit(53), true); EXPECT_EQ(u2.getBit(54), false);
    EXPECT_EQ(u2.getBit(55), false); EXPECT_EQ(u2.getBit(56), false); EXPECT_EQ(u2.getBit(57), true); EXPECT_EQ(u2.getBit(58), false); EXPECT_EQ(u2.getBit(59), true); EXPECT_EQ(u2.getBit(60), true); EXPECT_EQ(u2.getBit(61), false); EXPECT_EQ(u2.getBit(62), false); EXPECT_EQ(u2.getBit(63), true);
    EXPECT_EQ(u2.getBit(64), false); EXPECT_EQ(u2.getBit(65), true); EXPECT_EQ(u2.getBit(66), true); EXPECT_EQ(u2.getBit(67), true); EXPECT_EQ(u2.getBit(68), true); EXPECT_EQ(u2.getBit(69), true); EXPECT_EQ(u2.getBit(70), false); EXPECT_EQ(u2.getBit(71), false); EXPECT_EQ(u2.getBit(72), false);
    EXPECT_EQ(u2.getBit(73), true); EXPECT_EQ(u2.getBit(74), false); EXPECT_EQ(u2.getBit(75), true); EXPECT_EQ(u2.getBit(76), false); EXPECT_EQ(u2.getBit(77), true); EXPECT_EQ(u2.getBit(78), true); EXPECT_EQ(u2.getBit(79), true); EXPECT_EQ(u2.getBit(80), true); EXPECT_EQ(u2.getBit(81), false);
    EXPECT_EQ(u2.getBit(82), false); EXPECT_EQ(u2.getBit(83), false); EXPECT_EQ(u2.getBit(84), true); EXPECT_EQ(u2.getBit(85), true); EXPECT_EQ(u2.getBit(86), false); EXPECT_EQ(u2.getBit(87), true); EXPECT_EQ(u2.getBit(88), true); EXPECT_EQ(u2.getBit(89), false); EXPECT_EQ(u2.getBit(90), false);
    EXPECT_EQ(u2.getBit(91), false); EXPECT_EQ(u2.getBit(92), false); EXPECT_EQ(u2.getBit(93), true); EXPECT_EQ(u2.getBit(94), true); EXPECT_EQ(u2.getBit(95), false); EXPECT_EQ(u2.getBit(96), false); EXPECT_EQ(u2.getBit(97), true); EXPECT_EQ(u2.getBit(98), false); EXPECT_EQ(u2.getBit(99), false);
    EXPECT_EQ(u2.getBit(100), false); EXPECT_EQ(u2.getBit(101), true); EXPECT_EQ(u2.getBit(102), false); EXPECT_EQ(u2.getBit(103), true); EXPECT_EQ(u2.getBit(104), false); EXPECT_EQ(u2.getBit(105), false); EXPECT_EQ(u2.getBit(106), false); EXPECT_EQ(u2.getBit(107), true); EXPECT_EQ(u2.getBit(108), true);
    EXPECT_EQ(u2.getBit(109), false); EXPECT_EQ(u2.getBit(110), true); EXPECT_EQ(u2.getBit(111), false); EXPECT_EQ(u2.getBit(112), true); EXPECT_EQ(u2.getBit(113), false); EXPECT_EQ(u2.getBit(114), false); EXPECT_EQ(u2.getBit(115), false); EXPECT_EQ(u2.getBit(116), true); EXPECT_EQ(u2.getBit(117), true);
    EXPECT_EQ(u2.getBit(118), false); EXPECT_EQ(u2.getBit(119), true); EXPECT_EQ(u2.getBit(120), false); EXPECT_EQ(u2.getBit(121), false); EXPECT_EQ(u2.getBit(122), false); EXPECT_EQ(u2.getBit(123), true); EXPECT_EQ(u2.getBit(124), true); EXPECT_EQ(u2.getBit(125), false); EXPECT_EQ(u2.getBit(126), false);
    EXPECT_EQ(u2.getBit(127), true);

    Aesi512 u3 = "99039543241645718481744753772010007488.";
    EXPECT_EQ(u3.getBit(0), false);
    EXPECT_EQ(u3.getBit(1), false); EXPECT_EQ(u3.getBit(2), false); EXPECT_EQ(u3.getBit(3), false); EXPECT_EQ(u3.getBit(4), false); EXPECT_EQ(u3.getBit(5), false); EXPECT_EQ(u3.getBit(6), true); EXPECT_EQ(u3.getBit(7), true); EXPECT_EQ(u3.getBit(8), true); EXPECT_EQ(u3.getBit(9), true);
    EXPECT_EQ(u3.getBit(10), false); EXPECT_EQ(u3.getBit(11), true); EXPECT_EQ(u3.getBit(12), false); EXPECT_EQ(u3.getBit(13), true); EXPECT_EQ(u3.getBit(14), true); EXPECT_EQ(u3.getBit(15), true); EXPECT_EQ(u3.getBit(16), true); EXPECT_EQ(u3.getBit(17), false); EXPECT_EQ(u3.getBit(18), true);
    EXPECT_EQ(u3.getBit(19), true); EXPECT_EQ(u3.getBit(20), false); EXPECT_EQ(u3.getBit(21), true); EXPECT_EQ(u3.getBit(22), true); EXPECT_EQ(u3.getBit(23), false); EXPECT_EQ(u3.getBit(24), false); EXPECT_EQ(u3.getBit(25), true); EXPECT_EQ(u3.getBit(26), true); EXPECT_EQ(u3.getBit(27), true);
    EXPECT_EQ(u3.getBit(28), false); EXPECT_EQ(u3.getBit(29), false); EXPECT_EQ(u3.getBit(30), true); EXPECT_EQ(u3.getBit(31), true); EXPECT_EQ(u3.getBit(32), false); EXPECT_EQ(u3.getBit(33), false); EXPECT_EQ(u3.getBit(34), false); EXPECT_EQ(u3.getBit(35), false); EXPECT_EQ(u3.getBit(36), false);
    EXPECT_EQ(u3.getBit(37), false); EXPECT_EQ(u3.getBit(38), true); EXPECT_EQ(u3.getBit(39), true); EXPECT_EQ(u3.getBit(40), true); EXPECT_EQ(u3.getBit(41), true); EXPECT_EQ(u3.getBit(42), true); EXPECT_EQ(u3.getBit(43), true); EXPECT_EQ(u3.getBit(44), true); EXPECT_EQ(u3.getBit(45), false);
    EXPECT_EQ(u3.getBit(46), false); EXPECT_EQ(u3.getBit(47), true); EXPECT_EQ(u3.getBit(48), true); EXPECT_EQ(u3.getBit(49), false); EXPECT_EQ(u3.getBit(50), true); EXPECT_EQ(u3.getBit(51), true); EXPECT_EQ(u3.getBit(52), true); EXPECT_EQ(u3.getBit(53), true); EXPECT_EQ(u3.getBit(54), false);
    EXPECT_EQ(u3.getBit(55), false); EXPECT_EQ(u3.getBit(56), false); EXPECT_EQ(u3.getBit(57), true); EXPECT_EQ(u3.getBit(58), false); EXPECT_EQ(u3.getBit(59), true); EXPECT_EQ(u3.getBit(60), false); EXPECT_EQ(u3.getBit(61), true); EXPECT_EQ(u3.getBit(62), true); EXPECT_EQ(u3.getBit(63), true);
    EXPECT_EQ(u3.getBit(64), false); EXPECT_EQ(u3.getBit(65), true); EXPECT_EQ(u3.getBit(66), true); EXPECT_EQ(u3.getBit(67), true); EXPECT_EQ(u3.getBit(68), true); EXPECT_EQ(u3.getBit(69), false); EXPECT_EQ(u3.getBit(70), true); EXPECT_EQ(u3.getBit(71), true); EXPECT_EQ(u3.getBit(72), true);
    EXPECT_EQ(u3.getBit(73), true); EXPECT_EQ(u3.getBit(74), false); EXPECT_EQ(u3.getBit(75), false); EXPECT_EQ(u3.getBit(76), true); EXPECT_EQ(u3.getBit(77), false); EXPECT_EQ(u3.getBit(78), false); EXPECT_EQ(u3.getBit(79), true); EXPECT_EQ(u3.getBit(80), false); EXPECT_EQ(u3.getBit(81), false);
    EXPECT_EQ(u3.getBit(82), true); EXPECT_EQ(u3.getBit(83), true); EXPECT_EQ(u3.getBit(84), true); EXPECT_EQ(u3.getBit(85), false); EXPECT_EQ(u3.getBit(86), false); EXPECT_EQ(u3.getBit(87), true); EXPECT_EQ(u3.getBit(88), true); EXPECT_EQ(u3.getBit(89), true); EXPECT_EQ(u3.getBit(90), true);
    EXPECT_EQ(u3.getBit(91), true); EXPECT_EQ(u3.getBit(92), true); EXPECT_EQ(u3.getBit(93), true); EXPECT_EQ(u3.getBit(94), false); EXPECT_EQ(u3.getBit(95), true); EXPECT_EQ(u3.getBit(96), true); EXPECT_EQ(u3.getBit(97), true); EXPECT_EQ(u3.getBit(98), false); EXPECT_EQ(u3.getBit(99), true);
    EXPECT_EQ(u3.getBit(100), true); EXPECT_EQ(u3.getBit(101), true); EXPECT_EQ(u3.getBit(102), true); EXPECT_EQ(u3.getBit(103), false); EXPECT_EQ(u3.getBit(104), false); EXPECT_EQ(u3.getBit(105), true); EXPECT_EQ(u3.getBit(106), false); EXPECT_EQ(u3.getBit(107), false); EXPECT_EQ(u3.getBit(108), true);
    EXPECT_EQ(u3.getBit(109), false); EXPECT_EQ(u3.getBit(110), true); EXPECT_EQ(u3.getBit(111), false); EXPECT_EQ(u3.getBit(112), false); EXPECT_EQ(u3.getBit(113), true); EXPECT_EQ(u3.getBit(114), false); EXPECT_EQ(u3.getBit(115), false); EXPECT_EQ(u3.getBit(116), false); EXPECT_EQ(u3.getBit(117), false);
    EXPECT_EQ(u3.getBit(118), false); EXPECT_EQ(u3.getBit(119), true); EXPECT_EQ(u3.getBit(120), false); EXPECT_EQ(u3.getBit(121), true); EXPECT_EQ(u3.getBit(122), false); EXPECT_EQ(u3.getBit(123), true); EXPECT_EQ(u3.getBit(124), false); EXPECT_EQ(u3.getBit(125), false); EXPECT_EQ(u3.getBit(126), true);


    Aesi512 u4 = "185521423754314153400893953240556219688.";
    EXPECT_EQ(u4.getBit(0), false);
    EXPECT_EQ(u4.getBit(1), false); EXPECT_EQ(u4.getBit(2), false); EXPECT_EQ(u4.getBit(3), true); EXPECT_EQ(u4.getBit(4), false); EXPECT_EQ(u4.getBit(5), true); EXPECT_EQ(u4.getBit(6), false); EXPECT_EQ(u4.getBit(7), false); EXPECT_EQ(u4.getBit(8), true); EXPECT_EQ(u4.getBit(9), false);
    EXPECT_EQ(u4.getBit(10), true); EXPECT_EQ(u4.getBit(11), true); EXPECT_EQ(u4.getBit(12), false); EXPECT_EQ(u4.getBit(13), true); EXPECT_EQ(u4.getBit(14), false); EXPECT_EQ(u4.getBit(15), true); EXPECT_EQ(u4.getBit(16), true); EXPECT_EQ(u4.getBit(17), false); EXPECT_EQ(u4.getBit(18), false);
    EXPECT_EQ(u4.getBit(19), true); EXPECT_EQ(u4.getBit(20), false); EXPECT_EQ(u4.getBit(21), false); EXPECT_EQ(u4.getBit(22), false); EXPECT_EQ(u4.getBit(23), false); EXPECT_EQ(u4.getBit(24), true); EXPECT_EQ(u4.getBit(25), false); EXPECT_EQ(u4.getBit(26), false); EXPECT_EQ(u4.getBit(27), true);
    EXPECT_EQ(u4.getBit(28), false); EXPECT_EQ(u4.getBit(29), false); EXPECT_EQ(u4.getBit(30), false); EXPECT_EQ(u4.getBit(31), true); EXPECT_EQ(u4.getBit(32), false); EXPECT_EQ(u4.getBit(33), false); EXPECT_EQ(u4.getBit(34), false); EXPECT_EQ(u4.getBit(35), true); EXPECT_EQ(u4.getBit(36), false);
    EXPECT_EQ(u4.getBit(37), false); EXPECT_EQ(u4.getBit(38), false); EXPECT_EQ(u4.getBit(39), true); EXPECT_EQ(u4.getBit(40), false); EXPECT_EQ(u4.getBit(41), false); EXPECT_EQ(u4.getBit(42), false); EXPECT_EQ(u4.getBit(43), false); EXPECT_EQ(u4.getBit(44), true); EXPECT_EQ(u4.getBit(45), true);
    EXPECT_EQ(u4.getBit(46), false); EXPECT_EQ(u4.getBit(47), true); EXPECT_EQ(u4.getBit(48), false); EXPECT_EQ(u4.getBit(49), false); EXPECT_EQ(u4.getBit(50), true); EXPECT_EQ(u4.getBit(51), true); EXPECT_EQ(u4.getBit(52), false); EXPECT_EQ(u4.getBit(53), true); EXPECT_EQ(u4.getBit(54), false);
    EXPECT_EQ(u4.getBit(55), false); EXPECT_EQ(u4.getBit(56), true); EXPECT_EQ(u4.getBit(57), true); EXPECT_EQ(u4.getBit(58), true); EXPECT_EQ(u4.getBit(59), true); EXPECT_EQ(u4.getBit(60), false); EXPECT_EQ(u4.getBit(61), true); EXPECT_EQ(u4.getBit(62), false); EXPECT_EQ(u4.getBit(63), false);
    EXPECT_EQ(u4.getBit(64), false); EXPECT_EQ(u4.getBit(65), true); EXPECT_EQ(u4.getBit(66), false); EXPECT_EQ(u4.getBit(67), false); EXPECT_EQ(u4.getBit(68), true); EXPECT_EQ(u4.getBit(69), false); EXPECT_EQ(u4.getBit(70), false); EXPECT_EQ(u4.getBit(71), true); EXPECT_EQ(u4.getBit(72), false);
    EXPECT_EQ(u4.getBit(73), true); EXPECT_EQ(u4.getBit(74), true); EXPECT_EQ(u4.getBit(75), false); EXPECT_EQ(u4.getBit(76), false); EXPECT_EQ(u4.getBit(77), true); EXPECT_EQ(u4.getBit(78), true); EXPECT_EQ(u4.getBit(79), false); EXPECT_EQ(u4.getBit(80), true); EXPECT_EQ(u4.getBit(81), true);
    EXPECT_EQ(u4.getBit(82), false); EXPECT_EQ(u4.getBit(83), false); EXPECT_EQ(u4.getBit(84), true); EXPECT_EQ(u4.getBit(85), false); EXPECT_EQ(u4.getBit(86), true); EXPECT_EQ(u4.getBit(87), true); EXPECT_EQ(u4.getBit(88), true); EXPECT_EQ(u4.getBit(89), false); EXPECT_EQ(u4.getBit(90), true);
    EXPECT_EQ(u4.getBit(91), true); EXPECT_EQ(u4.getBit(92), true); EXPECT_EQ(u4.getBit(93), true); EXPECT_EQ(u4.getBit(94), true); EXPECT_EQ(u4.getBit(95), false); EXPECT_EQ(u4.getBit(96), false); EXPECT_EQ(u4.getBit(97), false); EXPECT_EQ(u4.getBit(98), true); EXPECT_EQ(u4.getBit(99), false);
    EXPECT_EQ(u4.getBit(100), false); EXPECT_EQ(u4.getBit(101), true); EXPECT_EQ(u4.getBit(102), true); EXPECT_EQ(u4.getBit(103), false); EXPECT_EQ(u4.getBit(104), false); EXPECT_EQ(u4.getBit(105), false); EXPECT_EQ(u4.getBit(106), false); EXPECT_EQ(u4.getBit(107), false); EXPECT_EQ(u4.getBit(108), false);
    EXPECT_EQ(u4.getBit(109), true); EXPECT_EQ(u4.getBit(110), false); EXPECT_EQ(u4.getBit(111), false); EXPECT_EQ(u4.getBit(112), false); EXPECT_EQ(u4.getBit(113), true); EXPECT_EQ(u4.getBit(114), false); EXPECT_EQ(u4.getBit(115), false); EXPECT_EQ(u4.getBit(116), true); EXPECT_EQ(u4.getBit(117), false);
    EXPECT_EQ(u4.getBit(118), false); EXPECT_EQ(u4.getBit(119), true); EXPECT_EQ(u4.getBit(120), true); EXPECT_EQ(u4.getBit(121), true); EXPECT_EQ(u4.getBit(122), false); EXPECT_EQ(u4.getBit(123), true); EXPECT_EQ(u4.getBit(124), false); EXPECT_EQ(u4.getBit(125), false); EXPECT_EQ(u4.getBit(126), false);
    EXPECT_EQ(u4.getBit(127), true);

    Aesi512 u5 = "212276592061158741188353364055816027621.";
    EXPECT_EQ(u5.getBit(0), true);
    EXPECT_EQ(u5.getBit(1), false); EXPECT_EQ(u5.getBit(2), true); EXPECT_EQ(u5.getBit(3), false); EXPECT_EQ(u5.getBit(4), false); EXPECT_EQ(u5.getBit(5), true); EXPECT_EQ(u5.getBit(6), true); EXPECT_EQ(u5.getBit(7), true); EXPECT_EQ(u5.getBit(8), true); EXPECT_EQ(u5.getBit(9), false);
    EXPECT_EQ(u5.getBit(10), true); EXPECT_EQ(u5.getBit(11), true); EXPECT_EQ(u5.getBit(12), true); EXPECT_EQ(u5.getBit(13), true); EXPECT_EQ(u5.getBit(14), false); EXPECT_EQ(u5.getBit(15), false); EXPECT_EQ(u5.getBit(16), false); EXPECT_EQ(u5.getBit(17), true); EXPECT_EQ(u5.getBit(18), true);
    EXPECT_EQ(u5.getBit(19), true); EXPECT_EQ(u5.getBit(20), true); EXPECT_EQ(u5.getBit(21), false); EXPECT_EQ(u5.getBit(22), true); EXPECT_EQ(u5.getBit(23), true); EXPECT_EQ(u5.getBit(24), true); EXPECT_EQ(u5.getBit(25), true); EXPECT_EQ(u5.getBit(26), true); EXPECT_EQ(u5.getBit(27), false);
    EXPECT_EQ(u5.getBit(28), false); EXPECT_EQ(u5.getBit(29), false); EXPECT_EQ(u5.getBit(30), false); EXPECT_EQ(u5.getBit(31), true); EXPECT_EQ(u5.getBit(32), true); EXPECT_EQ(u5.getBit(33), false); EXPECT_EQ(u5.getBit(34), false); EXPECT_EQ(u5.getBit(35), false); EXPECT_EQ(u5.getBit(36), true);
    EXPECT_EQ(u5.getBit(37), false); EXPECT_EQ(u5.getBit(38), true); EXPECT_EQ(u5.getBit(39), false); EXPECT_EQ(u5.getBit(40), true); EXPECT_EQ(u5.getBit(41), true); EXPECT_EQ(u5.getBit(42), true); EXPECT_EQ(u5.getBit(43), false); EXPECT_EQ(u5.getBit(44), false); EXPECT_EQ(u5.getBit(45), true);
    EXPECT_EQ(u5.getBit(46), false); EXPECT_EQ(u5.getBit(47), true); EXPECT_EQ(u5.getBit(48), false); EXPECT_EQ(u5.getBit(49), true); EXPECT_EQ(u5.getBit(50), true); EXPECT_EQ(u5.getBit(51), true); EXPECT_EQ(u5.getBit(52), true); EXPECT_EQ(u5.getBit(53), true); EXPECT_EQ(u5.getBit(54), true);
    EXPECT_EQ(u5.getBit(55), false); EXPECT_EQ(u5.getBit(56), false); EXPECT_EQ(u5.getBit(57), true); EXPECT_EQ(u5.getBit(58), true); EXPECT_EQ(u5.getBit(59), true); EXPECT_EQ(u5.getBit(60), true); EXPECT_EQ(u5.getBit(61), true); EXPECT_EQ(u5.getBit(62), false); EXPECT_EQ(u5.getBit(63), true);
    EXPECT_EQ(u5.getBit(64), true); EXPECT_EQ(u5.getBit(65), true); EXPECT_EQ(u5.getBit(66), true); EXPECT_EQ(u5.getBit(67), false); EXPECT_EQ(u5.getBit(68), false); EXPECT_EQ(u5.getBit(69), false); EXPECT_EQ(u5.getBit(70), false); EXPECT_EQ(u5.getBit(71), true); EXPECT_EQ(u5.getBit(72), true);
    EXPECT_EQ(u5.getBit(73), false); EXPECT_EQ(u5.getBit(74), true); EXPECT_EQ(u5.getBit(75), false); EXPECT_EQ(u5.getBit(76), true); EXPECT_EQ(u5.getBit(77), true); EXPECT_EQ(u5.getBit(78), true); EXPECT_EQ(u5.getBit(79), false); EXPECT_EQ(u5.getBit(80), false); EXPECT_EQ(u5.getBit(81), true);
    EXPECT_EQ(u5.getBit(82), false); EXPECT_EQ(u5.getBit(83), true); EXPECT_EQ(u5.getBit(84), false); EXPECT_EQ(u5.getBit(85), true); EXPECT_EQ(u5.getBit(86), false); EXPECT_EQ(u5.getBit(87), false); EXPECT_EQ(u5.getBit(88), true); EXPECT_EQ(u5.getBit(89), false); EXPECT_EQ(u5.getBit(90), false);
    EXPECT_EQ(u5.getBit(91), false); EXPECT_EQ(u5.getBit(92), false); EXPECT_EQ(u5.getBit(93), true); EXPECT_EQ(u5.getBit(94), false); EXPECT_EQ(u5.getBit(95), false); EXPECT_EQ(u5.getBit(96), true); EXPECT_EQ(u5.getBit(97), false); EXPECT_EQ(u5.getBit(98), false); EXPECT_EQ(u5.getBit(99), true);
    EXPECT_EQ(u5.getBit(100), false); EXPECT_EQ(u5.getBit(101), false); EXPECT_EQ(u5.getBit(102), false); EXPECT_EQ(u5.getBit(103), false); EXPECT_EQ(u5.getBit(104), false); EXPECT_EQ(u5.getBit(105), false); EXPECT_EQ(u5.getBit(106), true); EXPECT_EQ(u5.getBit(107), true); EXPECT_EQ(u5.getBit(108), true);
    EXPECT_EQ(u5.getBit(109), true); EXPECT_EQ(u5.getBit(110), true); EXPECT_EQ(u5.getBit(111), true); EXPECT_EQ(u5.getBit(112), false); EXPECT_EQ(u5.getBit(113), true); EXPECT_EQ(u5.getBit(114), false); EXPECT_EQ(u5.getBit(115), false); EXPECT_EQ(u5.getBit(116), true); EXPECT_EQ(u5.getBit(117), true);
    EXPECT_EQ(u5.getBit(118), false); EXPECT_EQ(u5.getBit(119), true); EXPECT_EQ(u5.getBit(120), true); EXPECT_EQ(u5.getBit(121), true); EXPECT_EQ(u5.getBit(122), true); EXPECT_EQ(u5.getBit(123), true); EXPECT_EQ(u5.getBit(124), true); EXPECT_EQ(u5.getBit(125), false); EXPECT_EQ(u5.getBit(126), false);
    EXPECT_EQ(u5.getBit(127), true);

    Aesi512 u6 = "29771602601962196722470093116016540190.";
    EXPECT_EQ(u6.getBit(0), false);
    EXPECT_EQ(u6.getBit(1), true); EXPECT_EQ(u6.getBit(2), true); EXPECT_EQ(u6.getBit(3), true); EXPECT_EQ(u6.getBit(4), true); EXPECT_EQ(u6.getBit(5), false); EXPECT_EQ(u6.getBit(6), false); EXPECT_EQ(u6.getBit(7), false); EXPECT_EQ(u6.getBit(8), false); EXPECT_EQ(u6.getBit(9), true);
    EXPECT_EQ(u6.getBit(10), false); EXPECT_EQ(u6.getBit(11), true); EXPECT_EQ(u6.getBit(12), true); EXPECT_EQ(u6.getBit(13), true); EXPECT_EQ(u6.getBit(14), false); EXPECT_EQ(u6.getBit(15), true); EXPECT_EQ(u6.getBit(16), true); EXPECT_EQ(u6.getBit(17), false); EXPECT_EQ(u6.getBit(18), true);
    EXPECT_EQ(u6.getBit(19), true); EXPECT_EQ(u6.getBit(20), false); EXPECT_EQ(u6.getBit(21), false); EXPECT_EQ(u6.getBit(22), false); EXPECT_EQ(u6.getBit(23), false); EXPECT_EQ(u6.getBit(24), false); EXPECT_EQ(u6.getBit(25), false); EXPECT_EQ(u6.getBit(26), true); EXPECT_EQ(u6.getBit(27), true);
    EXPECT_EQ(u6.getBit(28), true); EXPECT_EQ(u6.getBit(29), true); EXPECT_EQ(u6.getBit(30), true); EXPECT_EQ(u6.getBit(31), false); EXPECT_EQ(u6.getBit(32), true); EXPECT_EQ(u6.getBit(33), true); EXPECT_EQ(u6.getBit(34), false); EXPECT_EQ(u6.getBit(35), true); EXPECT_EQ(u6.getBit(36), true);
    EXPECT_EQ(u6.getBit(37), false); EXPECT_EQ(u6.getBit(38), false); EXPECT_EQ(u6.getBit(39), false); EXPECT_EQ(u6.getBit(40), false); EXPECT_EQ(u6.getBit(41), true); EXPECT_EQ(u6.getBit(42), true); EXPECT_EQ(u6.getBit(43), true); EXPECT_EQ(u6.getBit(44), true); EXPECT_EQ(u6.getBit(45), true);
    EXPECT_EQ(u6.getBit(46), false); EXPECT_EQ(u6.getBit(47), false); EXPECT_EQ(u6.getBit(48), true); EXPECT_EQ(u6.getBit(49), true); EXPECT_EQ(u6.getBit(50), false); EXPECT_EQ(u6.getBit(51), true); EXPECT_EQ(u6.getBit(52), false); EXPECT_EQ(u6.getBit(53), true); EXPECT_EQ(u6.getBit(54), false);
    EXPECT_EQ(u6.getBit(55), false); EXPECT_EQ(u6.getBit(56), true); EXPECT_EQ(u6.getBit(57), false); EXPECT_EQ(u6.getBit(58), true); EXPECT_EQ(u6.getBit(59), false); EXPECT_EQ(u6.getBit(60), true); EXPECT_EQ(u6.getBit(61), true); EXPECT_EQ(u6.getBit(62), true); EXPECT_EQ(u6.getBit(63), true);
    EXPECT_EQ(u6.getBit(64), false); EXPECT_EQ(u6.getBit(65), true); EXPECT_EQ(u6.getBit(66), true); EXPECT_EQ(u6.getBit(67), false); EXPECT_EQ(u6.getBit(68), false); EXPECT_EQ(u6.getBit(69), false); EXPECT_EQ(u6.getBit(70), false); EXPECT_EQ(u6.getBit(71), false); EXPECT_EQ(u6.getBit(72), true);
    EXPECT_EQ(u6.getBit(73), true); EXPECT_EQ(u6.getBit(74), true); EXPECT_EQ(u6.getBit(75), false); EXPECT_EQ(u6.getBit(76), false); EXPECT_EQ(u6.getBit(77), true); EXPECT_EQ(u6.getBit(78), true); EXPECT_EQ(u6.getBit(79), false); EXPECT_EQ(u6.getBit(80), false); EXPECT_EQ(u6.getBit(81), false);
    EXPECT_EQ(u6.getBit(82), false); EXPECT_EQ(u6.getBit(83), false); EXPECT_EQ(u6.getBit(84), true); EXPECT_EQ(u6.getBit(85), true); EXPECT_EQ(u6.getBit(86), false); EXPECT_EQ(u6.getBit(87), false); EXPECT_EQ(u6.getBit(88), false); EXPECT_EQ(u6.getBit(89), true); EXPECT_EQ(u6.getBit(90), true);
    EXPECT_EQ(u6.getBit(91), true); EXPECT_EQ(u6.getBit(92), false); EXPECT_EQ(u6.getBit(93), false); EXPECT_EQ(u6.getBit(94), false); EXPECT_EQ(u6.getBit(95), false); EXPECT_EQ(u6.getBit(96), false); EXPECT_EQ(u6.getBit(97), true); EXPECT_EQ(u6.getBit(98), true); EXPECT_EQ(u6.getBit(99), false);
    EXPECT_EQ(u6.getBit(100), true); EXPECT_EQ(u6.getBit(101), false); EXPECT_EQ(u6.getBit(102), true); EXPECT_EQ(u6.getBit(103), false); EXPECT_EQ(u6.getBit(104), true); EXPECT_EQ(u6.getBit(105), false); EXPECT_EQ(u6.getBit(106), true); EXPECT_EQ(u6.getBit(107), true); EXPECT_EQ(u6.getBit(108), false);
    EXPECT_EQ(u6.getBit(109), false); EXPECT_EQ(u6.getBit(110), true); EXPECT_EQ(u6.getBit(111), true); EXPECT_EQ(u6.getBit(112), true); EXPECT_EQ(u6.getBit(113), false); EXPECT_EQ(u6.getBit(114), true); EXPECT_EQ(u6.getBit(115), false); EXPECT_EQ(u6.getBit(116), false); EXPECT_EQ(u6.getBit(117), true);
    EXPECT_EQ(u6.getBit(118), true); EXPECT_EQ(u6.getBit(119), false); EXPECT_EQ(u6.getBit(120), false); EXPECT_EQ(u6.getBit(121), true); EXPECT_EQ(u6.getBit(122), true); EXPECT_EQ(u6.getBit(123), false); EXPECT_EQ(u6.getBit(124), true);

    Aesi512 u7 = "37894586767282899313108693355181559199.";
    EXPECT_EQ(u7.getBit(0), true);
    EXPECT_EQ(u7.getBit(1), true); EXPECT_EQ(u7.getBit(2), true); EXPECT_EQ(u7.getBit(3), true); EXPECT_EQ(u7.getBit(4), true); EXPECT_EQ(u7.getBit(5), false); EXPECT_EQ(u7.getBit(6), false); EXPECT_EQ(u7.getBit(7), true); EXPECT_EQ(u7.getBit(8), true); EXPECT_EQ(u7.getBit(9), false);
    EXPECT_EQ(u7.getBit(10), true); EXPECT_EQ(u7.getBit(11), true); EXPECT_EQ(u7.getBit(12), false); EXPECT_EQ(u7.getBit(13), false); EXPECT_EQ(u7.getBit(14), true); EXPECT_EQ(u7.getBit(15), true); EXPECT_EQ(u7.getBit(16), false); EXPECT_EQ(u7.getBit(17), true); EXPECT_EQ(u7.getBit(18), true);
    EXPECT_EQ(u7.getBit(19), true); EXPECT_EQ(u7.getBit(20), false); EXPECT_EQ(u7.getBit(21), true); EXPECT_EQ(u7.getBit(22), false); EXPECT_EQ(u7.getBit(23), false); EXPECT_EQ(u7.getBit(24), false); EXPECT_EQ(u7.getBit(25), false); EXPECT_EQ(u7.getBit(26), false); EXPECT_EQ(u7.getBit(27), false);
    EXPECT_EQ(u7.getBit(28), false); EXPECT_EQ(u7.getBit(29), false); EXPECT_EQ(u7.getBit(30), true); EXPECT_EQ(u7.getBit(31), false); EXPECT_EQ(u7.getBit(32), true); EXPECT_EQ(u7.getBit(33), false); EXPECT_EQ(u7.getBit(34), true); EXPECT_EQ(u7.getBit(35), true); EXPECT_EQ(u7.getBit(36), true);
    EXPECT_EQ(u7.getBit(37), true); EXPECT_EQ(u7.getBit(38), false); EXPECT_EQ(u7.getBit(39), false); EXPECT_EQ(u7.getBit(40), false); EXPECT_EQ(u7.getBit(41), false); EXPECT_EQ(u7.getBit(42), false); EXPECT_EQ(u7.getBit(43), false); EXPECT_EQ(u7.getBit(44), false); EXPECT_EQ(u7.getBit(45), false);
    EXPECT_EQ(u7.getBit(46), true); EXPECT_EQ(u7.getBit(47), false); EXPECT_EQ(u7.getBit(48), true); EXPECT_EQ(u7.getBit(49), true); EXPECT_EQ(u7.getBit(50), true); EXPECT_EQ(u7.getBit(51), true); EXPECT_EQ(u7.getBit(52), false); EXPECT_EQ(u7.getBit(53), false); EXPECT_EQ(u7.getBit(54), true);
    EXPECT_EQ(u7.getBit(55), true); EXPECT_EQ(u7.getBit(56), false); EXPECT_EQ(u7.getBit(57), false); EXPECT_EQ(u7.getBit(58), true); EXPECT_EQ(u7.getBit(59), false); EXPECT_EQ(u7.getBit(60), true); EXPECT_EQ(u7.getBit(61), true); EXPECT_EQ(u7.getBit(62), false); EXPECT_EQ(u7.getBit(63), true);
    EXPECT_EQ(u7.getBit(64), true); EXPECT_EQ(u7.getBit(65), false); EXPECT_EQ(u7.getBit(66), false); EXPECT_EQ(u7.getBit(67), true); EXPECT_EQ(u7.getBit(68), false); EXPECT_EQ(u7.getBit(69), true); EXPECT_EQ(u7.getBit(70), true); EXPECT_EQ(u7.getBit(71), true); EXPECT_EQ(u7.getBit(72), false);
    EXPECT_EQ(u7.getBit(73), false); EXPECT_EQ(u7.getBit(74), true); EXPECT_EQ(u7.getBit(75), true); EXPECT_EQ(u7.getBit(76), false); EXPECT_EQ(u7.getBit(77), true); EXPECT_EQ(u7.getBit(78), true); EXPECT_EQ(u7.getBit(79), true); EXPECT_EQ(u7.getBit(80), false); EXPECT_EQ(u7.getBit(81), false);
    EXPECT_EQ(u7.getBit(82), true); EXPECT_EQ(u7.getBit(83), true); EXPECT_EQ(u7.getBit(84), false); EXPECT_EQ(u7.getBit(85), false); EXPECT_EQ(u7.getBit(86), true); EXPECT_EQ(u7.getBit(87), true); EXPECT_EQ(u7.getBit(88), false); EXPECT_EQ(u7.getBit(89), true); EXPECT_EQ(u7.getBit(90), false);
    EXPECT_EQ(u7.getBit(91), false); EXPECT_EQ(u7.getBit(92), true); EXPECT_EQ(u7.getBit(93), false); EXPECT_EQ(u7.getBit(94), true); EXPECT_EQ(u7.getBit(95), false); EXPECT_EQ(u7.getBit(96), false); EXPECT_EQ(u7.getBit(97), false); EXPECT_EQ(u7.getBit(98), false); EXPECT_EQ(u7.getBit(99), false);
    EXPECT_EQ(u7.getBit(100), false); EXPECT_EQ(u7.getBit(101), true); EXPECT_EQ(u7.getBit(102), true); EXPECT_EQ(u7.getBit(103), false); EXPECT_EQ(u7.getBit(104), true); EXPECT_EQ(u7.getBit(105), true); EXPECT_EQ(u7.getBit(106), false); EXPECT_EQ(u7.getBit(107), true); EXPECT_EQ(u7.getBit(108), true);
    EXPECT_EQ(u7.getBit(109), true); EXPECT_EQ(u7.getBit(110), false); EXPECT_EQ(u7.getBit(111), false); EXPECT_EQ(u7.getBit(112), false); EXPECT_EQ(u7.getBit(113), true); EXPECT_EQ(u7.getBit(114), false); EXPECT_EQ(u7.getBit(115), false); EXPECT_EQ(u7.getBit(116), false); EXPECT_EQ(u7.getBit(117), false);
    EXPECT_EQ(u7.getBit(118), false); EXPECT_EQ(u7.getBit(119), true); EXPECT_EQ(u7.getBit(120), false); EXPECT_EQ(u7.getBit(121), false); EXPECT_EQ(u7.getBit(122), true); EXPECT_EQ(u7.getBit(123), true); EXPECT_EQ(u7.getBit(124), true);

    Aesi512 u8 = "132933768221848679423944309756035959186.";
    EXPECT_EQ(u8.getBit(0), false);
    EXPECT_EQ(u8.getBit(1), true); EXPECT_EQ(u8.getBit(2), false); EXPECT_EQ(u8.getBit(3), false); EXPECT_EQ(u8.getBit(4), true); EXPECT_EQ(u8.getBit(5), false); EXPECT_EQ(u8.getBit(6), false); EXPECT_EQ(u8.getBit(7), true); EXPECT_EQ(u8.getBit(8), true); EXPECT_EQ(u8.getBit(9), false);
    EXPECT_EQ(u8.getBit(10), false); EXPECT_EQ(u8.getBit(11), true); EXPECT_EQ(u8.getBit(12), false); EXPECT_EQ(u8.getBit(13), false); EXPECT_EQ(u8.getBit(14), false); EXPECT_EQ(u8.getBit(15), true); EXPECT_EQ(u8.getBit(16), true); EXPECT_EQ(u8.getBit(17), false); EXPECT_EQ(u8.getBit(18), false);
    EXPECT_EQ(u8.getBit(19), false); EXPECT_EQ(u8.getBit(20), true); EXPECT_EQ(u8.getBit(21), false); EXPECT_EQ(u8.getBit(22), false); EXPECT_EQ(u8.getBit(23), false); EXPECT_EQ(u8.getBit(24), true); EXPECT_EQ(u8.getBit(25), true); EXPECT_EQ(u8.getBit(26), true); EXPECT_EQ(u8.getBit(27), false);
    EXPECT_EQ(u8.getBit(28), true); EXPECT_EQ(u8.getBit(29), true); EXPECT_EQ(u8.getBit(30), true); EXPECT_EQ(u8.getBit(31), true); EXPECT_EQ(u8.getBit(32), false); EXPECT_EQ(u8.getBit(33), true); EXPECT_EQ(u8.getBit(34), true); EXPECT_EQ(u8.getBit(35), true); EXPECT_EQ(u8.getBit(36), false);
    EXPECT_EQ(u8.getBit(37), false); EXPECT_EQ(u8.getBit(38), false); EXPECT_EQ(u8.getBit(39), true); EXPECT_EQ(u8.getBit(40), true); EXPECT_EQ(u8.getBit(41), false); EXPECT_EQ(u8.getBit(42), true); EXPECT_EQ(u8.getBit(43), true); EXPECT_EQ(u8.getBit(44), true); EXPECT_EQ(u8.getBit(45), true);
    EXPECT_EQ(u8.getBit(46), false); EXPECT_EQ(u8.getBit(47), true); EXPECT_EQ(u8.getBit(48), false); EXPECT_EQ(u8.getBit(49), true); EXPECT_EQ(u8.getBit(50), false); EXPECT_EQ(u8.getBit(51), false); EXPECT_EQ(u8.getBit(52), true); EXPECT_EQ(u8.getBit(53), false); EXPECT_EQ(u8.getBit(54), true);
    EXPECT_EQ(u8.getBit(55), true); EXPECT_EQ(u8.getBit(56), true); EXPECT_EQ(u8.getBit(57), true); EXPECT_EQ(u8.getBit(58), false); EXPECT_EQ(u8.getBit(59), false); EXPECT_EQ(u8.getBit(60), false); EXPECT_EQ(u8.getBit(61), true); EXPECT_EQ(u8.getBit(62), true); EXPECT_EQ(u8.getBit(63), false);
    EXPECT_EQ(u8.getBit(64), true); EXPECT_EQ(u8.getBit(65), true); EXPECT_EQ(u8.getBit(66), false); EXPECT_EQ(u8.getBit(67), false); EXPECT_EQ(u8.getBit(68), false); EXPECT_EQ(u8.getBit(69), false); EXPECT_EQ(u8.getBit(70), true); EXPECT_EQ(u8.getBit(71), true); EXPECT_EQ(u8.getBit(72), false);
    EXPECT_EQ(u8.getBit(73), true); EXPECT_EQ(u8.getBit(74), true); EXPECT_EQ(u8.getBit(75), true); EXPECT_EQ(u8.getBit(76), true); EXPECT_EQ(u8.getBit(77), true); EXPECT_EQ(u8.getBit(78), true); EXPECT_EQ(u8.getBit(79), true); EXPECT_EQ(u8.getBit(80), false); EXPECT_EQ(u8.getBit(81), true);
    EXPECT_EQ(u8.getBit(82), false); EXPECT_EQ(u8.getBit(83), true); EXPECT_EQ(u8.getBit(84), false); EXPECT_EQ(u8.getBit(85), true); EXPECT_EQ(u8.getBit(86), false); EXPECT_EQ(u8.getBit(87), false); EXPECT_EQ(u8.getBit(88), false); EXPECT_EQ(u8.getBit(89), true); EXPECT_EQ(u8.getBit(90), true);
    EXPECT_EQ(u8.getBit(91), true); EXPECT_EQ(u8.getBit(92), true); EXPECT_EQ(u8.getBit(93), true); EXPECT_EQ(u8.getBit(94), false); EXPECT_EQ(u8.getBit(95), true); EXPECT_EQ(u8.getBit(96), true); EXPECT_EQ(u8.getBit(97), true); EXPECT_EQ(u8.getBit(98), false); EXPECT_EQ(u8.getBit(99), true);
    EXPECT_EQ(u8.getBit(100), false); EXPECT_EQ(u8.getBit(101), false); EXPECT_EQ(u8.getBit(102), true); EXPECT_EQ(u8.getBit(103), true); EXPECT_EQ(u8.getBit(104), false); EXPECT_EQ(u8.getBit(105), false); EXPECT_EQ(u8.getBit(106), true); EXPECT_EQ(u8.getBit(107), true); EXPECT_EQ(u8.getBit(108), true);
    EXPECT_EQ(u8.getBit(109), false); EXPECT_EQ(u8.getBit(110), false); EXPECT_EQ(u8.getBit(111), false); EXPECT_EQ(u8.getBit(112), false); EXPECT_EQ(u8.getBit(113), true); EXPECT_EQ(u8.getBit(114), false); EXPECT_EQ(u8.getBit(115), false); EXPECT_EQ(u8.getBit(116), false); EXPECT_EQ(u8.getBit(117), false);
    EXPECT_EQ(u8.getBit(118), false); EXPECT_EQ(u8.getBit(119), false); EXPECT_EQ(u8.getBit(120), false); EXPECT_EQ(u8.getBit(121), false); EXPECT_EQ(u8.getBit(122), true); EXPECT_EQ(u8.getBit(123), false); EXPECT_EQ(u8.getBit(124), false); EXPECT_EQ(u8.getBit(125), true); EXPECT_EQ(u8.getBit(126), true);


    Aesi512 u9 = "112312108742573550610800981189693677283.";
    EXPECT_EQ(u9.getBit(0), true);
    EXPECT_EQ(u9.getBit(1), true); EXPECT_EQ(u9.getBit(2), false); EXPECT_EQ(u9.getBit(3), false); EXPECT_EQ(u9.getBit(4), false); EXPECT_EQ(u9.getBit(5), true); EXPECT_EQ(u9.getBit(6), true); EXPECT_EQ(u9.getBit(7), true); EXPECT_EQ(u9.getBit(8), false); EXPECT_EQ(u9.getBit(9), true);
    EXPECT_EQ(u9.getBit(10), true); EXPECT_EQ(u9.getBit(11), true); EXPECT_EQ(u9.getBit(12), true); EXPECT_EQ(u9.getBit(13), false); EXPECT_EQ(u9.getBit(14), false); EXPECT_EQ(u9.getBit(15), false); EXPECT_EQ(u9.getBit(16), false); EXPECT_EQ(u9.getBit(17), true); EXPECT_EQ(u9.getBit(18), true);
    EXPECT_EQ(u9.getBit(19), true); EXPECT_EQ(u9.getBit(20), true); EXPECT_EQ(u9.getBit(21), true); EXPECT_EQ(u9.getBit(22), true); EXPECT_EQ(u9.getBit(23), false); EXPECT_EQ(u9.getBit(24), false); EXPECT_EQ(u9.getBit(25), false); EXPECT_EQ(u9.getBit(26), false); EXPECT_EQ(u9.getBit(27), true);
    EXPECT_EQ(u9.getBit(28), false); EXPECT_EQ(u9.getBit(29), true); EXPECT_EQ(u9.getBit(30), true); EXPECT_EQ(u9.getBit(31), true); EXPECT_EQ(u9.getBit(32), true); EXPECT_EQ(u9.getBit(33), true); EXPECT_EQ(u9.getBit(34), false); EXPECT_EQ(u9.getBit(35), false); EXPECT_EQ(u9.getBit(36), false);
    EXPECT_EQ(u9.getBit(37), false); EXPECT_EQ(u9.getBit(38), true); EXPECT_EQ(u9.getBit(39), true); EXPECT_EQ(u9.getBit(40), true); EXPECT_EQ(u9.getBit(41), false); EXPECT_EQ(u9.getBit(42), false); EXPECT_EQ(u9.getBit(43), false); EXPECT_EQ(u9.getBit(44), false); EXPECT_EQ(u9.getBit(45), true);
    EXPECT_EQ(u9.getBit(46), false); EXPECT_EQ(u9.getBit(47), true); EXPECT_EQ(u9.getBit(48), true); EXPECT_EQ(u9.getBit(49), false); EXPECT_EQ(u9.getBit(50), false); EXPECT_EQ(u9.getBit(51), false); EXPECT_EQ(u9.getBit(52), false); EXPECT_EQ(u9.getBit(53), true); EXPECT_EQ(u9.getBit(54), false);
    EXPECT_EQ(u9.getBit(55), false); EXPECT_EQ(u9.getBit(56), false); EXPECT_EQ(u9.getBit(57), true); EXPECT_EQ(u9.getBit(58), true); EXPECT_EQ(u9.getBit(59), false); EXPECT_EQ(u9.getBit(60), true); EXPECT_EQ(u9.getBit(61), true); EXPECT_EQ(u9.getBit(62), true); EXPECT_EQ(u9.getBit(63), false);
    EXPECT_EQ(u9.getBit(64), true); EXPECT_EQ(u9.getBit(65), false); EXPECT_EQ(u9.getBit(66), false); EXPECT_EQ(u9.getBit(67), false); EXPECT_EQ(u9.getBit(68), true); EXPECT_EQ(u9.getBit(69), false); EXPECT_EQ(u9.getBit(70), false); EXPECT_EQ(u9.getBit(71), true); EXPECT_EQ(u9.getBit(72), false);
    EXPECT_EQ(u9.getBit(73), true); EXPECT_EQ(u9.getBit(74), true); EXPECT_EQ(u9.getBit(75), true); EXPECT_EQ(u9.getBit(76), true); EXPECT_EQ(u9.getBit(77), true); EXPECT_EQ(u9.getBit(78), false); EXPECT_EQ(u9.getBit(79), false); EXPECT_EQ(u9.getBit(80), false); EXPECT_EQ(u9.getBit(81), false);
    EXPECT_EQ(u9.getBit(82), true); EXPECT_EQ(u9.getBit(83), false); EXPECT_EQ(u9.getBit(84), true); EXPECT_EQ(u9.getBit(85), true); EXPECT_EQ(u9.getBit(86), false); EXPECT_EQ(u9.getBit(87), true); EXPECT_EQ(u9.getBit(88), true); EXPECT_EQ(u9.getBit(89), false); EXPECT_EQ(u9.getBit(90), false);
    EXPECT_EQ(u9.getBit(91), false); EXPECT_EQ(u9.getBit(92), true); EXPECT_EQ(u9.getBit(93), false); EXPECT_EQ(u9.getBit(94), true); EXPECT_EQ(u9.getBit(95), false); EXPECT_EQ(u9.getBit(96), false); EXPECT_EQ(u9.getBit(97), false); EXPECT_EQ(u9.getBit(98), true); EXPECT_EQ(u9.getBit(99), true);
    EXPECT_EQ(u9.getBit(100), true); EXPECT_EQ(u9.getBit(101), true); EXPECT_EQ(u9.getBit(102), true); EXPECT_EQ(u9.getBit(103), false); EXPECT_EQ(u9.getBit(104), false); EXPECT_EQ(u9.getBit(105), true); EXPECT_EQ(u9.getBit(106), true); EXPECT_EQ(u9.getBit(107), false); EXPECT_EQ(u9.getBit(108), false);
    EXPECT_EQ(u9.getBit(109), false); EXPECT_EQ(u9.getBit(110), false); EXPECT_EQ(u9.getBit(111), true); EXPECT_EQ(u9.getBit(112), false); EXPECT_EQ(u9.getBit(113), true); EXPECT_EQ(u9.getBit(114), true); EXPECT_EQ(u9.getBit(115), true); EXPECT_EQ(u9.getBit(116), true); EXPECT_EQ(u9.getBit(117), true);
    EXPECT_EQ(u9.getBit(118), true); EXPECT_EQ(u9.getBit(119), false); EXPECT_EQ(u9.getBit(120), false); EXPECT_EQ(u9.getBit(121), false); EXPECT_EQ(u9.getBit(122), true); EXPECT_EQ(u9.getBit(123), false); EXPECT_EQ(u9.getBit(124), true); EXPECT_EQ(u9.getBit(125), false); EXPECT_EQ(u9.getBit(126), true);

    Aesi512 o0{}; o0.setBit(36, true); o0.setBit(233, true); o0.setBit(68, true);
    EXPECT_EQ(o0, "13803492693581127574869511724554050904902217944341068258230296519901184.");
    o0.setBit(36, false); o0.setBit(233, false); o0.setBit(68, false); EXPECT_EQ(o0, 0);

    Aesi512 o1{}; o1.setBit(40, true); o1.setBit(43, true); o1.setBit(101, true);
    EXPECT_EQ(o1, "2535301200456458812889011060736.");
    o1.setBit(40, false); o1.setBit(43, false); o1.setBit(101, false); EXPECT_EQ(o1, 0);

    Aesi512 o2{}; o2.setBit(467, true); o2.setBit(34, true); o2.setBit(143, true);
    EXPECT_EQ(o2, "381072821083495145432323880589986121307201921712032611188861933548019011086397170424842053596617683411093660193217652664276125542861053624320.");
    o2.setBit(467, false); o2.setBit(34, false); o2.setBit(143, false); EXPECT_EQ(o2, 0);

    Aesi512 o3{}; o3.setBit(50, true); o3.setBit(122, true); o3.setBit(142, true);
    EXPECT_EQ(o3, "5575191616544638925047421184516231404716032.");
    o3.setBit(50, false); o3.setBit(122, false); o3.setBit(142, false); EXPECT_EQ(o3, 0);

    Aesi512 o4{}; o4.setBit(82, true); o4.setBit(393, true); o4.setBit(263, true);
    EXPECT_EQ(o4, "20173827172553973356686868531273530268215647893900685167003743309054921599211790391081306314702963082124464928329826304.");
    o4.setBit(82, false); o4.setBit(393, false); o4.setBit(263, false); EXPECT_EQ(o4, 0);

    Aesi512 o5{}; o5.setBit(15, true); o5.setBit(95, true); o5.setBit(300, true);
    EXPECT_EQ(o5, "2037035976334486086268445688409378161051468393665936250636140488968462556895505502955405312.");
    o5.setBit(15, false); o5.setBit(95, false); o5.setBit(300, false); EXPECT_EQ(o5, 0);

    Aesi512 o6{}; o6.setBit(372, true); o6.setBit(454, true); o6.setBit(95, true);
    EXPECT_EQ(o6, "46517678354918840995156733324462709240253948519300880307139836878356634767735657509940046641609035065324985849726214028890990919808974848.");
    o6.setBit(372, false); o6.setBit(454, false); o6.setBit(95, false); EXPECT_EQ(o6, 0);

    Aesi512 o7{}; o7.setBit(207, true); o7.setBit(117, true); o7.setBit(413, true);
    EXPECT_EQ(o7, "21153791001287955166461289857048673274508949854856999017108761654469054984712718335777202719893201711079386101974112919355392.");
    o7.setBit(207, false); o7.setBit(117, false); o7.setBit(413, false); EXPECT_EQ(o7, 0);

    Aesi512 o8{}; o8.setBit(467, true); o8.setBit(453, true); o8.setBit(159, true);
    EXPECT_EQ(o8, "381096079922672604852821458951838537452301238235574605366790941234392791543616799158388491710971263513484456787699324830469003835743202705408.");
    o8.setBit(467, false); o8.setBit(453, false); o8.setBit(159, false); EXPECT_EQ(o8, 0);

    Aesi512 o9{}; o9.setBit(58, true); o9.setBit(299, true); o9.setBit(348, true);
    EXPECT_EQ(o9, "573374653997518896420693391068564869421985451981150806668594042814856037092345879208154808176260665049088.");
    o9.setBit(58, false); o9.setBit(299, false); o9.setBit(348, false); EXPECT_EQ(o9, 0);

    Aesi512 o10{}; o10.setBit(19, true); o10.setBit(138, true); o10.setBit(475, true);
    EXPECT_EQ(o10, "97554642197374757230674913431036447054643691958280348464348654988292866838117675628759565720734124099093040741270997952069244837987889564876800.");
    o10.setBit(19, false); o10.setBit(138, false); o10.setBit(475, false); EXPECT_EQ(o10, 0);

    Aesi512 o11{}; o11.setBit(77, true); o11.setBit(12, true); o11.setBit(475, true);
    EXPECT_EQ(o11, "97554642197374757230674913431036447054643691958280348464348654988292866838117675628759565720734124098744591597543956965633864967429587562663936.");
    o11.setBit(77, false); o11.setBit(12, false); o11.setBit(475, false); EXPECT_EQ(o11, 0);

    Aesi512 o12{}; o12.setBit(381, true); o12.setBit(286, true); o12.setBit(43, true);
    EXPECT_EQ(o12, "4925250774549309901534880012642282534737414069347026395530241885555829259136488255786899276083904329362757935693824.");
    o12.setBit(381, false); o12.setBit(286, false); o12.setBit(43, false); EXPECT_EQ(o12, 0);

    Aesi512 o13{}; o13.setBit(337, true); o13.setBit(97, true); o13.setBit(201, true);
    EXPECT_EQ(o13, "279968092772225526319680285071055534765208901030419709843049721544658666381181487607125927431709392896.");
    o13.setBit(337, false); o13.setBit(97, false); o13.setBit(201, false); EXPECT_EQ(o13, 0);

    Aesi512 o14{}; o14.setBit(194, true); o14.setBit(425, true); o14.setBit(218, true);
    EXPECT_EQ(o14, "86645927941275464361825443254471365732388658605494267974077487315456107651561489058394366268593783921789521354162084493037404160.");
    o14.setBit(194, false); o14.setBit(425, false); o14.setBit(218, false); EXPECT_EQ(o14, 0);

    Aesi512 o15{}; o15.setBit(344, true); o15.setBit(130, true); o15.setBit(455, true);
    EXPECT_EQ(o15, "93035356709837681990313447409664616233181969012844080635107234590230278824842889940145012891187353173577009331931050723030751081249374208.");
    o15.setBit(344, false); o15.setBit(130, false); o15.setBit(455, false); EXPECT_EQ(o15, 0);

    Aesi512 o16{}; o16.setBit(361, true); o16.setBit(305, true); o16.setBit(498, true);
    EXPECT_EQ(o16, "818347651974035467503297424206899788054165208595931745037343806136469731329983980347944439822371100858262045409960603463570581736060544575336064483328.");
    o16.setBit(361, false); o16.setBit(305, false); o16.setBit(498, false); EXPECT_EQ(o16, 0);

    Aesi512 o17{}; o17.setBit(239, true); o17.setBit(143, true); o17.setBit(338, true);
    EXPECT_EQ(o17, "559936185544451052639360570142994493062800566473454032475379884871460975055048813683416500599301079040.");
    o17.setBit(239, false); o17.setBit(143, false); o17.setBit(338, false); EXPECT_EQ(o17, 0);

    Aesi512 o18{}; o18.setBit(429, true); o18.setBit(244, true); o18.setBit(150, true);
    EXPECT_EQ(o18, "1386334847060407429789207092071541851718218537687908287613509343343764803176145571533301146998015491279763794780096980271551741952.");
    o18.setBit(429, false); o18.setBit(244, false); o18.setBit(150, false); EXPECT_EQ(o18, 0);

    Aesi512 o19{}; o19.setBit(342, true); o19.setBit(286, true); o19.setBit(46, true);
    EXPECT_EQ(o19, "8958978968711216966560578224720437651332144025643808164713994103192096844836638004813384296588516524032.");
    o19.setBit(342, false); o19.setBit(286, false); o19.setBit(46, false); EXPECT_EQ(o19, 0);

#ifdef NDEBUG
    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(),
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}

TEST(Bitwise, GetSetByte) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi512 m {}; uint8_t byte {};
    m = "25500030152205279384249040235552387539274952473526657228704168006265771.";
    m.setByte(3, 130); byte = m.getByte(8);
    EXPECT_EQ(m, "25500030152205279384249040235552387539274952473526657228704168543136683."); EXPECT_EQ(byte, 114);
    m = "11973067663711446855645029841783928448494082142254567843368572898704850322653440454347984366272327292124441799397540746.";
    m.setByte(48, 17); byte = m.getByte(45);
    EXPECT_EQ(m, "10791007477819612479276658638779620034341689964140604443330124096577478669508524136005556378625779822900242959688336266."); EXPECT_EQ(byte, 31);
    m = "38766513705573349407986864851812593727672370420584096379717774965542388658671188340380285228435220355317930507752476255589469641877980698898243.";
    m.setByte(48, 225); byte = m.getByte(18);
    EXPECT_EQ(m, "38766513705573349407986867176530959314946643945047462288190989465247005616132541749329596078932804873653356581194185293799492449469032127000387."); EXPECT_EQ(byte, 51);
    m = "10209847901010179163405590740748846728932893765953460450856065623952064384871938528316302635587242810753539861219085459173875586170032846039368061.";
    m.setByte(48, 33); byte = m.getByte(13);
    EXPECT_EQ(m, "10209847901010179163405590732277415396708080735313466829325188655859920441721867494707419553674412629881639579765018214373618723396607828123402621."); EXPECT_EQ(byte, 46);
    m = "2015189327385899601434050704482979709608324068009805268669057088954254915929141776282171679180249471919481.";
    m.setByte(24, 3); byte = m.getByte(5);
    EXPECT_EQ(m, "2015189327385899601434050704482979709608324067708504385370496412290137023615173788309258617845975815300473."); EXPECT_EQ(byte, 37);
    m = "18718590587298140828829309254524948374429187392444911099799871459576073231656356618694044469727252605534345383658571755378738.";
    m.setByte(16, 121); byte = m.getByte(46);
    EXPECT_EQ(m, "18718590587298140828829309254524948374429187392444911099799871459576073231656356618674988657179680051580396405642392735537202."); EXPECT_EQ(byte, 241);
    m = "525022161679952953670395509613537510460871990725983691373975785604101648001280746454074607372686170831287.";
    m.setByte(8, 148); byte = m.getByte(5);
    EXPECT_EQ(m, "525022161679952953670395509613537510460871990725983691373975785604101648001280746452580421102715697150391."); EXPECT_EQ(byte, 248);
    m = "196206845334197451698107697668285779009914678252827564371272562250095484.";
    m.setByte(20, 66); byte = m.getByte(25);
    EXPECT_EQ(m, "196206845334197451698049237602792542893186530859518913050486324948376444."); EXPECT_EQ(byte, 93);
    m = "3784669068155155387740712606655250521196702590402769498.";
    m.setByte(12, 183); byte = m.getByte(0);
    EXPECT_EQ(m, "3784669068155155387740712289742600464139352216226968154."); EXPECT_EQ(byte, 90);
    m = "1267217244931949321246003980709992388302511962449639931433176546307921895020482804361903101176546617585258398898210366900168883268646483200361361.";
    m.setByte(40, 119); byte = m.getByte(47);
    EXPECT_EQ(m, "1267217244931949321246003980709992388302511962310800774098317390952245484119461916912727307223364187538700694389766084724427247517153947549483921."); EXPECT_EQ(byte, 57);
    m = "4660542240303479918651015069160831470157382095267786598931088090234791028075350804679257254148437711462858566496578524734644482704199701200326281120262136.";
    m.setByte(34, 56); byte = m.getByte(43);
    EXPECT_EQ(m, "4660542240303479918651015069160831470157382095267786598931088090234790792830289636719877572494847432052365965267625358264624331117540709379885494552045560."); EXPECT_EQ(byte, 222);
    m = "5882861481174708272515590658267037896829231252803932767344477541714430028887826331610199811756703362720569099.";
    m.setByte(30, 218); byte = m.getByte(30);
    EXPECT_EQ(m, "5882861481174708272515590658267038241364408884588877036087490186583540615247186222355896645469912614781425419."); EXPECT_EQ(byte, 218);
    m = "654607854758614823699478186797673005680933847718935115956157419943376045451008292263708009790424145827279730093893803046889111572549.";
    m.setByte(25, 231); byte = m.getByte(14);
    EXPECT_EQ(m, "654607854758614823699478186797673005680933847718935115956157419943376075982831133184523245087703900309369178015750684919952982298693."); EXPECT_EQ(byte, 16);
    m = "51391635968034080968235143931943244814854725581.";
    m.setByte(8, 199); byte = m.getByte(10);
    EXPECT_EQ(m, "51391635968034080968235147141676713640316706765."); EXPECT_EQ(byte, 170);
    m = "2703210061869264465928657652842835549518302189457245330338214557734368232872577539088199704199045756151335.";
    m.setByte(42, 81); byte = m.getByte(5);
    EXPECT_EQ(m, "2699150524524067195797022288709305244264206706993507528056208327488871360106063497374696863742490968253991."); EXPECT_EQ(byte, 121);
    m = "53886330462185772711467215637667078502272024181372945896827325446926247886414947400664557577708271641134253006435222129616566692983430585.";
    m.setByte(26, 202); byte = m.getByte(41);
    EXPECT_EQ(m, "53886330462185772711467215637667078502272024181372945896827325446926247865023388155488879029693672267888696441659655876380028469451513273."); EXPECT_EQ(byte, 171);
    m = "420282292186490072823697164798274997441726315143171645188942493901333371793022513254973822464120875452926.";
    m.setByte(21, 244); byte = m.getByte(29);
    EXPECT_EQ(m, "420282292186490072823697164798274997441726315143171725255848193437518842663692388783822771294729309850110."); EXPECT_EQ(byte, 57);
    m = "641149140161898337883774982032843409818313787405228991243553210071105423423222991887451.";
    m.setByte(32, 187); byte = m.getByte(1);
    EXPECT_EQ(m, "641149142014571765680834108809979169957320313057548745893802234702426767549833066126427."); EXPECT_EQ(byte, 28);
    m = "7095872742782960838669576139031373860195307904778004233328801671985844754.";
    m.setByte(20, 15); byte = m.getByte(13);
    EXPECT_EQ(m, "7095872742782960838669245839661337076135793872005810353366359431231132178."); EXPECT_EQ(byte, 108);
    m = "69369513050798592599018686684274045923503120.";
    m.setByte(10, 242); byte = m.getByte(15);
    EXPECT_EQ(m, "69369513050798592852893108803346172611800080."); EXPECT_EQ(byte, 173);
    m = "2962611506976377741074982101223066205248126798938955893735246608385573920155633.";
    m.setByte(19, 133); byte = m.getByte(4);
    EXPECT_EQ(m, "2962611506976377741074982101223066205248126798938955893735246608385573920155633."); EXPECT_EQ(byte, 172);
    m = "1895087054472930971347434110566541886052466406275548191698994019506228291907649087325211597462039.";
    m.setByte(6, 27); byte = m.getByte(33);
    EXPECT_EQ(m, "1895087054472930971347434110566541886052466406275548191698994019506228291907649055237064252447255."); EXPECT_EQ(byte, 138);
    m = "3727710584317747315864044339393878980224545305290603645482373789133370032867341032647579386.";
    m.setByte(29, 225); byte = m.getByte(27);
    EXPECT_EQ(m, "3727710584317747317099456935469389898175366604638191201471122295151869226241432868707653370."); EXPECT_EQ(byte, 174);
    m = "160269685900709903017808373923991600665930.";
    m.setByte(12, 89); byte = m.getByte(16);
    EXPECT_EQ(m, "160269685893737824716553112215759733036362."); EXPECT_EQ(byte, 214);
    m = "10508309163485329714717037142191754547449136985942971556952.";
    m.setByte(9, 180); byte = m.getByte(6);
    EXPECT_EQ(m, "10508309163485329714717037142191754443557074362810776855640."); EXPECT_EQ(byte, 228);
    m = "111102977031713641674174197978365139792654107808542650792459477335437066361683448084.";
    m.setByte(8, 114); byte = m.getByte(0);
    EXPECT_EQ(m, "111102977031713641674174197978365139792654107808542650792459475988824748980886180116."); EXPECT_EQ(byte, 20);
    m = "112578714913510269036928215992079535807521781.";
    m.setByte(11, 147); byte = m.getByte(7);
    EXPECT_EQ(m, "112578714913510241802247351713713488026788853."); EXPECT_EQ(byte, 113);
    m = "20773901254363119471620870402639012738947407151580610415157876255794975742902516977253730226841830394541489529519173135134365685823731619549926314701.";
    m.setByte(43, 126); byte = m.getByte(45);
    EXPECT_EQ(m, "20773901254363119471620870402639012738947407008236946915778406780118669786522083177468418403824260160942187067836493379604065181447572050167070905037."); EXPECT_EQ(byte, 217);
    m = "201597637203785798797953938331673561091488455350571584431051426374181195.";
    m.setByte(15, 48); byte = m.getByte(15);
    EXPECT_EQ(m, "201597637203785798797953938331673405571812948515414454685625373573865803."); EXPECT_EQ(byte, 48);
    m = "1273099179911989256007890929447820995344548162282325938173213561655912873711039769362742292989968822599603191407871687860.";
    m.setByte(10, 105); byte = m.getByte(13);
    EXPECT_EQ(m, "1273099179911989256007890929447820995344548162282325938173213561655912873711039769362742292990026851038944693608257584308."); EXPECT_EQ(byte, 74);
    m = "2638710524646020090465303056768019305932473342019.";
    m.setByte(20, 33); byte = m.getByte(11);
    EXPECT_EQ(m, "49406762919234913472983217703689075934922314717251."); EXPECT_EQ(byte, 147);
    m = "31999212342009031986744849378199594297045373447607.";
    m.setByte(9, 207); byte = m.getByte(19);
    EXPECT_EQ(m, "31999212342009031986744849609595551957657988918711."); EXPECT_EQ(byte, 229);
    m = "35087192645158151269133339872096994654592615394631.";
    m.setByte(3, 84); byte = m.getByte(6);
    EXPECT_EQ(m, "35087192645158151269133339872096994654593689136455."); EXPECT_EQ(byte, 50);
    m = "291228422026370938625869604419923779450368.";
    m.setByte(16, 159); byte = m.getByte(15);
    EXPECT_EQ(m, "315728752444678507995232576155011090675200."); EXPECT_EQ(byte, 215);
    m = "465673394661093310764445934267423333606317747385252379248542450646161678745543101141824945518810222677625240365173547657428761078.";
    m.setByte(50, 101); byte = m.getByte(49);
    EXPECT_EQ(m, "465673206156852210420118889385323777386450921316729502715226014008028651278266693912417737554699433227563545544241993646861744630."); EXPECT_EQ(byte, 79);
    m = "290101750950999312213039444518955864872086306858985238167343950143241.";
    m.setByte(9, 87); byte = m.getByte(14);
    EXPECT_EQ(m, "290101750950999312213039444518955864872086306377303856914640138346249."); EXPECT_EQ(byte, 126);
    m = "348078713664321910111850667944298147653512968372716726271684182190576725100332384926365708.";
    m.setByte(6, 29); byte = m.getByte(4);
    EXPECT_EQ(m, "348078713664321910111850667944298147653512968372716726271684182190576725083162411347015692."); EXPECT_EQ(byte, 71);
    m = "17183495705326843067303991572885921560074965987609578451775601507586593393742206536366778600225095913507698274118240627593197060679.";
    m.setByte(32, 168); byte = m.getByte(0);
    EXPECT_EQ(m, "17183495705326843067303991572885921560074965987609574167468299726887362721615761214914188029235663284806828814187632334807400383047."); EXPECT_EQ(byte, 71);
    m = "115493522494918195722613577735684103729932045295348141595603101481896210913508427413936970361917711555146640797730688159388368088.";
    m.setByte(28, 190); byte = m.getByte(13);
    EXPECT_EQ(m, "115493522494918195722613577735684103729932045295348141595603317161469548118626784750057666518963100652302021122310536988270361816."); EXPECT_EQ(byte, 163);
    m = "5562807300607821983303438698013381735289977046199667414819803888262555484924377168982681832413375826816099726993702476032342512855415529787367015.";
    m.setByte(23, 225); byte = m.getByte(37);
    EXPECT_EQ(m, "5562807300607821983303438698013381735289977046199667414819803888262555484924377168982685167123672750990255514756833555105126067231745401305701991."); EXPECT_EQ(byte, 57);
    m = "46476223558803333439547350188187739174148987245997067275699716148.";
    m.setByte(11, 188); byte = m.getByte(4);
    EXPECT_EQ(m, "46476223558803333439547350188187739176315382314746482756773183540."); EXPECT_EQ(byte, 67);
    m = "1056084716594942775358538951120102968181960559109474502505673967106451345018094862844458463943815682492683943984853910.";
    m.setByte(14, 198); byte = m.getByte(13);
    EXPECT_EQ(m, "1056084716594942775358538951120102968181960559109474502505673967106451345018094863607726102148435343886666904380208022."); EXPECT_EQ(byte, 114);
    m = "1934626198763332123161722839367426245490770909655229040991416162699069968202878392954941304256008847790219627238496671039766.";
    m.setByte(41, 77); byte = m.getByte(41);
    EXPECT_EQ(m, "1934626198763332123161680734790974172511226738987357026780133114187525273863477196870767912660774259803697574609838977253654."); EXPECT_EQ(byte, 77);
    m = "2102951809858015776056628841411429556690003098454774990678013392861347740391199946754217775965995600491859202779.";
    m.setByte(3, 12); byte = m.getByte(11);
    EXPECT_EQ(m, "2102951809858015776056628841411429556690003098454774990678013392861347740391199946754217775965995600489611055835."); EXPECT_EQ(byte, 180);
    m = "4276514576750030029829647085975374372063557647717830580690439840639538106876432930831508526773160576491990132107030820061613555330684241666.";
    m.setByte(8, 126); byte = m.getByte(41);
    EXPECT_EQ(m, "4276514576750030029829647085975374372063557647717830580690439840639538106876432930831508526773160576491990132107030821260651920121805096706."); EXPECT_EQ(byte, 114);
    m = "5270957713426030684987553923002745495719.";
    m.setByte(4, 37); byte = m.getByte(0);
    EXPECT_EQ(m, "5270957713426030684987553922341320532135."); EXPECT_EQ(byte, 167);
    m = "1810019463212630161229225735906487090124331777980760941175488394210634267303621721134578971665395285233946049155499596114690.";
    m.setByte(37, 119); byte = m.getByte(16);
    EXPECT_EQ(m, "1810019463212630161229225735906476650314953063739568815391335296147558878528104183211294461445592344029784762054880406203138."); EXPECT_EQ(byte, 193);
    m = "16057991375434093123432454323061689391672.";
    m.setByte(15, 110); byte = m.getByte(0);
    EXPECT_EQ(m, "16140403511172757907552490360799070755384."); EXPECT_EQ(byte, 56);
    m = "6177859178807629291011956649258930852428927962903598715840908056060555503364145349987198981272360442714039812144026849005495197371728.";
    m.setByte(38, 88); byte = m.getByte(16);
    EXPECT_EQ(m, "6177859178807629291011956649258930852431046480318986581370627239576501256651638877116611554973022028781368368695780719179925930642768."); EXPECT_EQ(byte, 65);
    m = "621478024739904716223180743056579165744997535422960345221476097183615689776.";
    m.setByte(16, 53); byte = m.getByte(8);
    EXPECT_EQ(m, "621478024739904716223180743056579164724150434660144954831352274888311055408."); EXPECT_EQ(byte, 160);
    m = "66134752615439768993960240082581357891527133480215851289338861987218308299500960547297074539918237613762885360433286518564.";
    m.setByte(23, 147); byte = m.getByte(2);
    EXPECT_EQ(m, "66134752615439768993960240082581357891527133480215851289338861985256714007192622808598390345165841858730899284038275733284."); EXPECT_EQ(byte, 2);
    m = "1485004686580204777608746117376941160121158721950763536730.";
    m.setByte(0, 202); byte = m.getByte(21);
    EXPECT_EQ(m, "1485004686580204777608746117376941160121158721950763536842."); EXPECT_EQ(byte, 44);
    m = "45867265252607230522208461404638921493920501576779512200404388102964162379501130340590.";
    m.setByte(7, 127); byte = m.getByte(9);
    EXPECT_EQ(m, "45867265252607230522208461404638921493920501576779512200404388102964738840253433764078."); EXPECT_EQ(byte, 65);
    m = "7137440861238914938639440457363081009158196948141604506814724380176028675837751555610603509998270743224748809586635343003566582884.";
    m.setByte(23, 218); byte = m.getByte(39);
    EXPECT_EQ(m, "7137440861238914938639440457363081009158196948141604506814724380176028677308947274841856814022283889289045625860624900299824671844."); EXPECT_EQ(byte, 63);
    m = "13731791177170774580929854100369547024037381358146959252514251517702945181148905298437602368706585210514768232633597266754734082189.";
    m.setByte(8, 14); byte = m.getByte(45);
    EXPECT_EQ(m, "13731791177170774580929854100369547024037381358146959252514251517702945181148905298437602368706585210514768229737458447182334478477."); EXPECT_EQ(byte, 16);
    m = "16262263275612795439518620016861099109251053071274191909832297141860678949053996736984767597365644682997974828257336653838302218735461480200949492842701.";
    m.setByte(21, 146); byte = m.getByte(56);
    EXPECT_EQ(m, "16262263275612795439518620016861099109251053071274191909832297141860678949053996736984767597365644681127252732473780918537585632858619215041355837833421."); EXPECT_EQ(byte, 235);
    m = "6639883088500219477216161912431723191534636015959980805476710177662073850201537393715276499076312062881651782093565395872452787984.";
    m.setByte(35, 13); byte = m.getByte(20);
    EXPECT_EQ(m, "6639883088500219477216161912431723191534635897457178379707236851574897533966884088528459316895801266521094788020403998354734399248."); EXPECT_EQ(byte, 110);
    m = "73994250519711384469513487172938121517400311694199629013672500425483897993597953338482441209847546665803788032487793304945145116892718326.";
    m.setByte(9, 39); byte = m.getByte(4);
    EXPECT_EQ(m, "73994250519711384469513487172938121517400311694199629013672500425483897993597953338482441209847546665803788032487103839438646148691518710."); EXPECT_EQ(byte, 100);
    m = "793564991844287200630047323074742917736625995479452670020511301846822912259025330856120150603599350.";
    m.setByte(15, 100); byte = m.getByte(37);
    EXPECT_EQ(m, "793564991844287200630047323074742917736625995479452670020511401538922596127715798641649671629442550."); EXPECT_EQ(byte, 121);
    m = "13090797092636019816163623996579298165684641721209098983921178899956399986740.";
    m.setByte(29, 66); byte = m.getByte(22);
    EXPECT_EQ(m, "13091031752011810695332396778278615584550025058914152777064054425780009162804."); EXPECT_EQ(byte, 220);
    m = "629411669098325908812721438426937759091490014194301388162120149385198278003027033553768188335377727761676574034453.";
    m.setByte(14, 11); byte = m.getByte(40);
    EXPECT_EQ(m, "629411669098325908812721438426937759091490014194301388162120149385198278003026213170864539832612419943256557259285."); EXPECT_EQ(byte, 213);
    m = "2789173390702549845834028264408739290703958576685791603970032398834960074.";
    m.setByte(30, 149); byte = m.getByte(12);
    EXPECT_EQ(m, "264282538977903430624162058374360679633171575314277397405967750190142686922."); EXPECT_EQ(byte, 231);
    m = "3419802312413369670736731265829631703528336250153109395.";
    m.setByte(0, 68); byte = m.getByte(4);
    EXPECT_EQ(m, "3419802312413369670736731265829631703528336250153109316."); EXPECT_EQ(byte, 168);
    m = "1011543270904368030564683659208195407756.";
    m.setByte(13, 40); byte = m.getByte(8);
    EXPECT_EQ(m, "1011543331751596841519694931049949265804."); EXPECT_EQ(byte, 56);
    m = "50434230858711963742992984066540008923658133806292734425760603926952040921278462490564352668910625208917741063231316733253460324874861403445573966387662.";
    m.setByte(60, 145); byte = m.getByte(5);
    EXPECT_EQ(m, "50434324511168473222759925514456902718647306264237014374895129701660829682430627083532956278093717113676875858039250375452147188314131782094133155092942."); EXPECT_EQ(byte, 74);
    m = "2249093479877734359993614809207903904159754982030340557700833539344438317419765647884417875378193723062.";
    m.setByte(27, 255); byte = m.getByte(36);
    EXPECT_EQ(m, "2249093479877734359993614809207903922378781440690733856440652328619423108588617152179414122373992836790."); EXPECT_EQ(byte, 106);
    m = "42378477282261856731312705087274447911007417.";
    m.setByte(7, 122); byte = m.getByte(4);
    EXPECT_EQ(m, "42378477282261856731312700403530835445691577."); EXPECT_EQ(byte, 102);
    m = "11158577342622727202407368960763197418512921124081071302412917213598225519423853203737.";
    m.setByte(14, 105); byte = m.getByte(11);
    EXPECT_EQ(m, "11158577342622727202407368960763197418512921124080744187710829519457628098155112337689."); EXPECT_EQ(byte, 248);
    m = "1360984031730203166524502513415255695786358198569015208303948751194603380151405228647085529167021595166038385389960427003295724047921121593.";
    m.setByte(43, 175); byte = m.getByte(46);
    EXPECT_EQ(m, "1360984031730203166524502513415253043928583460048829908292288558156578084123136502822036207579926054036462908079401096044343690465096042809."); EXPECT_EQ(byte, 178);
    m = "400448052625448285010913014174622259382568896358407596609171724286336871441394884063534428171575511523016086778003.";
    m.setByte(46, 72); byte = m.getByte(28);
    EXPECT_EQ(m, "351147446727859977891057037486820496503898568291724145883284272211932739586189023618344458797482446908969707761811."); EXPECT_EQ(byte, 229);
    m = "788481636686698326104097825448480015212706753936591699466897625701192782629355447620402763997708336971084513236893443.";
    m.setByte(38, 98); byte = m.getByte(48);
    EXPECT_EQ(m, "788481636686698326104095120264703443015184189440717491812699749351165994266014602825886021379342251259938701685178115."); EXPECT_EQ(byte, 20);
    m = "100320208852603372314593488711214821536648761163583991456510547097467355638686801807582.";
    m.setByte(8, 8); byte = m.getByte(24);
    EXPECT_EQ(m, "100320208852603372314593488711214821536648761163583991456510547097430462150539382704350."); EXPECT_EQ(byte, 145);
    m = "17964089316120025336185856027508610243247568407315737763150915110312001685766680379813003316386440241775065171758848346.";
    m.setByte(36, 54); byte = m.getByte(43);
    EXPECT_EQ(m, "17964089316120025336185856027450920747824033156825713422365883892860348772274188044589909495066833740121611300197013850."); EXPECT_EQ(byte, 45);
    m = "182029462883275069130930933729621005217865543902454317675690643879022769402615098698687917179095002323370601540433429480076355648727620114921831584229.";
    m.setByte(4, 127); byte = m.getByte(26);
    EXPECT_EQ(m, "182029462883275069130930933729621005217865543902454317675690643879022769402615098698687917179095002323370601540433429480076355648727620114449385181669."); EXPECT_EQ(byte, 50);
    m = "1911016421641240776465206014697183340299086607661634602261942084274172160628366.";
    m.setByte(26, 212); byte = m.getByte(27);
    EXPECT_EQ(m, "1911016421641236251327673381380567414133834574947745899738311591929547951953550."); EXPECT_EQ(byte, 41);
    m = "3325321678389425978579879479669246890801583364181684521492317892506182612444455221286978862148095036873348750218315923807.";
    m.setByte(8, 29); byte = m.getByte(36);
    EXPECT_EQ(m, "3325321678389425978579879479669246890801583364181684521492317892506182612444455221286978862148095033645168537319144391007."); EXPECT_EQ(byte, 46);
    m = "2173152753648370924619686483406380788147394971772578675702331090856352504746793519260469825010373594649300034.";
    m.setByte(32, 111); byte = m.getByte(26);
    EXPECT_EQ(m, "2173152753648370924619686483396191084294511146575304429021566554965264746096217149624997557617677239240985666."); EXPECT_EQ(byte, 245);
    m = "93652610052884203582534995443554757943547541322139246512766228981548346339241273072793239086966174535601928402454452635426588922162113464372.";
    m.setByte(50, 169); byte = m.getByte(26);
    EXPECT_EQ(m, "93652610052884203394030754343210430898665441765919379686697706105015029902603140045325962679736767327637817613004390940605657368151546447924."); EXPECT_EQ(byte, 175);
    m = "36200731266180541177114883906785686649558621833690070854977534830924009076982586251508338088.";
    m.setByte(34, 244); byte = m.getByte(36);
    EXPECT_EQ(m, "36200731380008796580966196655972907752499182769768596580688834904272392460121509212751023528."); EXPECT_EQ(byte, 87);
    m = "17391309863097998239045521972761256492230203723500188887105697900117907047466214389118395451427371071410889669425846.";
    m.setByte(1, 251); byte = m.getByte(23);
    EXPECT_EQ(m, "17391309863097998239045521972761256492230203723500188887105697900117907047466214389118395451427371071410889669475254."); EXPECT_EQ(byte, 212);
    m = "1407130623260803283530828213974028495775069119579623017958160626061384.";
    m.setByte(18, 176); byte = m.getByte(19);
    EXPECT_EQ(m, "1407130623260803283530830555552274341490498980830041646036118754005064."); EXPECT_EQ(byte, 46);
    m = "456908067635076234443672710378015958650713719.";
    m.setByte(16, 112); byte = m.getByte(15);
    EXPECT_EQ(m, "456942095871768328290019047838759135471859319."); EXPECT_EQ(byte, 33);
    m = "360795999685155339629839375054277164342266052643001946340636652018023181989014381766576434352377790719405780428924023.";
    m.setByte(18, 169); byte = m.getByte(47);
    EXPECT_EQ(m, "360795999685155339629839375054277164342266052643001946340636652018023180338759237075310321878734638543427028986373239."); EXPECT_EQ(byte, 40);
    m = "311302337171820240565985277178449260725120130419545776073427385976405885902151672908620838775090128321518313429218456606110.";
    m.setByte(46, 49); byte = m.getByte(30);
    EXPECT_EQ(m, "311302337133942945791008894879047717660101702842030767924634003101638697113280205751572433798907590875568763786719409313182."); EXPECT_EQ(byte, 54);
    m = "39518713092252491376689299367517442106606935267310688866652622.";
    m.setByte(22, 83); byte = m.getByte(5);
    EXPECT_EQ(m, "39518703418374389660765880980451833220280213674019980082663886."); EXPECT_EQ(byte, 233);
    m = "36360988318521532933111285235496337387964127473039718584891264860140912458851521464217754380375249405922191705077674264595439100339253973314.";
    m.setByte(14, 147); byte = m.getByte(47);
    EXPECT_EQ(m, "36360988318521532933111285235496337387964127473039718584891264860140912458851521464217754380375249405922539588967196098046550643593311719746."); EXPECT_EQ(byte, 141);
    m = "13483091628297795559035205069669460060934696416344517948751.";
    m.setByte(7, 94); byte = m.getByte(16);
    EXPECT_EQ(m, "13483091628297795559035205069669460060931525882206849119567."); EXPECT_EQ(byte, 25);
    m = "421102176685849998087496851490258579044790730156201829241301523717673247975941823263867577055811595798600524486563225325720154118.";
    m.setByte(6, 102); byte = m.getByte(30);
    EXPECT_EQ(m, "421102176685849998087496851490258579044790730156201829241301523717673247975941823263867577055811595798600524486532544553258692614."); EXPECT_EQ(byte, 233);
    m = "1539712383991468731147959762434313463069860657898734107297303689523010892604102562681352187645897385474586138546469072682924655.";
    m.setByte(35, 27); byte = m.getByte(24);
    EXPECT_EQ(m, "1539712383991468731147959762434313463069423557397983318256346810593975600850108021142565456253615727682394885082297900770835055."); EXPECT_EQ(byte, 58);
    m = "12217128826724450276060706007664749680431372927830876608.";
    m.setByte(7, 90); byte = m.getByte(8);
    EXPECT_EQ(m, "12217128826724450276060706007664749673585901494227722688."); EXPECT_EQ(byte, 82);
    m = "17419232812597520462000268842221105792335455052508003111544287661735608594230419136530608911762953775398262838718895743839862901792.";
    m.setByte(40, 141); byte = m.getByte(17);
    EXPECT_EQ(m, "17419232812597520462000268842221392014598268454459044044452914381718965356636414928001781815338401952841517444552598820759512402976."); EXPECT_EQ(byte, 111);
    m = "184467690679076334321460806100772548333461128504629.";
    m.setByte(16, 56); byte = m.getByte(19);
    EXPECT_EQ(m, "184467690669548428047674529123798059325371618583861."); EXPECT_EQ(byte, 55);
    m = "1429673876746702587655009481558601402934302287124320111801480475362784362832763974997670394852661915831868699746793295118584346763256034810.";
    m.setByte(31, 28); byte = m.getByte(8);
    EXPECT_EQ(m, "1429673876746702587655009481558601402934302287124320111801480408420482772509338495745694686704965188160158814923342209807168592188477961722."); EXPECT_EQ(byte, 145);
    m = "15711311577106209191751215367128206320014449.";
    m.setByte(12, 58); byte = m.getByte(14);
    EXPECT_EQ(m, "15711311577093928826561504394801207007712369."); EXPECT_EQ(byte, 31);
    m = "473124467356504233367935373407112941865942055.";
    m.setByte(10, 224); byte = m.getByte(8);
    EXPECT_EQ(m, "473124467356504233450142329140907725745962023."); EXPECT_EQ(byte, 34);
    m = "478353996718892500934865980259176213453100805047672774814707703913271296499182597921077456832772909336174142773007532111530774.";
    m.setByte(43, 155); byte = m.getByte(2);
    EXPECT_EQ(m, "478353996718892500938664587341909769394206227155516856310402014676581262110372979436312047846294462299540111001596177779886870."); EXPECT_EQ(byte, 109);
    m = "265295974621243484664052991725735921041305149616509439218495821669521369030442567113160428726301251093246869277096.";
    m.setByte(14, 229); byte = m.getByte(24);
    EXPECT_EQ(m, "265295974621243484664052991725735921041305149616509439218495821669521369030443460188220096716653358338615495133608."); EXPECT_EQ(byte, 141);
    m = "9697532499093658222346399676087833087654803124597239646227722660469182853334637903772941177037087266060646120869344893346866247.";
    m.setByte(22, 58); byte = m.getByte(23);
    EXPECT_EQ(m, "9697532499093658222346399676087833087654803124597239646227722660469182837434996667289344271569236859376188340824926500691993671."); EXPECT_EQ(byte, 2);
    m = "76037161695865804934571367569966851600682818152405514254217961538225004989182609775661990076143914514406211790571896357318.";
    m.setByte(14, 196); byte = m.getByte(47);
    EXPECT_EQ(m, "76037161695865804934571367569966851600682818152405514254217961538225004989182609775662986997140753201310889645867106615750."); EXPECT_EQ(byte, 240);
    m = "8218490602206465278523501514619487822031474091.";
    m.setByte(17, 158); byte = m.getByte(10);
    EXPECT_EQ(m, "8220494184782895764196373864308046073260527019."); EXPECT_EQ(byte, 136);

    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(),
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());;
}

TEST(Bitwise, CountBitsBytes) {
    const auto timeStart = std::chrono::system_clock::now();

    {
        Aesi512 m = 0; EXPECT_EQ(m.bitCount(), 0); EXPECT_EQ(m.byteCount(), 0);
    }{
        Aesi512 m = "8125433280239853252548211056150939786662988783355321803610299695202888531280061795935472804268151222945147553964613898040510775501409983.";
        EXPECT_EQ(m.bitCount(), 452); EXPECT_EQ(m.byteCount(), 57);
    }{
        Aesi512 m = "17574267251873162264218761220759873754600756500939123609416723673535177168061667995476463924261008261536776611267749.";
        EXPECT_EQ(m.bitCount(), 383); EXPECT_EQ(m.byteCount(), 48);
    }{
        Aesi512 m = "21802890716849309559167483785510602695344058930676827225663397326140604460817662307.";
        EXPECT_EQ(m.bitCount(), 274); EXPECT_EQ(m.byteCount(), 35);
    }{
        Aesi512 m = "131809808420798093069108302425012024571137335830720744893629509248644271768860200114478317853560728340486283732831799757717499791405.";
        EXPECT_EQ(m.bitCount(), 436); EXPECT_EQ(m.byteCount(), 55);
    }{
        Aesi512 m = "433465209374823396906000597351582380255601365637951225491.";
        EXPECT_EQ(m.bitCount(), 189); EXPECT_EQ(m.byteCount(), 24);
    }{
        Aesi512 m = "2629456433952760142429083664445582597876949492372857231139584353910376944544651100012259827002301897509955630054342321196527349466023286261.";
        EXPECT_EQ(m.bitCount(), 460); EXPECT_EQ(m.byteCount(), 58);
    }{
        Aesi512 m = "96406040941996537769947305839001509182667236508149012765337105619119234275587188230156962357657907962935261056152266015569090413578126311393192507827.";
        EXPECT_EQ(m.bitCount(), 495); EXPECT_EQ(m.byteCount(), 62);
    }{
        Aesi512 m = "779975898910398721588512843488292737216066708787279402842479265134130380576933342820188191171817284387486614000372261528684686600311889254943.";
        EXPECT_EQ(m.bitCount(), 469); EXPECT_EQ(m.byteCount(), 59);
    }{
        Aesi512 m = "235942049152378908081179353103850208970447097919975182053246678543156938370379557034727388517479366065434560109466574006814835006.";
        EXPECT_EQ(m.bitCount(), 427); EXPECT_EQ(m.byteCount(), 54);
    }{
        Aesi512 m = "201875921226053996734972115605743747569650032938025946229041017159355210997256571122101768564433281896618227162.";
        EXPECT_EQ(m.bitCount(), 367); EXPECT_EQ(m.byteCount(), 46);
    }{
        Aesi512 m = "15073045815333316943023135965557590030059580430131020419417594043252226785264218082879695545355297413008838849654297643522962831581276346739748452411005.";
        EXPECT_EQ(m.bitCount(), 503); EXPECT_EQ(m.byteCount(), 63);
    }{
        Aesi512 m = "4779143266357253920771631149725776855741684141203010232217393689880280981988983708.";
        EXPECT_EQ(m.bitCount(), 272); EXPECT_EQ(m.byteCount(), 34);
    }{
        Aesi512 m = "2073421040132906101606968078759317225438721792138284094607812513052260945297068866042052395426362224798695857992574.";
        EXPECT_EQ(m.bitCount(), 380); EXPECT_EQ(m.byteCount(), 48);
    }{
        Aesi512 m = "8056659416481504751897050853317416496949290710572167878886003741841335717120626786797209972094490663311696280531.";
        EXPECT_EQ(m.bitCount(), 372); EXPECT_EQ(m.byteCount(), 47);
    }{
        Aesi512 m = "2142462200685170464758953014183775179871969460337463575081194035221992317627132584005119954695168082695849861091.";
        EXPECT_EQ(m.bitCount(), 370); EXPECT_EQ(m.byteCount(), 47);
    }{
        Aesi512 m = "62436602421201799605543908261767952405552978341834407578470057072479070877359989051911219180730700900405485784333527578189654108644284314.";
        EXPECT_EQ(m.bitCount(), 455); EXPECT_EQ(m.byteCount(), 57);
    }{
        Aesi512 m = "8708886923751208896369646390414934986338160207927996314637234694235206909979086845075358681085923263473283036019941211939578987417.";
        EXPECT_EQ(m.bitCount(), 432); EXPECT_EQ(m.byteCount(), 54);
    }{
        Aesi512 m = "89350417268629761945311302081940817634086062965758014462352660629259870134745.";
        EXPECT_EQ(m.bitCount(), 256); EXPECT_EQ(m.byteCount(), 32);
    }{
        Aesi512 m = "947042416456442092300475377118864687689487941476858357893004304488139014532690257721893907183506176965809182242.";
        EXPECT_EQ(m.bitCount(), 369); EXPECT_EQ(m.byteCount(), 47);
    }{
        Aesi512 m = "51023079086295460865928137217812586222771465721882979976763630417920201452678256173660956.";
        EXPECT_EQ(m.bitCount(), 295); EXPECT_EQ(m.byteCount(), 37);
    }{
        Aesi512 m = "394659189019993910186127325893947611904290244834873046551135935656.";
        EXPECT_EQ(m.bitCount(), 218); EXPECT_EQ(m.byteCount(), 28);
    }{
        Aesi512 m = "37877413867687525478310222305154391480148401261437989997408830529823347525043333150254230178590071374435760769836732108165534044.";
        EXPECT_EQ(m.bitCount(), 424); EXPECT_EQ(m.byteCount(), 53);
    }{
        Aesi512 m = "238070679664433351272792579356666604707321803317863286672042730788775395964738812011349334479630745454353018796.";
        EXPECT_EQ(m.bitCount(), 367); EXPECT_EQ(m.byteCount(), 46);
    }{
        Aesi512 m = "267012496918653157167419170959622614804856381218193711762331242352905645916107540494367875074170692.";
        EXPECT_EQ(m.bitCount(), 327); EXPECT_EQ(m.byteCount(), 41);
    }{
        Aesi512 m = "1073603691947597544383478242209761023668674955975778069033461724734109933796134001131961241790993489349003858738008559432.";
        EXPECT_EQ(m.bitCount(), 399); EXPECT_EQ(m.byteCount(), 50);
    }{
        Aesi512 m = "562298521164442233158113896589082715246806788884332216264624631405816988860814005265019204093949261669027769520262704189761495475125709632881919.";
        EXPECT_EQ(m.bitCount(), 478); EXPECT_EQ(m.byteCount(), 60);
    }{
        Aesi512 m = "4158978421816859451128522099515091277614005938463674827112430404792191330568135712633235561579918390866268466676374544980567880949604447805648565.";
        EXPECT_EQ(m.bitCount(), 481); EXPECT_EQ(m.byteCount(), 61);
    }{
        Aesi512 m = "14924619216098556351851155524145079961383025330646663269574812166523143205418868252624810697655395.";
        EXPECT_EQ(m.bitCount(), 323); EXPECT_EQ(m.byteCount(), 41);
    }{
        Aesi512 m = "19263125433922139070353143174488847529421346973177209705127231844581796037119306630930585539489675909638996614473661956001635767638852.";
        EXPECT_EQ(m.bitCount(), 443); EXPECT_EQ(m.byteCount(), 56);
    }{
        Aesi512 m = "682806395672504228438313395837976674746610475712229974529528356835885995631486207670649897686632166147160247453750254696315344897085670207253.";
        EXPECT_EQ(m.bitCount(), 468); EXPECT_EQ(m.byteCount(), 59);
    }{
        Aesi512 m = "44165561608214009062731411641807426531080875031708.";
        EXPECT_EQ(m.bitCount(), 165); EXPECT_EQ(m.byteCount(), 21);
    }{
        Aesi512 m = "1810068275259181735129583643533689916163146029484908317346080710885725781416625194936179579160908146170735805.";
        EXPECT_EQ(m.bitCount(), 360); EXPECT_EQ(m.byteCount(), 45);
    }{
        Aesi512 m = "22203856399318573146469259769128778849626717054324.";
        EXPECT_EQ(m.bitCount(), 164); EXPECT_EQ(m.byteCount(), 21);
    }{
        Aesi512 m = "7694432564154108129168181465478548430868392855046604.";
        EXPECT_EQ(m.bitCount(), 173); EXPECT_EQ(m.byteCount(), 22);
    }{
        Aesi512 m = "1285401729605331393667176392943810500024205211000972253877823290865149302163170008070385888597921032464419651868028052643582626094727703773739.";
        EXPECT_EQ(m.bitCount(), 469); EXPECT_EQ(m.byteCount(), 59);
    }{
        Aesi512 m = "7965985649459576135397069050747352074976336307021856132905783547305822022499543909128907103263748211242642667.";
        EXPECT_EQ(m.bitCount(), 362); EXPECT_EQ(m.byteCount(), 46);
    }{
        Aesi512 m = "590007771335133157576169972021585528245811490226372057497198894444640402943288325.";
        EXPECT_EQ(m.bitCount(), 269); EXPECT_EQ(m.byteCount(), 34);
    }{
        Aesi512 m = "503140962366666306559517743623355538379506158349730754767227786462059097766284010030410635094454058647949382107093092344428041640646540768699921800282.";
        EXPECT_EQ(m.bitCount(), 498); EXPECT_EQ(m.byteCount(), 63);
    }{
        Aesi512 m = "3384777352685069629216151157031766989015071181826830603580543832463469699978010082081.";
        EXPECT_EQ(m.bitCount(), 281); EXPECT_EQ(m.byteCount(), 36);
    }{
        Aesi512 m = "7664372817785990326514266663237307917142062491158403533767880373365052616667133649.";
        EXPECT_EQ(m.bitCount(), 273); EXPECT_EQ(m.byteCount(), 35);
    }{
        Aesi512 m = "334215638884608286471654924286086692624854.";
        EXPECT_EQ(m.bitCount(), 138); EXPECT_EQ(m.byteCount(), 18);
    }{
        Aesi512 m = "34080246917087621931619781132472798534654778567128547384895677598758.";
        EXPECT_EQ(m.bitCount(), 225); EXPECT_EQ(m.byteCount(), 29);
    }{
        Aesi512 m = "29652837005697212319365507476545093089128339637930923529833544380900521909328030875059991197894410099757577321170398507.";
        EXPECT_EQ(m.bitCount(), 394); EXPECT_EQ(m.byteCount(), 50);
    }{
        Aesi512 m = "5507539735498776368361465467701902930055554588691378901463982655670782159582225073135111662243238408663182059630766021416042316106489465754074808853540974.";
        EXPECT_EQ(m.bitCount(), 511); EXPECT_EQ(m.byteCount(), 64);
    }{
        Aesi512 m = "3004975643443764126226151707508807531199878892716451932163126478736284424236635636281558878832196029.";
        EXPECT_EQ(m.bitCount(), 331); EXPECT_EQ(m.byteCount(), 42);
    }{
        Aesi512 m = "33813851846197160289962577402206813088513580354756664599207825023964752.";
        EXPECT_EQ(m.bitCount(), 235); EXPECT_EQ(m.byteCount(), 30);
    }{
        Aesi512 m = "377651090648098570965685387456182588498516.";
        EXPECT_EQ(m.bitCount(), 139); EXPECT_EQ(m.byteCount(), 18);
    }{
        Aesi512 m = "7410608557526364458156047672247498762540719035935927862959737209627950642463171897639112563283795045371205928539576466838870380014318225241722933868505375.";
        EXPECT_EQ(m.bitCount(), 512); EXPECT_EQ(m.byteCount(), 64);
    }{
        Aesi512 m = "10044990313301674850074738713697972584889987858.";
        EXPECT_EQ(m.bitCount(), 153); EXPECT_EQ(m.byteCount(), 20);
    }{
        Aesi512 m = "266099640168931550489566500155710899364300315367325913153890064383032281808.";
        EXPECT_EQ(m.bitCount(), 248); EXPECT_EQ(m.byteCount(), 31);
    }{
        Aesi512 m = "1482187387782878685086114755119507673273045170923240176317716983194935913741533035499712098299877.";
        EXPECT_EQ(m.bitCount(), 320); EXPECT_EQ(m.byteCount(), 40);
    }{
        Aesi512 m = "1173795849701350193393898818464811973771747962148586952799081909088.";
        EXPECT_EQ(m.bitCount(), 220); EXPECT_EQ(m.byteCount(), 28);
    }{
        Aesi512 m = "1075760111541429825505644691336694226487310.";
        EXPECT_EQ(m.bitCount(), 140); EXPECT_EQ(m.byteCount(), 18);
    }{
        Aesi512 m = "45171859382745997119719399550040468527765767975014522226860885063314372527502004679908084036170446028018456989547511984231472473256392440632.";
        EXPECT_EQ(m.bitCount(), 464); EXPECT_EQ(m.byteCount(), 58);
    }{
        Aesi512 m = "3550083230609724792508047331304227753793138230616730680629604946225570522563666068400196764430870432027087067680305391.";
        EXPECT_EQ(m.bitCount(), 391); EXPECT_EQ(m.byteCount(), 49);
    }{
        Aesi512 m = "238597071144511397377349027982816966185598914027694352122412323366822958481130189865972825210896.";
        EXPECT_EQ(m.bitCount(), 317); EXPECT_EQ(m.byteCount(), 40);
    }{
        Aesi512 m = "585583350737808456521626773441702619917565834.";
        EXPECT_EQ(m.bitCount(), 149); EXPECT_EQ(m.byteCount(), 19);
    }{
        Aesi512 m = "36049165193710656383674476318098849309081833966574925144648178032947125994642089205867044968998920577865084298839120084785200.";
        EXPECT_EQ(m.bitCount(), 414); EXPECT_EQ(m.byteCount(), 52);
    }{
        Aesi512 m = "19234485954353740431198448391386775220848294547901362245689574.";
        EXPECT_EQ(m.bitCount(), 204); EXPECT_EQ(m.byteCount(), 26);
    }{
        Aesi512 m = "14427157670805621590883010144772437273027.";
        EXPECT_EQ(m.bitCount(), 134); EXPECT_EQ(m.byteCount(), 17);
    }{
        Aesi512 m = "294592116837137192128962078358363421245896052688274598857112028471180048502021366005547689198067574009837.";
        EXPECT_EQ(m.bitCount(), 348); EXPECT_EQ(m.byteCount(), 44);
    }{
        Aesi512 m = "78014649575862615159507540438874452796220617919545457775300748208619982480459908586326061058242840934850085928995554769357091592190.";
        EXPECT_EQ(m.bitCount(), 435); EXPECT_EQ(m.byteCount(), 55);
    }{
        Aesi512 m = "4790741095386749449906937030566866031746046169304656278584923858487306270948959401572155081456882084125079092127248920113.";
        EXPECT_EQ(m.bitCount(), 401); EXPECT_EQ(m.byteCount(), 51);
    }{
        Aesi512 m = "633161550773844933515432109510252603991101078804467201160861966691510376183451.";
        EXPECT_EQ(m.bitCount(), 259); EXPECT_EQ(m.byteCount(), 33);
    }{
        Aesi512 m = "60122452179660513236341354307732611719576507171909615307390365442026320913187824739846223109852159341204602037787795321449601183014848534293967195178.";
        EXPECT_EQ(m.bitCount(), 495); EXPECT_EQ(m.byteCount(), 62);
    }{
        Aesi512 m = "161995631938575534869925629985890633415808895867596764222761722202191754575.";
        EXPECT_EQ(m.bitCount(), 247); EXPECT_EQ(m.byteCount(), 31);
    }{
        Aesi512 m = "911446817435997746933062831691885516611915352665999458511930156309284363237318454237971176312159689716360014561389228715376718662509862481060645832339.";
        EXPECT_EQ(m.bitCount(), 499); EXPECT_EQ(m.byteCount(), 63);
    }{
        Aesi512 m = "6823785331803106103597794618266911145262172838581223.";
        EXPECT_EQ(m.bitCount(), 173); EXPECT_EQ(m.byteCount(), 22);
    }{
        Aesi512 m = "155239918270315971266524910862097900140673811776184706157309200861937455.";
        EXPECT_EQ(m.bitCount(), 237); EXPECT_EQ(m.byteCount(), 30);
    }{
        Aesi512 m = "31791717464914174502002045742730878568603229720932639534325451917358893334089540214885596440815774662097524.";
        EXPECT_EQ(m.bitCount(), 354); EXPECT_EQ(m.byteCount(), 45);
    }{
        Aesi512 m = "1003586762829871808712634204549567450358275489539260017484895107722613982163260348845481532808575114339347311595215.";
        EXPECT_EQ(m.bitCount(), 379); EXPECT_EQ(m.byteCount(), 48);
    }{
        Aesi512 m = "4880164245754362338097205112813929292550604675359560497016587.";
        EXPECT_EQ(m.bitCount(), 202); EXPECT_EQ(m.byteCount(), 26);
    }{
        Aesi512 m = "19749483887303390161030962495768683795272790.";
        EXPECT_EQ(m.bitCount(), 144); EXPECT_EQ(m.byteCount(), 18);
    }{
        Aesi512 m = "63148738539734553392341133462985836036740639371798360597452567097525683306066658917516309566802305685801145960072360549701.";
        EXPECT_EQ(m.bitCount(), 405); EXPECT_EQ(m.byteCount(), 51);
    }{
        Aesi512 m = "107590342114875162918348472307699251018874103592474832066873484053754113708388326214150395.";
        EXPECT_EQ(m.bitCount(), 296); EXPECT_EQ(m.byteCount(), 37);
    }{
        Aesi512 m = "6698953979797583020827904972213947616977062455698932958829035301051011495115.";
        EXPECT_EQ(m.bitCount(), 252); EXPECT_EQ(m.byteCount(), 32);
    }{
        Aesi512 m = "5430207379526057381214725804197658314755709506858996920618690886502342024339746344612062226768199148738037128023372725688185796337503888328.";
        EXPECT_EQ(m.bitCount(), 461); EXPECT_EQ(m.byteCount(), 58);
    }{
        Aesi512 m = "48581874368844478342704760472464069633282490659897264814743946699473.";
        EXPECT_EQ(m.bitCount(), 225); EXPECT_EQ(m.byteCount(), 29);
    }{
        Aesi512 m = "125649816413620037750566080548892812403575569822422682828388593453790534195113364133371382224759702685614495034150.";
        EXPECT_EQ(m.bitCount(), 376); EXPECT_EQ(m.byteCount(), 47);
    }{
        Aesi512 m = "9090966395883606169412743491416218582727668678050520137422123160210277873679028562444511041122860878620562547703977100496584276850469220021458.";
        EXPECT_EQ(m.bitCount(), 472); EXPECT_EQ(m.byteCount(), 59);
    }{
        Aesi512 m = "174662122049287992457319598417451937883451980333462349197.";
        EXPECT_EQ(m.bitCount(), 187); EXPECT_EQ(m.byteCount(), 24);
    }{
        Aesi512 m = "813507411993390554879828672051435218814298367924349835202824439987190213087714864429892828655932570189200528471003010841188643835728860126206357.";
        EXPECT_EQ(m.bitCount(), 479); EXPECT_EQ(m.byteCount(), 60);
    }{
        Aesi512 m = "318619099993310255354692135487104214301662979.";
        EXPECT_EQ(m.bitCount(), 148); EXPECT_EQ(m.byteCount(), 19);
    }{
        Aesi512 m = "1777693857708080545844538076702187890341442968457518484.";
        EXPECT_EQ(m.bitCount(), 181); EXPECT_EQ(m.byteCount(), 23);
    }{
        Aesi512 m = "3204193290698947238931672972341151508380406116470290072152808452149735241615.";
        EXPECT_EQ(m.bitCount(), 251); EXPECT_EQ(m.byteCount(), 32);
    }{
        Aesi512 m = "1307947124275081320267802092014598881787726639944553485813418415062992248276001190919512309616545929415598763.";
        EXPECT_EQ(m.bitCount(), 360); EXPECT_EQ(m.byteCount(), 45);
    }{
        Aesi512 m = "28534850101898570474710141413913927972313704060038621588874963.";
        EXPECT_EQ(m.bitCount(), 205); EXPECT_EQ(m.byteCount(), 26);
    }{
        Aesi512 m = "186547211011565960360834677399752966883178428551654724629512194266072930.";
        EXPECT_EQ(m.bitCount(), 237); EXPECT_EQ(m.byteCount(), 30);
    }{
        Aesi512 m = "1369416445055542330663441898194145697102539611876570531.";
        EXPECT_EQ(m.bitCount(), 180); EXPECT_EQ(m.byteCount(), 23);
    }{
        Aesi512 m = "6148572308034937684753221787621890260678881623206328037140328020336862939698811176604453648297521709448290.";
        EXPECT_EQ(m.bitCount(), 352); EXPECT_EQ(m.byteCount(), 44);
    }{
        Aesi512 m = "51821718761454376853562798270553712164259354897877246623891326751545476237393228375021.";
        EXPECT_EQ(m.bitCount(), 285); EXPECT_EQ(m.byteCount(), 36);
    }{
        Aesi512 m = "444871851149431479626983465266102420775422.";
        EXPECT_EQ(m.bitCount(), 139); EXPECT_EQ(m.byteCount(), 18);
    }{
        Aesi512 m = "42146200574268847311698523498680873676706783693183222358186972843444316710.";
        EXPECT_EQ(m.bitCount(), 245); EXPECT_EQ(m.byteCount(), 31);
    }{
        Aesi512 m = "35453974191750223599443971448232408256392725686144289604176002346292.";
        EXPECT_EQ(m.bitCount(), 225); EXPECT_EQ(m.byteCount(), 29);
    }{
        Aesi512 m = "114399773536456471179538929162643059034663209191754021300097056704143055752447627827621176.";
        EXPECT_EQ(m.bitCount(), 296); EXPECT_EQ(m.byteCount(), 37);
    }{
        Aesi512 m = "306096489943631752388051638909060328435655026159556576651782659822077606.";
        EXPECT_EQ(m.bitCount(), 238); EXPECT_EQ(m.byteCount(), 30);
    }{
        Aesi512 m = "847445665267889245456213278113730865596260025864447116412907433580615172641770582449326549958998709878135166809715.";
        EXPECT_EQ(m.bitCount(), 379); EXPECT_EQ(m.byteCount(), 48);
    }{
        Aesi512 m = "855205585087312211296445789458057963768041132396148430034035722540576391486192631.";
        EXPECT_EQ(m.bitCount(), 269); EXPECT_EQ(m.byteCount(), 34);
    }{
        Aesi512 m = "7790374562144467155944484053910374470391879335804156417076224.";
        EXPECT_EQ(m.bitCount(), 203); EXPECT_EQ(m.byteCount(), 26);
    }{
        Aesi512 m = "4798011951051675050122476777454810968720955910594973468568707443382840023724197.";
        EXPECT_EQ(m.bitCount(), 262); EXPECT_EQ(m.byteCount(), 33);
    }


    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(),
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());;
}