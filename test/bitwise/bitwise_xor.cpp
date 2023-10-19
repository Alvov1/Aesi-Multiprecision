#include <gtest/gtest.h>
#include "../../AesiMultiprecision.h"

TEST(Bitwise, XOR) {
    Aesi<96> m0 = 8980983974862324411ULL, m1 = 2275821107446334292ULL;
    EXPECT_EQ(m0 ^ m1, 7149335124743143919);
    Aesi<96> m2 = 5929039959812382730ULL, m3 = 5865990284982001136ULL;
    EXPECT_EQ(m2 ^ m3, 225181612106071546);
    Aesi<96> m4 = 2348733683970235220ULL, m5 = 8037993418087636272ULL;
    EXPECT_EQ(m4 ^ m5, 5698408927172019812);
    Aesi<96> m6 = 4731703132876683316ULL, m7 = 2866670897528785032ULL;
    EXPECT_EQ(m6 ^ m7, 7377484497988677820);
    Aesi<96> m8 = 8846197703556754491ULL, m9 = 89813386367205630ULL;
    EXPECT_EQ(m8 ^ m9, 8933757640696032453);
    Aesi<96> m10 = 6937171463592985735ULL, m11 = 4563941165456893861ULL;
    EXPECT_EQ(m10 ^ m11, 6851005267702244130);
    Aesi<96> m12 = 1518628271953756023ULL, m13 = 3946611315143942938ULL;
    EXPECT_EQ(m12 ^ m13, 2582372703773793389);
    Aesi<96> m14 = 5301450474978658558ULL, m15 = 6723090551834134897ULL;
    EXPECT_EQ(m14 ^ m15, 1504129857657110927);
    Aesi<96> m16 = 4223988422606718532ULL, m17 = 9219199883406280541ULL;
    EXPECT_EQ(m16 ^ m17, 5003413861837504793);
    Aesi<96> m18 = 9014466296910835551ULL, m19 = 8652528329170907179ULL;
    EXPECT_EQ(m18 ^ m19, 363144475731926900);
    Aesi<96> m20 = 6713998067778146158ULL, m21 = 6942300296191815209ULL;
    EXPECT_EQ(m20 ^ m21, 4428420511131328839);
    Aesi<96> m22 = 8176197850194658266ULL, m23 = 8744843587304529273ULL;
    EXPECT_EQ(m22 ^ m23, 588947120141754019);
    Aesi<96> m24 = 7048871244264337309ULL, m25 = 8609093729948658707ULL;
    EXPECT_EQ(m24 ^ m25, 1633463262290281358);
    Aesi<96> m26 = 3411559924976466180ULL, m27 = 8268042228590725135ULL;
    EXPECT_EQ(m26 ^ m27, 6766015453006033163);
    Aesi<96> m28 = 8832732530923275635ULL, m29 = 2160031969745157988ULL;
    EXPECT_EQ(m28 ^ m29, 7452844756437940759);
    Aesi<96> m30 = 1859952019346371937ULL, m31 = 491403244050845705ULL;
    EXPECT_EQ(m30 ^ m31, 2242282596984235368);
    Aesi<96> m32 = 69328175451101827ULL, m33 = 9136904141346616231ULL;
    EXPECT_EQ(m32 ^ m33, 9095723631210850596);
    Aesi<96> m34 = 5519209732596004881ULL, m35 = 1318401553819460099ULL;
    EXPECT_EQ(m34 ^ m35, 6833028504770583058);
    Aesi<96> m36 = 3109216737461888011ULL, m37 = 8347327756193470689ULL;
    EXPECT_EQ(m36 ^ m37, 6409047198741757162);
    Aesi<96> m38 = 6445545695326738505ULL, m39 = 3828317484873197560ULL;
    EXPECT_EQ(m38 ^ m39, 7805802279293370289);

    Aesi<96> r0 = 9122165470357914114ULL, r1 = 913347652617637195ULL;
    r0 ^= r1; EXPECT_EQ(r0, 8229393666183171913);
    Aesi<96> r2 = 8891362178826352742ULL, r3 = 8120718805155836972ULL;
    r2 ^= r3; EXPECT_EQ(r2, 853115956852706378);
    Aesi<96> r4 = 4170345981088879111ULL, r5 = 8511921372414751734ULL;
    r4 ^= r5; EXPECT_EQ(r4, 5746699712049832433);
    Aesi<96> r6 = 2199662578192301232ULL, r7 = 6665562358360395428ULL;
    r6 ^= r7; EXPECT_EQ(r6, 4757508558287172116);
    Aesi<96> r8 = 1817722836024882476ULL, r9 = 3712305808697824701ULL;
    r8 ^= r9; EXPECT_EQ(r8, 3079647677662453905);
    Aesi<96> r10 = 3847405400368673618ULL, r11 = 2970148324676566177ULL;
    r10 ^= r11; EXPECT_EQ(r10, 2043698520350101491);
    Aesi<96> r12 = 1440385263592145174ULL, r13 = 2670364454407021991ULL;
    r12 ^= r13; EXPECT_EQ(r12, 3959311232726246577);
    Aesi<96> r14 = 1308114057494421891ULL, r15 = 8648522490507570692ULL;
    r14 ^= r15; EXPECT_EQ(r14, 7647925461813055367);
    Aesi<96> r16 = 6425607198590700850ULL, r17 = 8362660867458128108ULL;
    r16 ^= r17; EXPECT_EQ(r16, 3252289920464049630);
    Aesi<96> r18 = 1905324811879118393ULL, r19 = 1207777558113759312ULL;
    r18 ^= r19; EXPECT_EQ(r18, 771225598206264937);
    Aesi<96> r20 = 7522557097996989209ULL, r21 = 3658099529145873901ULL;
    r20 ^= r21; EXPECT_EQ(r20, 6530593228804540148);
    Aesi<96> r22 = 6655607508161154119ULL, r23 = 7527531017527016207ULL;
    r22 ^= r23; EXPECT_EQ(r22, 3758920570628011848);
    Aesi<96> r24 = 2623629862769400602ULL, r25 = 1315083192849878500ULL;
    r24 ^= r25; EXPECT_EQ(r24, 3902681645905172222);
    Aesi<96> r26 = 4420157800326454539ULL, r27 = 7538214389803591162ULL;
    r26 ^= r27; EXPECT_EQ(r26, 6181911861934061809);
    Aesi<96> r28 = 3546712561762565144ULL, r29 = 5025233326456180804ULL;
    r28 ^= r29; EXPECT_EQ(r28, 8396191148970208348);
    Aesi<96> r30 = 8876018422015848960ULL, r31 = 284960104116629356ULL;
    r30 ^= r31; EXPECT_EQ(r30, 8708151944349506924);
    Aesi<96> r32 = 1581905550141243622ULL, r33 = 3640422532779266785ULL;
    r32 ^= r33; EXPECT_EQ(r32, 2842175667502682631);
    Aesi<96> r34 = 3383225491851133179ULL, r35 = 5000179251454908659ULL;
    r34 ^= r35; EXPECT_EQ(r34, 7752829254201169928);
    Aesi<96> r36 = 4476723811974996976ULL, r37 = 1396664004355549946ULL;
    r36 ^= r37; EXPECT_EQ(r36, 3261017439913281802);
    Aesi<96> r38 = 934443020163735547ULL, r39 = 1299518146827248963ULL;
    r38 ^= r39; EXPECT_EQ(r38, 2233537094250937016);

    Aesi y0 = -6652583106797519413; EXPECT_EQ(y0 ^ 0, -6652583106797519413); Aesi y1 = 7735878406530979626; EXPECT_EQ(y1 ^ 0, 7735878406530979626);
    Aesi y2 = 2162627523902914222; EXPECT_EQ(y2 ^ 0, 2162627523902914222); Aesi y3 = 1171815049526031649; EXPECT_EQ(y3 ^ 0, 1171815049526031649);
    Aesi y4 = 8557217326701597875; EXPECT_EQ(y4 ^ 0, 8557217326701597875); Aesi y5 = 5888932561581891950; EXPECT_EQ(y5 ^ 0, 5888932561581891950);
    Aesi y6 = 5043879029109165922; EXPECT_EQ(y6 ^ 0, 5043879029109165922); Aesi y7 = -1454113503148561589; EXPECT_EQ(y7 ^ 0, -1454113503148561589);
    Aesi y8 = -1459088315943265714; EXPECT_EQ(y8 ^ 0, -1459088315943265714); Aesi y9 = 1762258331777406157; EXPECT_EQ(y9 ^ 0, 1762258331777406157);
    Aesi y10 = -770974676523450011; EXPECT_EQ(y10 ^ 0, -770974676523450011); Aesi y11 = -4999307849436956084; EXPECT_EQ(y11 ^ 0, -4999307849436956084);
    Aesi y12 = 4354598598581085920; EXPECT_EQ(y12 ^ 0, 4354598598581085920); Aesi y13 = 148678106723966883; EXPECT_EQ(y13 ^ 0, 148678106723966883);
    Aesi y14 = -3422048039334349806; EXPECT_EQ(y14 ^ 0, -3422048039334349806); Aesi y15 = -3846906216831977821; EXPECT_EQ(y15 ^ 0, -3846906216831977821);
    Aesi y16 = 1187216367367166461; EXPECT_EQ(y16 ^ 0, 1187216367367166461); Aesi y17 = 5696250726926076608; EXPECT_EQ(y17 ^ 0, 5696250726926076608);
    Aesi y18 = 5163734250655534484; EXPECT_EQ(y18 ^ 0, 5163734250655534484); Aesi y19 = -3439774873613818458; EXPECT_EQ(y19 ^ 0, -3439774873613818458);
    Aesi y20 = -3004398981146302089; EXPECT_EQ(y20 ^ 0, -3004398981146302089); Aesi y21 = -2849073514542617295; EXPECT_EQ(y21 ^ 0, -2849073514542617295);
    Aesi y22 = 4351504996019293703; EXPECT_EQ(y22 ^ 0, 4351504996019293703); Aesi y23 = 8648251167683573561; EXPECT_EQ(y23 ^ 0, 8648251167683573561);
    Aesi y24 = -5738482727104124346; EXPECT_EQ(y24 ^ 0, -5738482727104124346); Aesi y25 = -3907626487262575028; EXPECT_EQ(y25 ^ 0, -3907626487262575028);
    Aesi y26 = 2705074888495144170; EXPECT_EQ(y26 ^ 0, 2705074888495144170); Aesi y27 = 4499344126070650797; EXPECT_EQ(y27 ^ 0, 4499344126070650797);
    Aesi y28 = 3165840682768100164; EXPECT_EQ(y28 ^ 0, 3165840682768100164); Aesi y29 = -591438519799156109; EXPECT_EQ(y29 ^ 0, -591438519799156109);
    Aesi y30 = 1539497900613426651; EXPECT_EQ(y30 ^ 0, 1539497900613426651); Aesi y31 = -3815268260161607160; EXPECT_EQ(y31 ^ 0, -3815268260161607160);
    Aesi y32 = 7440782379717764332; EXPECT_EQ(y32 ^ 0, 7440782379717764332); Aesi y33 = -7556059161534389080; EXPECT_EQ(y33 ^ 0, -7556059161534389080);
    Aesi y34 = -5614644988445039641; EXPECT_EQ(y34 ^ 0, -5614644988445039641); Aesi y35 = 3954451072551111104; EXPECT_EQ(y35 ^ 0, 3954451072551111104);
    Aesi y36 = -1242416396248218622; EXPECT_EQ(y36 ^ 0, -1242416396248218622); Aesi y37 = 5680201312284594976; EXPECT_EQ(y37 ^ 0, 5680201312284594976);
    Aesi y38 = -5454609604686326633; EXPECT_EQ(y38 ^ 0, -5454609604686326633); Aesi y39 = -4622311989677889067; EXPECT_EQ(y39 ^ 0, -4622311989677889067);
    Aesi y40 = -2474115385712135834; EXPECT_EQ(y40 ^ 0, -2474115385712135834); Aesi y41 = -8506743866264905679; EXPECT_EQ(y41 ^ 0, -8506743866264905679);
    Aesi y42 = -7866936034851706220; EXPECT_EQ(y42 ^ 0, -7866936034851706220); Aesi y43 = -8806792467525009614; EXPECT_EQ(y43 ^ 0, -8806792467525009614);
    Aesi y44 = 7503015719520133947; EXPECT_EQ(y44 ^ 0, 7503015719520133947); Aesi y45 = 6038827430243272660; EXPECT_EQ(y45 ^ 0, 6038827430243272660);
    Aesi y46 = -1030896399084174952; EXPECT_EQ(y46 ^ 0, -1030896399084174952); Aesi y47 = 5815047418117056630; EXPECT_EQ(y47 ^ 0, 5815047418117056630);
    Aesi y48 = -5829737776573732504; EXPECT_EQ(y48 ^ 0, -5829737776573732504); Aesi y49 = -3908082412208505143; EXPECT_EQ(y49 ^ 0, -3908082412208505143);
    Aesi y50 = 8241705532620992109; EXPECT_EQ(y50 ^ 0, 8241705532620992109); Aesi y51 = 1789725445496298122; EXPECT_EQ(y51 ^ 0, 1789725445496298122);
    Aesi y52 = 3547088116508155647; EXPECT_EQ(y52 ^ 0, 3547088116508155647); Aesi y53 = -845481703297730061; EXPECT_EQ(y53 ^ 0, -845481703297730061);
    Aesi y54 = 2934222797803673249; EXPECT_EQ(y54 ^ 0, 2934222797803673249); Aesi y55 = -2543088764666443823; EXPECT_EQ(y55 ^ 0, -2543088764666443823);
    Aesi y56 = 2217818106922620497; EXPECT_EQ(y56 ^ 0, 2217818106922620497); Aesi y57 = -4137323814594963163; EXPECT_EQ(y57 ^ 0, -4137323814594963163);
    Aesi y58 = 4814566605532490742; EXPECT_EQ(y58 ^ 0, 4814566605532490742); Aesi y59 = -6428660675943912762; EXPECT_EQ(y59 ^ 0, -6428660675943912762);

    Aesi e0 = -872399371349274581; EXPECT_EQ(e0 ^ e0, 0); Aesi e1 = 4655876137305428030; EXPECT_EQ(e1 ^ e1, 0);
    Aesi e2 = 4225124115658427348; EXPECT_EQ(e2 ^ e2, 0); Aesi e3 = 7088320902054361636; EXPECT_EQ(e3 ^ e3, 0);
    Aesi e4 = -3418738210646662755; EXPECT_EQ(e4 ^ e4, 0); Aesi e5 = 2970064323217158776; EXPECT_EQ(e5 ^ e5, 0);
    Aesi e6 = -6554273501984264009; EXPECT_EQ(e6 ^ e6, 0); Aesi e7 = 3560049859667286910; EXPECT_EQ(e7 ^ e7, 0);
    Aesi e8 = 2796596525801380326; EXPECT_EQ(e8 ^ e8, 0); Aesi e9 = -3680962316551344967; EXPECT_EQ(e9 ^ e9, 0);
    Aesi e10 = -2275148996157343394; EXPECT_EQ(e10 ^ e10, 0); Aesi e11 = 1510963602371676019; EXPECT_EQ(e11 ^ e11, 0);
    Aesi e12 = 2615335576808209499; EXPECT_EQ(e12 ^ e12, 0); Aesi e13 = -5483836872589866918; EXPECT_EQ(e13 ^ e13, 0);
    Aesi e14 = 7478822047765580133; EXPECT_EQ(e14 ^ e14, 0); Aesi e15 = -2123741133112998366; EXPECT_EQ(e15 ^ e15, 0);
    Aesi e16 = 2660392548653021630; EXPECT_EQ(e16 ^ e16, 0); Aesi e17 = -8445326165875817619; EXPECT_EQ(e17 ^ e17, 0);
    Aesi e18 = 4876777829330614480; EXPECT_EQ(e18 ^ e18, 0); Aesi e19 = 8798348075196592505; EXPECT_EQ(e19 ^ e19, 0);
    Aesi e20 = 3759335478943361691; EXPECT_EQ(e20 ^ e20, 0); Aesi e21 = 7476161062704668043; EXPECT_EQ(e21 ^ e21, 0);
    Aesi e22 = 2878822222492835562; EXPECT_EQ(e22 ^ e22, 0); Aesi e23 = -2096225550442118875; EXPECT_EQ(e23 ^ e23, 0);
    Aesi e24 = 2389137150982677634; EXPECT_EQ(e24 ^ e24, 0); Aesi e25 = 977195684750857383; EXPECT_EQ(e25 ^ e25, 0);
    Aesi e26 = -181765273376632167; EXPECT_EQ(e26 ^ e26, 0); Aesi e27 = -6774299673585191159; EXPECT_EQ(e27 ^ e27, 0);
    Aesi e28 = -4359446246755790347; EXPECT_EQ(e28 ^ e28, 0); Aesi e29 = -6889367208525652850; EXPECT_EQ(e29 ^ e29, 0);
    Aesi e30 = 2532841348560638083; EXPECT_EQ(e30 ^ e30, 0); Aesi e31 = -6551650670623996088; EXPECT_EQ(e31 ^ e31, 0);
    Aesi e32 = -8074970477546236102; EXPECT_EQ(e32 ^ e32, 0); Aesi e33 = 6621774932800906521; EXPECT_EQ(e33 ^ e33, 0);
    Aesi e34 = -203051090539507323; EXPECT_EQ(e34 ^ e34, 0); Aesi e35 = -8523904323010071395; EXPECT_EQ(e35 ^ e35, 0);
    Aesi e36 = -5588016298856232703; EXPECT_EQ(e36 ^ e36, 0); Aesi e37 = -3072159085675554861; EXPECT_EQ(e37 ^ e37, 0);
    Aesi e38 = 4538407123913009898; EXPECT_EQ(e38 ^ e38, 0); Aesi e39 = -8678267948539782427; EXPECT_EQ(e39 ^ e39, 0);
    Aesi e40 = 5580103516555131807; EXPECT_EQ(e40 ^ e40, 0); Aesi e41 = -4252781619698947210; EXPECT_EQ(e41 ^ e41, 0);
    Aesi e42 = -6872747642310508664; EXPECT_EQ(e42 ^ e42, 0); Aesi e43 = -7143038148456854072; EXPECT_EQ(e43 ^ e43, 0);
    Aesi e44 = 7718402901465646931; EXPECT_EQ(e44 ^ e44, 0); Aesi e45 = -2283146554465274577; EXPECT_EQ(e45 ^ e45, 0);
    Aesi e46 = 8271372903316036199; EXPECT_EQ(e46 ^ e46, 0); Aesi e47 = 1349821174167213479; EXPECT_EQ(e47 ^ e47, 0);
    Aesi e48 = 5657399588981528159; EXPECT_EQ(e48 ^ e48, 0); Aesi e49 = -2336213843584333446; EXPECT_EQ(e49 ^ e49, 0);
    Aesi e50 = 7796780472781494262; EXPECT_EQ(e50 ^ e50, 0); Aesi e51 = 712989842011531055; EXPECT_EQ(e51 ^ e51, 0);
    Aesi e52 = 3333140542037539605; EXPECT_EQ(e52 ^ e52, 0); Aesi e53 = 8775266287560378042; EXPECT_EQ(e53 ^ e53, 0);
    Aesi e54 = -2531547591749992014; EXPECT_EQ(e54 ^ e54, 0); Aesi e55 = -4358331220058001262; EXPECT_EQ(e55 ^ e55, 0);
    Aesi e56 = 627142671780653697; EXPECT_EQ(e56 ^ e56, 0); Aesi e57 = -478975383411509354; EXPECT_EQ(e57 ^ e57, 0);
    Aesi e58 = 779413273234111004; EXPECT_EQ(e58 ^ e58, 0); Aesi e59 = 2987880071363343624; EXPECT_EQ(e59 ^ e59, 0);
}

TEST(Bitwise, DifferentPrecisionXOR) {
    {
        Aesi<448> first = "-3179857521619637484238747986927154980033758458337296525182790672089056246244.";
        Aesi<288> second = "-5533784449681278852061514331656626833263004121888632131908.";
        EXPECT_EQ(first ^ second, "-3179857521619637485014084370992276992422676982152110142279094461483155695776.");

        Aesi<416> third = "-274020239836115431940011776656040618501338662129745246073114940473323523792.";
        Aesi<256> forth = "-3907991008567283212028595563959962968013689106014038909895.";
        third ^= forth; EXPECT_EQ(third, "-274020239836115429570709335640956298643114926790370398629160468050703838487.");
    }
    {
        Aesi<448> first = "-44098988525877645387886437347166567702813365595532179779097431633311259488186.";
        Aesi<320> second = "3534354131407176500805994947148845186193645574505333060497.";
        EXPECT_EQ(first ^ second, "-44098988525877645385137535433391499999423275368760800820432166616580599061547.");

        Aesi<320> third = "3145126618379139574144790607544817471413323639789049508183305679970207687273.";
        Aesi<320> forth = "66991002351755223210620150985055498645907353134317074351.";
        third ^= forth; EXPECT_EQ(third, "3145126618379139574088582450466030482796584562618368646004971923501753046470.");
    }
    {
        Aesi<448> first = "-14368021904913845827095760764406964239842753113421542261479209150649578765569.";
        Aesi<288> second = "-498745418071599413408605505389886010320020785512859424562.";
        EXPECT_EQ(first ^ second, "-14368021904913845827384721904233331845249241917922205669209149902159341349427.");

        Aesi<384> third = "-29772370083208158656820951633212619203489647539014778838295836849166551893138.";
        Aesi<256> forth = "-4699632821134331348763640839601018929161677837571968153277.";
        third ^= forth; EXPECT_EQ(third, "-29772370083208158654677955385288450043028670017325907684288990044761835663919.");
    }
    {
        Aesi<384> first = "-63956352616214622268384774607904086103658742596606859479634863758158694829562.";
        Aesi<224> second = "2908790816350979790454282162283424384790761299901562697349.";
        EXPECT_EQ(first ^ second, "-63956352616214622271091047593635921646440571027110230321023776834009161988991.");

        Aesi<320> third = "-114147842447941408553196604185444481559011804842688383086934120663150726384467.";
        Aesi<288> forth = "-4383082463925741735706770518132802704572068862785644561118.";
        third ^= forth; EXPECT_EQ(third, "-114147842447941408550419867387355839485423818255075517901453923192390582536589.");
    }
    {
        Aesi<448> first = "91269328977884516372662562520352552570071259823161411095149666656388038379762.";
        Aesi<288> second = "1146008996583431140208982075471222282665452696645955741745.";
        EXPECT_EQ(first ^ second, "91269328977884516372031520217149298430197152565935933475347794854205641859267.");

        Aesi<352> third = "95247181009699450490115958481452273190203959032032615618603904868838198575742.";
        Aesi<320> forth = "5869892551767738400244831245111321115540754862820737396827.";
        third ^= forth; EXPECT_EQ(third, "95247181009699450493956053954741811522994366353039940324060556042988795605541.");
    }
    {
        Aesi<448> first = "-7438548282325391860559150295547835725390565035859209488235958114935059206199.";
        Aesi<320> second = "-1442260574263586776485268035499744454834741611401590336496.";
        EXPECT_EQ(first ^ second, "-7438548282325391861115580462051234195051365404837043084065596544273393433543.");

        Aesi<352> third = "-59858022913898207239847737374280826888103281430894219245907359653709969931649.";
        Aesi<256> forth = "6050912656343368400621911653017065864161448053299600116145.";
        third ^= forth; EXPECT_EQ(third, "-59858022913898207240184278886073366733624668154096435329709490097237877584944.");
    }
    {
        Aesi<416> first = "114216889653717432506948448761204238653938270998706514140194067713783108574831.";
        Aesi<224> second = "2999683497236945181789030557704938712965290241120462009539.";
        EXPECT_EQ(first ^ second, "114216889653717432508366189790359373366878710208890735118819942585630140197548.");

        Aesi<384> third = "77609254042607545397349717995444569598315107315666295028581917028371247774266.";
        Aesi<224> forth = "-4207344216287446072853916296007494580803166573324288512526.";
        third ^= forth; EXPECT_EQ(third, "77609254042607545393563541513182036649880726558099863897191081411482458548276.");
    }
    {
        Aesi<320> first = "109586265226808486248911061769344550794187251256552645874665079717319926311090.";
        Aesi<256> second = "6121499926434696004340748522146589658293528248658715325381.";
        EXPECT_EQ(first ^ second, "109586265226808486243183270045386306881106920853645984488373155693028495252343.");

        Aesi<320> third = "11974407202561979719303645244269208103804810443938426289252938280080906072001.";
        Aesi<288> forth = "-1716810719556339174718834863262664101860046625237554983326.";
        third ^= forth; EXPECT_EQ(third, "11974407202561979720922376172846655867693077224603644167775715832129811340895.");
    }
    {
        Aesi<416> first = "-6340517594220183104289441247460414058213989826355852897025849060380245381218.";
        Aesi<320> second = "2906644158211689113896901117390172112973478826449683611059.";
        EXPECT_EQ(first ^ second, "-6340517594220183102267096515196597067512976355089325514198433982580168699345.");

        Aesi<352> third = "53561072822619458763935213792395363554870011019811995410013798578174981087174.";
        Aesi<256> forth = "-6105120170834023382907261389795073830642137393421059562583.";
        third ^= forth; EXPECT_EQ(third, "53561072822619458765709474245577276231834741300161977140205003037592897684369.");
    }
    {
        Aesi<320> first = "-3497697060155725785486235281195456615574963972794243150378782002178894045459.";
        Aesi<192> second = "128057515283375020238898814341003849452087595204778571056.";
        EXPECT_EQ(first ^ second, "-3497697060155725785606549824923710034572387758852886627864151015587971929123.");

        Aesi<352> third = "-101225665448196163583646222271600374897047770083898167665535334893527816454508.";
        Aesi<288> forth = "5851695208319018015419932819461373292935938187637558785475.";
        third ^= forth; EXPECT_EQ(third, "-101225665448196163579388331551432738195604616713689067794391069641489272894639.");
    }
    {
        Aesi<416> first = "65004167656690841174706626999933889672108294302303270843027672011710464666022.";
        Aesi<224> second = "-1243025077294825789612555447213426477529905494851003049289.";
        EXPECT_EQ(first ^ second, "65004167656690841175063676954872013417372015576470148256749391973213004451055.");

        Aesi<448> third = "12377424605630658178762033818300472461540920033947466610547993219128001828385.";
        Aesi<256> forth = "4007231143134804401374426256540545783769137368102425763316.";
        third ^= forth; EXPECT_EQ(third, "12377424605630658182767528358386061724592357150863968836566997059407654325205.");
    }
    {
        Aesi<320> first = "-86118806024233802989558608026890603039362764260412020932991748140092353800863.";
        Aesi<288> second = "-6156922909607417853799960807220935323246039793130092299660.";
        EXPECT_EQ(first ^ second, "-86118806024233802985464428616740879948539575570179371098662340849911089262355.");

        Aesi<384> third = "99411515310250768548390580350768770527292041615044628685459302195798122336704.";
        Aesi<320> forth = "2357066091775141782757821914279336949750155941647800223215.";
        third ^= forth; EXPECT_EQ(third, "99411515310250768549172072660774399663587690169218112029116240003548639974447.");
    }
    {
        Aesi<448> first = "30673161869447446087710628616293176713571067683470166504762926857499611250773.";
        Aesi<192> second = "2086762364725935357926874189403209837075137709779509852862.";
        EXPECT_EQ(first ^ second, "30673161869447446089547438148610373113657404068061276871708932932774935952107.");

        Aesi<352> third = "-18606004061941757684559089369235627286506576219348671817550700698218995290290.";
        Aesi<320> forth = "2007712654228494578597700462776515414133079442783891300325.";
        third ^= forth; EXPECT_EQ(third, "-18606004061941757682594510716044431610790072860373251943240233678904811687767.");
    }
    {
        Aesi<352> first = "28289351542428413638056649093536079250855062821388649142803339599426226235365.";
        Aesi<224> second = "-2756622692924885940520674220176630529075621475460154586687.";
        EXPECT_EQ(first ^ second, "28289351542428413640014025927962398073580231786004824456716942071692659708378.");

        Aesi<416> third = "32177760742553569526643049849393443524837258038221432539898162317403539052736.";
        Aesi<256> forth = "4867300341234196538583415668441962554059878465463517097555.";
        third ^= forth; EXPECT_EQ(third, "32177760742553569524938826727809170563980273904426579323322200066311562311315.");
    }
    {
        Aesi<416> first = "-73949397179831443490161161409930530926481370555899030118416353652880477185519.";
        Aesi<320> second = "1378153897929082241371021509790661856798205471661757303965.";
        EXPECT_EQ(first ^ second, "-73949397179831443489963124906300712665930653832676997403745494013174357092722.");

        Aesi<416> third = "65853188090826968330308076741895729884002142943728838242458563266589222320886.";
        Aesi<256> forth = "-738229036379895933042910421420538087881352551578727742268.";
        third ^= forth; EXPECT_EQ(third, "65853188090826968329672794664887202618235736445177456481256875477435895562698.");
    }
    {
        Aesi<448> first = "81437156913239400956331548767454825964027849275285571654615160816395604261163.";
        Aesi<224> second = "291210483635221278667370448706478446868323204890786040449.";
        EXPECT_EQ(first ^ second, "81437156913239400956083272133689333980090718164908786035560728125503753450410.");

        Aesi<352> third = "-63826226824473130704565945284210924849342895139967320337043051375758750047201.";
        Aesi<320> forth = "3529018893722821021503353055518307764491102980204076895315.";
        third ^= forth; EXPECT_EQ(third, "-63826226824473130701061455298710853530197631453949179316517834810893664933810.");
    }
    {
        Aesi<352> first = "2749575869579591329628629964591502015165500707827213567104471828740970969904.";
        Aesi<288> second = "2911678920176451906885942355407509485204852884872583952392.";
        EXPECT_EQ(first ^ second, "2749575869579591329203826522863066444817503875302595518546884711194585678648.");

        Aesi<320> third = "-43012010900528730941889803781908157518902156831388190474352149674439111838297.";
        Aesi<320> forth = "-1955630495501632765094949870378340656532081244018908641924.";
        third ^= forth; EXPECT_EQ(third, "-43012010900528730939946624836066107934996410772740244456338824350698174099677.");
    }
    {
        Aesi<448> first = "82645724364138123132055837384889626455033423869266189335443260689832426052692.";
        Aesi<320> second = "2378552689908353339100306766797941886836278724144465919993.";
        EXPECT_EQ(first ^ second, "82645724364138123132865079094850790386367721775327029227039778766196628562861.");

        Aesi<320> third = "-65946550500353357297136099354099914031702818906061042688963674043975250769223.";
        Aesi<320> forth = "-1745572081448783477986398824569942805178241380219676491279.";
        third ^= forth; EXPECT_EQ(third, "-65946550500353357295488610963433331625328646963246225400363604324122099387208.");
    }
    {
        Aesi<352> first = "-26276146569419674529288277161465912892398761765898095030213069395192703127139.";
        Aesi<256> second = "-5780246864857993287054000139895578072418293929519046753648.";
        EXPECT_EQ(first ^ second, "-26276146569419674533348968248500672387940579190809365833899985187382890918675.");

        Aesi<320> third = "48263791050037963478014884010701895571265128844352457800460804414924378550076.";
        Aesi<192> forth = "2318082455823299195677719652341567427509309717528987283624.";
        third ^= forth; EXPECT_EQ(third, "48263791050037963478958071872464070149083156196113962091652039759792266846100.");
    }
    {
        Aesi<384> first = "96031357767415375111683166504963932533766477968582481170228124524738722534047.";
        Aesi<288> second = "-3961347553144184275500960695051542046265395832726472706508.";
        EXPECT_EQ(first ^ second, "96031357767415375107797681834238487852819378047499308055993682782550299990867.");

        Aesi<384> third = "-1541656332023294419039927617059536359269630729027228764980457421500371058563.";
        Aesi<256> forth = "5176752448613683795935552654989429421451399522491351525774.";
        third ^= forth; EXPECT_EQ(third, "-1541656332023294421028849138470606476249024343437759720642088363779981792781.");
    }
    {
        Aesi<384> first = "94901637270971671318474898637997958094880595625409687936698108965884030002278.";
        Aesi<288> second = "2778573104825540177599070888390218846661157776114052156296.";
        EXPECT_EQ(first ^ second, "94901637270971671318102636864488431048948488687470447671255984319869719179246.");

        Aesi<448> third = "16337798837897417291447994002684014965700528011822036388339889761557514205149.";
        Aesi<224> forth = "580619006620176335829094100137388914319230593697521957805.";
        third ^= forth; EXPECT_EQ(third, "16337798837897417292019991703108853425159528601831973427504761108113427516528.");
    }
    {
        Aesi<320> first = "19209640729299931085673249337785947581449107283995375051503436865418453650421.";
        Aesi<192> second = "3960591960926673276811511147021682479416582490372430379208.";
        EXPECT_EQ(first ^ second, "19209640729299931089632596017285470589566340527594670795428691601102693282621.");

        Aesi<352> third = "-94313446509853684373232414063588198568349280692486382078090952301584677582095.";
        Aesi<192> forth = "5848677167581492631887032486688376278599195876282044504405.";
        third ^= forth; EXPECT_EQ(third, "-94313446509853684375843118736035081071013234674577983460569609289782556126298.");
    }
    {
        Aesi<352> first = "15266383949784996890984071368352160436368034598115907331733717596975510646546.";
        Aesi<320> second = "-3897124490819619921682028222995909596438647531018520716779.";
        EXPECT_EQ(first ^ second, "15266383949784996893573641917527477231446480009768281204151568815515103759097.");

        Aesi<352> third = "107916976661968908862023197626605272807363110937187123756980073261436544189233.";
        Aesi<256> forth = "-4345265053790129409809621498212644821600610930387229933974.";
        third ^= forth; EXPECT_EQ(third, "107916976661968908866308947511951264512156804255372810269219088200847681028775.");
    }
    {
        Aesi<352> first = "-11960778347456109577076816575072769784515897426595693222021990635046241549173.";
        Aesi<288> second = "5159243871137294986192744032761676612479968249261053126246.";
        EXPECT_EQ(first ^ second, "-11960778347456109581432840986783813264560272422385540577132495058293234638099.");

        Aesi<320> third = "96374556157265769577401964640888353033102226767685450393568392159222742125274.";
        Aesi<288> forth = "1099567263478431999083433604935975567872784936904523931042.";
        third ^= forth; EXPECT_EQ(third, "96374556157265769576515414643432437558723494490201376183402676170490131666808.");
    }
    {
        Aesi<384> first = "21210017189062563870675417266764464151145812124156987448194812096846278335687.";
        Aesi<288> second = "-128243119559901777215716808123871146081922842487719066339.";
        EXPECT_EQ(first ^ second, "21210017189062563870553454176355423529902317850273759007441668712851976447524.");

        Aesi<384> third = "-48727989445331916372580666525793356790269588776865532507353480146791816026765.";
        Aesi<288> forth = "-1757658821920339410453582721934352836781099295679635019823.";
        third ^= forth; EXPECT_EQ(third, "-48727989445331916374281526997489035598537328082932065358108113094599983268514.");
    }
    {
        Aesi<320> first = "77250519992308600126649588221030942211235636842482333220550386364626143803296.";
        Aesi<224> second = "1318036324507742804153107635878390455332046134233995061987.";
        EXPECT_EQ(first ^ second, "77250519992308600125380628025492757616028422951196693185009783439721519209795.");

        Aesi<320> third = "-67890229666935805288050627706613771253727956534138460667642438447420606290437.";
        Aesi<320> forth = "820936364922008051879798238888291351799585828242635186828.";
        third ^= forth; EXPECT_EQ(third, "-67890229666935805288851114372122306139587254191409540647836215952738553559177.");
    }
    {
        Aesi<352> first = "-90473006983603008810776981946961070347219210177360924253323412243003791697170.";
        Aesi<320> second = "-4283115548412205313120704624605460670685718259838245230848.";
        EXPECT_EQ(first ^ second, "-90473006983603008807114814447298988640586032733203153781840201882632591358994.");

        Aesi<384> third = "-56073621987417914210492311679179495173984469122489764259950247004470747105877.";
        Aesi<224> forth = "-4816772669846485604260834224266777736407910096224966739240.";
        third ^= forth; EXPECT_EQ(third, "-56073621987417914205691115389804371655402279545793629453134000977666556399485.");
    }
    {
        Aesi<448> first = "-83193535780629828813482473320553306090705194973109999428338942404141078521744.";
        Aesi<224> second = "-1690314085265694176975834022398157864770271982972028888303.";
        EXPECT_EQ(first ^ second, "-83193535780629828815160023229271593480644507446949689266072905185025844773759.");

        Aesi<384> third = "-49940436120273954283422393760804600306598590122163994506019949013368622247686.";
        Aesi<288> forth = "769263516266918279747556174867366615623267221838444462154.";
        third ^= forth; EXPECT_EQ(third, "-49940436120273954283783354688322117576783303946175975910732141528008046341964.");
    }
    {
        Aesi<384> first = "89451077944398504026161757568114570994145873940611634552002638658563226754335.";
        Aesi<288> second = "4591820560991812797358034372077664626734528262713282120502.";
        EXPECT_EQ(first ^ second, "89451077944398504029527565981399840431556675946331504927438261331779407812137.");

        Aesi<352> third = "-66282549291636009078051417574105178487893015402408617668283227391535859601499.";
        Aesi<224> forth = "-568389082961950212184733899439435230753214915176787353646.";
        third ^= forth; EXPECT_EQ(third, "-66282549291636009077686852105342182199340638515574798568399505399865336639605.");
    }
    {
        Aesi<352> first = "48060135072505190598974202293968033511501917629059956486481039243357978650664.";
        Aesi<192> second = "4694913037634202575528739587435934550665816594280177198458.";
        EXPECT_EQ(first ^ second, "48060135072505190602530461785169171461851163843682018912523382755700189400402.");

        Aesi<384> third = "75323385331160001796895636900049014682561705005544415189938137604215021768174.";
        Aesi<320> forth = "-3484967930148281169805980604707343704097115389040861372555.";
        third ^= forth; EXPECT_EQ(third, "75323385331160001800374379031484065288229963472172902652510029146222425009509.");
    }
    {
        Aesi<320> first = "109254473891992017158909066089113057924706686383217306075024847704343193572738.";
        Aesi<256> second = "-3710899057450997415870270912984700338267900780385145001042.";
        EXPECT_EQ(first ^ second, "109254473891992017156326472869069674096459368599288936398338959559999188812240.");

        Aesi<416> third = "-111270325754917620078207112649584570012154911019360372320003926218076334133596.";
        Aesi<224> forth = "-3302007719739999073817680359924862347931186235771280429447.";
        third ^= forth; EXPECT_EQ(third, "-111270325754917620074907129097920751757009735521451904004540494123803693875419.");
    }
    {
        Aesi<416> first = "-110158920207354992921334322361076794696945492738569815695128704954407437615918.";
        Aesi<288> second = "4844273801817723328025354796919748586575215704029072360614.";
        EXPECT_EQ(first ^ second, "-110158920207354992922819270155580045211694811599034036755911557070944407963528.");

        Aesi<320> third = "3285826390047916477747590779228864567982795715661285355968136118075611753527.";
        Aesi<288> forth = "-6066896152594693373644911525357257057680913375667341551901.";
        third ^= forth; EXPECT_EQ(third, "3285826390047916483746224340985642559129344452520709776083334040513278493994.");
    }
}