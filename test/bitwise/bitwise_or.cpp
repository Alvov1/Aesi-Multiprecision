#include <gtest/gtest.h>
#include "../../Aesi.h"
#include "../benchmarks/benchmarks.h"

TEST(Bitwise, OR) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi < 96 > m0 = 2020645940044775524ULL, m1 = 5357268711087358525ULL;
    EXPECT_EQ(m0 | m1, 6798983837096166013);
    Aesi < 96 > m2 = 1525558537524866468ULL, m3 = 780162980750373835ULL;
    EXPECT_EQ(m2 | m3, 2304716529481809903);
    Aesi < 96 > m4 = 5499392792818025779ULL, m5 = 3630553699293589542ULL;
    EXPECT_EQ(m4 | m5, 9111861610500060471);
    Aesi < 96 > m6 = 8217820161313030374ULL, m7 = 4471571076368997795ULL;
    EXPECT_EQ(m6 | m7, 9083690044765232615);
    Aesi < 96 > m8 = 5382728063723026549ULL, m9 = 6474072337812696999ULL;
    EXPECT_EQ(m8 | m9, 6628118875407826935);
    Aesi < 96 > m10 = 5531067238533613298ULL, m11 = 6205920378303271579ULL;
    EXPECT_EQ(m10 | m11, 6836429038125307643);
    Aesi < 96 > m12 = 726801461705099060ULL, m13 = 7197119885199483082ULL;
    EXPECT_EQ(m12 | m13, 7779790748433508350);
    Aesi < 96 > m14 = 2857227495610299548ULL, m15 = 6851320276054652512ULL;
    EXPECT_EQ(m14 | m15, 9202806212469438204);
    Aesi < 96 > m16 = 8647879979545733973ULL, m17 = 7657869608651681530ULL;
    EXPECT_EQ(m16 | m17, 8811142956979941375);
    Aesi < 96 > m18 = 7993340150400714359ULL, m19 = 5363863660180581854ULL;
    EXPECT_EQ(m18 | m19, 7997915288348700671);
    Aesi < 96 > m20 = 4407478874354858993ULL, m21 = 2858798932850881759ULL;
    EXPECT_EQ(m20 | m21, 4588885444677582847);
    Aesi < 96 > m22 = 148798269284364709ULL, m23 = 1937830191614830151ULL;
    EXPECT_EQ(m22 | m23, 1942370298493646823);
    Aesi < 96 > m24 = 9162479857370113771ULL, m25 = 3760340113672901768ULL;
    EXPECT_EQ(m24 | m25, 9164803164194977515);
    Aesi < 96 > m26 = 192295714791438702ULL, m27 = 3279685136564267485ULL;
    EXPECT_EQ(m26 | m27, 3435098554140178943);
    Aesi < 96 > m28 = 624540544870946014ULL, m29 = 5129147622974336686ULL;
    EXPECT_EQ(m28 | m29, 5741795781346377470);
    Aesi < 96 > m30 = 3917631123186086388ULL, m31 = 3272514707466701547ULL;
    EXPECT_EQ(m30 | m31, 4575233676275579903);
    Aesi < 96 > m32 = 5834378838445282859ULL, m33 = 5114723924435243853ULL;
    EXPECT_EQ(m32 | m33, 6268983122486351727);
    Aesi < 96 > m34 = 3924055531914283595ULL, m35 = 5293342119300753052ULL;
    EXPECT_EQ(m34 | m35, 9184453447320661727);
    Aesi < 96 > m36 = 8301970154162996009ULL, m37 = 4428614890117278083ULL;
    EXPECT_EQ(m36 | m37, 9184983549889473451);
    Aesi < 96 > m38 = 3615782113039458075ULL, m39 = 5873554648245063345ULL;
    EXPECT_EQ(m38 | m39, 8336126617688334267);

    Aesi < 96 > r0 = 2047835587796591202ULL, r1 = 5443049987772906675ULL;
    r0 |= r1; EXPECT_EQ(r0, 6911864060667887347);
    Aesi < 96 > r2 = 4510151453757533823ULL, r3 = 2286555397785640804ULL;
    r2 |= r3; EXPECT_EQ(r2, 4593529636695048063);
    Aesi < 96 > r4 = 4388167193733001566ULL, r5 = 8839533454280551609ULL;
    r4 |= r5; EXPECT_EQ(r4, 9146237825563950591);
    Aesi < 96 > r6 = 9013148001953800496ULL, r7 = 8934686380896798720ULL;
    r6 |= r7; EXPECT_EQ(r6, 9223200508427102512);
    Aesi < 96 > r8 = 3629301352859462527ULL, r9 = 486578371779714171ULL;
    r8 |= r9; EXPECT_EQ(r8, 3953595854238613375);
    Aesi < 96 > r10 = 3124590658226067896ULL, r11 = 4571240866427317772ULL;
    r10 |= r11; EXPECT_EQ(r10, 4574760016669032380);
    Aesi < 96 > r12 = 2012961834446362382ULL, r13 = 1947448811664489642ULL;
    r12 |= r13; EXPECT_EQ(r12, 2013102985543729070);
    Aesi < 96 > r14 = 292813269398872651ULL, r15 = 7327551113656379806ULL;
    r14 |= r15; EXPECT_EQ(r14, 7327630372111117279);
    Aesi < 96 > r16 = 8530393272709473230ULL, r17 = 1860394511317219841ULL;
    r16 |= r17; EXPECT_EQ(r16, 9219849200781425615);
    Aesi < 96 > r18 = 2063171400442100177ULL, r19 = 1870769038441214218ULL;
    r18 |= r19; EXPECT_EQ(r18, 2159440800362985947);
    Aesi < 96 > r20 = 8117769925308468990ULL, r21 = 6293667702621214049ULL;
    r20 |= r21; EXPECT_EQ(r20, 8646803527032528895);
    Aesi < 96 > r22 = 2778498898382626760ULL, r23 = 4502673125009460622ULL;
    r22 |= r23; EXPECT_EQ(r22, 4539548572356124622);
    Aesi < 96 > r24 = 6203857817238052914ULL, r25 = 6439600255934219825ULL;
    r24 |= r25; EXPECT_EQ(r24, 6872088209224459827);
    Aesi < 96 > r26 = 8327999777747565718ULL, r27 = 2963108099365356848ULL;
    r26 |= r27; EXPECT_EQ(r26, 8908119779469819318);
    Aesi < 96 > r28 = 8345926349317658801ULL, r29 = 5439611742273630497ULL;
    r28 |= r29; EXPECT_EQ(r28, 8935123862071802289);
    Aesi < 96 > r30 = 589517085897666687ULL, r31 = 1699142349415543288ULL;
    r30 |= r31; EXPECT_EQ(r30, 2287532837244269055);
    Aesi < 96 > r32 = 4895087935113931520ULL, r33 = 4397533764706897032ULL;
    r32 |= r33; EXPECT_EQ(r32, 9218865872828116872);
    Aesi < 96 > r34 = 8787699061185064661ULL, r35 = 6278261499548717087ULL;
    r34 |= r35; EXPECT_EQ(r34, 9220273329485446879);
    Aesi < 96 > r36 = 3517043394867606010ULL, r37 = 7106425036492048894ULL;
    r36 |= r37; EXPECT_EQ(r36, 8277369822149327358);
    Aesi < 96 > r38 = 2440620310881118278ULL, r39 = 5365949614884060230ULL;
    r38 |= r39; EXPECT_EQ(r38, 7782219880780987462);

    Aesi512 y0 = -7379608232792186683; EXPECT_EQ(y0 | 0, -7379608232792186683); Aesi512 y1 = 2970120816951561012; EXPECT_EQ(y1 | 0, 2970120816951561012);
    Aesi512 y2 = -5696220288572448165; EXPECT_EQ(y2 | 0, -5696220288572448165); Aesi512 y3 = 7900362436724175479; EXPECT_EQ(y3 | 0, 7900362436724175479);
    Aesi512 y4 = -777376026862911334; EXPECT_EQ(y4 | 0, -777376026862911334); Aesi512 y5 = -2069419668651330974; EXPECT_EQ(y5 | 0, -2069419668651330974);
    Aesi512 y6 = 1259778954916885065; EXPECT_EQ(y6 | 0, 1259778954916885065); Aesi512 y7 = -2905725874441331650; EXPECT_EQ(y7 | 0, -2905725874441331650);
    Aesi512 y8 = 4879706157249442504; EXPECT_EQ(y8 | 0, 4879706157249442504); Aesi512 y9 = 8027626211878286429; EXPECT_EQ(y9 | 0, 8027626211878286429);
    Aesi512 y10 = 570797157455018920; EXPECT_EQ(y10 | 0, 570797157455018920); Aesi512 y11 = -427390551710800597; EXPECT_EQ(y11 | 0, -427390551710800597);
    Aesi512 y12 = -2642894643427682157; EXPECT_EQ(y12 | 0, -2642894643427682157); Aesi512 y13 = -2738503629311636134; EXPECT_EQ(y13 | 0, -2738503629311636134);
    Aesi512 y14 = 6337689599550252319; EXPECT_EQ(y14 | 0, 6337689599550252319); Aesi512 y15 = -5586759305152622746; EXPECT_EQ(y15 | 0, -5586759305152622746);
    Aesi512 y16 = 5456108041498276808; EXPECT_EQ(y16 | 0, 5456108041498276808); Aesi512 y17 = 5452584994720316618; EXPECT_EQ(y17 | 0, 5452584994720316618);
    Aesi512 y18 = -2316362324001525966; EXPECT_EQ(y18 | 0, -2316362324001525966); Aesi512 y19 = 7325722336685152086; EXPECT_EQ(y19 | 0, 7325722336685152086);
    Aesi512 y20 = 377998544433893902; EXPECT_EQ(y20 | 0, 377998544433893902); Aesi512 y21 = -2465288838779088870; EXPECT_EQ(y21 | 0, -2465288838779088870);
    Aesi512 y22 = -5468006971160031732; EXPECT_EQ(y22 | 0, -5468006971160031732); Aesi512 y23 = 3506965552698968660; EXPECT_EQ(y23 | 0, 3506965552698968660);
    Aesi512 y24 = -7300963666391738798; EXPECT_EQ(y24 | 0, -7300963666391738798); Aesi512 y25 = 3103134765104197354; EXPECT_EQ(y25 | 0, 3103134765104197354);
    Aesi512 y26 = 3165186477704156988; EXPECT_EQ(y26 | 0, 3165186477704156988); Aesi512 y27 = 1709646565921426320; EXPECT_EQ(y27 | 0, 1709646565921426320);
    Aesi512 y28 = 1097774869699502254; EXPECT_EQ(y28 | 0, 1097774869699502254); Aesi512 y29 = 2491211643943272387; EXPECT_EQ(y29 | 0, 2491211643943272387);
    Aesi512 y30 = -8927353262635906157; EXPECT_EQ(y30 | 0, -8927353262635906157); Aesi512 y31 = -38766286252702324; EXPECT_EQ(y31 | 0, -38766286252702324);
    Aesi512 y32 = 5514084479833422753; EXPECT_EQ(y32 | 0, 5514084479833422753); Aesi512 y33 = 2194479717102096810; EXPECT_EQ(y33 | 0, 2194479717102096810);
    Aesi512 y34 = 2751242616132452994; EXPECT_EQ(y34 | 0, 2751242616132452994); Aesi512 y35 = -2390446791466101241; EXPECT_EQ(y35 | 0, -2390446791466101241);
    Aesi512 y36 = 8822853861241688997; EXPECT_EQ(y36 | 0, 8822853861241688997); Aesi512 y37 = -4950353505985954258; EXPECT_EQ(y37 | 0, -4950353505985954258);
    Aesi512 y38 = -8136431100828633976; EXPECT_EQ(y38 | 0, -8136431100828633976); Aesi512 y39 = 8055314313686600301; EXPECT_EQ(y39 | 0, 8055314313686600301);
    Aesi512 y40 = 3365309689234796764; EXPECT_EQ(y40 | 0, 3365309689234796764); Aesi512 y41 = 8794773495649163247; EXPECT_EQ(y41 | 0, 8794773495649163247);
    Aesi512 y42 = -515275048992317636; EXPECT_EQ(y42 | 0, -515275048992317636); Aesi512 y43 = 3929495410971836324; EXPECT_EQ(y43 | 0, 3929495410971836324);
    Aesi512 y44 = 5301875904659701242; EXPECT_EQ(y44 | 0, 5301875904659701242); Aesi512 y45 = -3342985643104422805; EXPECT_EQ(y45 | 0, -3342985643104422805);
    Aesi512 y46 = -8770275292430445047; EXPECT_EQ(y46 | 0, -8770275292430445047); Aesi512 y47 = -7256290758481884326; EXPECT_EQ(y47 | 0, -7256290758481884326);
    Aesi512 y48 = -6395402439121471762; EXPECT_EQ(y48 | 0, -6395402439121471762); Aesi512 y49 = 8481418200485971540; EXPECT_EQ(y49 | 0, 8481418200485971540);
    Aesi512 y50 = -6826837351772736686; EXPECT_EQ(y50 | 0, -6826837351772736686); Aesi512 y51 = 5378333528058019140; EXPECT_EQ(y51 | 0, 5378333528058019140);
    Aesi512 y52 = 7949354112667923489; EXPECT_EQ(y52 | 0, 7949354112667923489); Aesi512 y53 = 758475868520796204; EXPECT_EQ(y53 | 0, 758475868520796204);
    Aesi512 y54 = -6506001218570206180; EXPECT_EQ(y54 | 0, -6506001218570206180); Aesi512 y55 = -7151596374648398466; EXPECT_EQ(y55 | 0, -7151596374648398466);
    Aesi512 y56 = -2312960319391550338; EXPECT_EQ(y56 | 0, -2312960319391550338); Aesi512 y57 = -205899936757631221; EXPECT_EQ(y57 | 0, -205899936757631221);
    Aesi512 y58 = -8988800751555513056; EXPECT_EQ(y58 | 0, -8988800751555513056); Aesi512 y59 = -5137089545283585480; EXPECT_EQ(y59 | 0, -5137089545283585480);

    Logging::addRecord("Bitwise_OR",
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());;
}

TEST(Bitwise, DifferentPrecisionOR) {
    {
        Aesi < 384 > first = "46221828550383548671803689981323585789309810345838952982919178641296601829066.";
        Aesi < 256 > second = "6200415520367090353171802511135999424687148510071774509794.";
        EXPECT_EQ(first | second, "46221828550383548673975434459295650877043740365634515208999257693296983539434.");

        Aesi < 320 > third = "-86582451695755212696197158387178075350298844513431632265869987089621360392922.";
        Aesi < 256 > forth = "-136364000506691303813165579161739687768394246053898549458.";
        third |= forth; EXPECT_EQ(third, "-86582451695755212696235243530316223704466964855412642128216807575229325769434.");
    }
    {
        Aesi < 384 > first = "-46221828550383548671803689981323585789309810345838952982919178641296601829066.";
        Aesi < 256 > second = "6200415520367090353171802511135999424687148510071774509794.";
        EXPECT_EQ(first | second, "-46221828550383548673975434459295650877043740365634515208999257693296983539434.");

        Aesi < 320 > third = "86582451695755212696197158387178075350298844513431632265869987089621360392922.";
        Aesi < 256 > forth = "-136364000506691303813165579161739687768394246053898549458.";
        third |= forth; EXPECT_EQ(third, "86582451695755212696235243530316223704466964855412642128216807575229325769434.");
    }
    {
        Aesi < 352 > first = "-48709546203770858091244258034139682067405085241382288597585194396155623610883.";
        Aesi < 288 > second = "1430888126194076391632801377020460168282593925879063994202.";
        EXPECT_EQ(first | second, "-48709546203770858092422271568239806411423834826317868858069186933675281735515.");

        Aesi < 448 > third = "-98163424244583349783272619780878624982351029984357235806595480931137289429060.";
        Aesi < 256 > forth = "-5387395878352056304235249922650367974608065874012852471206.";
        third |= forth; EXPECT_EQ(third, "-98163424244583349785053858691545071466933694342173933201721864217305561163238.");
    }
    {
        Aesi < 352 > first = "69484720740330411164426034664912585846212085398136074828444953194972432897245.";
        Aesi < 288 > second = "5583299350392958780179489220763592751535525567357374245815.";
        EXPECT_EQ(first | second, "69484720740330411167617108464274767649241311371569910925488368187689221832703.");

        Aesi < 384 > third = "39132688648825796936471201962618052033168559823199788069847595325981593695523.";
        Aesi < 224 > forth = "4645766705285778938096989561903058999891365757822266438915.";
        third |= forth; EXPECT_EQ(third, "39132688648825796937776121928818921397295301735200952356958947245675257001251.");
    }
    {
        Aesi < 352 > first = "-103738240934674960986379078448802675680236849441138101291684756322516115743178.";
        Aesi < 256 > second = "3104939837795390092948498080059892739865457982945428973915.";
        EXPECT_EQ(first | second, "-103738240934674960987375200784354913666381582623815177712982280696737183001051.");

        Aesi < 352 > third = "67592052690489413760295375911469011384901557392110257979159386203713187520019.";
        Aesi < 192 > forth = "2244334311125138880392089666114494364666751271123312016687.";
        third |= forth; EXPECT_EQ(third, "67592052690489413762281502116889598241520278323889624079494910290093634748223.");
    }
    {
        Aesi < 448 > first = "76752495526065054733976248158404938255342790304989726459192902751986165265328.";
        Aesi < 192 > second = "495589113104539174595259910369819520346421662335516936504.";
        EXPECT_EQ(first | second, "76752495526065054734371823969658243843611548071956984587965024821576146533304.");

        Aesi < 448 > third = "91615730622339808100046881288038243507870144947834578409522955695534720335648.";
        Aesi < 288 > forth = "2628201840772295691060081519106483597184126537896522686091.";
        third |= forth; EXPECT_EQ(third, "91615730622339808102649310142997501299450239458543659046659364881760067174315.");
    }
    {
        Aesi < 416 > first = "15772200816371374827083903579379733042742697002210598904666923902203206064883.";
        Aesi < 256 > second = "5606953445930285552814582169223836594761637526138652234764.";
        EXPECT_EQ(first | second, "15772200816371374832591813033648036382261329377098379070436069253199871926015.");

        Aesi < 352 > third = "-36155086291353127922050183020483309168889820971726920810820466138672837990218.";
        Aesi < 224 > forth = "3973319801150395844846791049657410769602155620652093548280.";
        third |= forth; EXPECT_EQ(third, "-36155086291353127925973409019544171509521851589830011761464577920044546963450.");
    }
    {
        Aesi < 416 > first = "73951008172966829480189659198560292004098601490753413305616876862656954658316.";
        Aesi < 288 > second = "-2438438598101727741533547726417552066902648557548502294952.";
        EXPECT_EQ(first | second, "73951008172966829481788055228744681933979013899562507028039313162698789679020.");

        Aesi < 384 > third = "-96244525607652019657397679525706089893840544545583852682014251800836043660296.";
        Aesi < 224 > forth = "-1676716566652170495098096217737276714945597169092562767824.";
        third |= forth; EXPECT_EQ(third, "-96244525607652019659071308559072055944911159516699245203340466659632477102040.");
    }
    {
        Aesi < 320 > first = "49607756871400128259691988893127860722939170577218823462405421709931153689934.";
        Aesi < 256 > second = "-1501754870462735612682965440814990895341862671792072070913.";
        EXPECT_EQ(first | second, "49607756871400128260697976962362511244370291253405370918550081043953159741263.");

        Aesi < 384 > third = "9815134023030734725212430732083991554256591366529240815547565456373066606614.";
        Aesi < 320 > forth = "-16501220801857935205155693691333985220455588310504416603.";
        third |= forth; EXPECT_EQ(third, "9815134023030734725212814651579960042917794070560101496450695374407336963423.");
    }
    {
        Aesi < 384 > first = "-68240968110386693298200521965846216919562784333604280415650153946166248501196.";
        Aesi < 320 > second = "4674141496001557840032974155124745921873144432523996180794.";
        EXPECT_EQ(first | second, "-68240968110386693299279404930231463290279206339400573876039780274547294371838.");

        Aesi < 320 > third = "62235779636897389969954022082309859982355382995056108846294072819268397351985.";
        Aesi < 288 > forth = "-4428721367786790329459725032631824837304356848971930532088.";
        third |= forth; EXPECT_EQ(third, "62235779636897389973892344077943545666288241523695556944557367452841493003513.");
    }
    {
        Aesi < 320 > first = "-2184376944847706296836680020764739441470522176519261992943353867430934520116.";
        Aesi < 256 > second = "-2534931283450050797705226322576285725720697727432116932346.";
        EXPECT_EQ(first | second, "-2184376944847706297694938368939370019436398934659209998823995851876162252798.");

        Aesi < 384 > third = "-11916748102685320107134862867047062080014007074073260220966138461055468035460.";
        Aesi < 224 > forth = "6200463017497022744115470453921028952333842869379853511795.";
        third |= forth; EXPECT_EQ(third, "-11916748102685320112040121515002707157546382842962441646585417834114926378487.");
    }
    {
        Aesi < 384 > first = "-4788419524022100461044907824792354082635339435355079343581184186366074170508.";
        Aesi < 224 > second = "-2552736234698278001003834333573584069150673881970426768825.";
        EXPECT_EQ(first | second, "-4788419524022100461242030676633276266916085655551915457095012948892252630461.");

        Aesi < 448 > third = "-114494178122579045792130966709935511919715093874598422805214757664769928218209.";
        Aesi < 320 > forth = "4437772134380824660882673405088482570029023477697839582426.";
        third |= forth; EXPECT_EQ(third, "-114494178122579045792921399456231775556229863779611273246804741948398037323515.");
    }
    {
        Aesi < 416 > first = "21629402376844582370076045237386817018702611235921120697970027560688947743391.";
        Aesi < 256 > second = "5955981833897658771555559420670755619257134127332853717385.";
        EXPECT_EQ(first | second, "21629402376844582374014971837607203374995512230604195287721202973506683198367.");

        Aesi < 384 > third = "-58604546921252061415782032100651304303740174302683643072590302516636727210849.";
        Aesi < 288 > forth = "3364268928260023369081528387542463649733658487351982168682.";
        third |= forth; EXPECT_EQ(third, "-58604546921252061416006211205438671594586335236081733057159097040697550535531.");
    }
    {
        Aesi < 448 > first = "-64953783172872426755872097854389709286177075851103824052723103539152973655724.";
        Aesi < 224 > second = "-5072458949798190504490040230713471571764591732601447477754.";
        EXPECT_EQ(first | second, "-64953783172872426759311221859423170359350700348570621695922238600120408735742.");

        Aesi < 320 > third = "22786756003032093683873517927263446854389776433704376140810581356328900433674.";
        Aesi < 288 > forth = "6015017017822749641956970447090175344706659889526689033388.";
        third |= forth; EXPECT_EQ(third, "22786756003032093683899097854191717732324870445500928422119538385784323571630.");
    }
    {
        Aesi < 352 > first = "-55807037146456907436280304038943434923897001297786623754309676042096905187612.";
        Aesi < 192 > second = "5200358677575514254229678217710032260505787457893262461659.";
        EXPECT_EQ(first | second, "-55807037146456907440988729768261078838984799841590042866924272330026299620319.");

        Aesi < 320 > third = "90538376066530682391271822463092534193346867778735978563611969417502732601697.";
        Aesi < 224 > forth = "-2104277719799218510703259398255219739007947010794522021150.";
        third |= forth; EXPECT_EQ(third, "90538376066530682391676502352418828116162745569557348830345327716620375575935.");
    }
    {
        Aesi < 416 > first = "-54226172443937446348452375347235371352034967670928752408257633950126627010135.";
        Aesi < 288 > second = "-200784272764774419612511455957241579372693537980069328091.";
        EXPECT_EQ(first | second, "-54226172443937446348453935184923883379176066395436160960687736242917703857887.");

        Aesi < 320 > third = "114737159617371717679531697386187951302288002489683407985764764153775367134069.";
        Aesi < 224 > forth = "-2730519406776103682513742995869512131539035411747053111608.";
        third |= forth; EXPECT_EQ(third, "114737159617371717681279796080714631241539984714014448536393456468454077766525.");
    }
    {
        Aesi < 448 > first = "20577552298723799323480226921155794363700292859470955136690499977988815050772.";
        Aesi < 320 > second = "-1118777258339803894429236210225272961742222347374629174518.";
        EXPECT_EQ(first | second, "20577552298723799324292451174236240075368589005293150926860946941947024506102.");

        Aesi < 448 > third = "27725416948270402186852330011774118177371766004042726229237148507598849863807.";
        Aesi < 256 > forth = "-3383813184027000226576661107168509956291215759004595454342.";
        third |= forth; EXPECT_EQ(third, "27725416948270402190039968776473212078224479317305056229815626650944804844031.");
    }
    {
        Aesi < 352 > first = "89476408807276198713515705367407944704823457666683318660124457972920698219964.";
        Aesi < 256 > second = "4374715988925107120043594291070795637447312878094202018780.";
        EXPECT_EQ(first | second, "89476408807276198717886589367730077268169351858844638138877705995833758433276.");

        Aesi < 352 > third = "38310965107576982412476086576495230357548336834184741663431989508325596678442.";
        Aesi < 192 > forth = "-5656999668023741087166070375102879591942321217448341656231.";
        third |= forth; EXPECT_EQ(third, "38310965107576982415729606412591144785252597109336797131614280203162835475375.");
    }
    {
        Aesi < 384 > first = "-45678087558307209439660535455440814752181102150881298259594711296820602260690.";
        Aesi < 256 > second = "3059186912956997058217790794018613183180725319047856880183.";
        EXPECT_EQ(first | second, "-45678087558307209440457529128118757647719799307770338353546635193733584649975.");

        Aesi < 320 > third = "-47922983542616337641504620496620018763713251892103353673888648761095776543821.";
        Aesi < 224 > forth = "637558504026635345817464046312872110472298401627244606456.";
        third |= forth; EXPECT_EQ(third, "-47922983542616337641945983076622916682367393197679129724653926050137500203005.");
    }
    {
        Aesi < 448 > first = "64176895496584542219390840923546015397035449210213006070398724456383884636677.";
        Aesi < 256 > second = "2355221475889358969523182792329295475072292299469399301998.";
        EXPECT_EQ(first | second, "64176895496584542220176355951191731227699854277488089009951896754845470539631.");

        Aesi < 320 > third = "-7224147389265436270624565753871237128106430333047702432559502521448731627600.";
        Aesi < 288 > forth = "-2007376267108295903611718178751886564171034390144949850798.";
        third |= forth; EXPECT_EQ(third, "-7224147389265436272202732522397367897984396231432327499453747540620682973950.");
    }
    {
        Aesi < 320 > first = "-97677080679810936782563215020262426938529880590218941192297306920273465280392.";
        Aesi < 288 > second = "3736661887789896502969813631568716183863896895172322371759.";
        EXPECT_EQ(first | second, "-97677080679810936782572847748464650589185491296630158603232876285957933346735.");

        Aesi < 416 > third = "-31300573833219677149573835574932569787808178235803611966995005128651017251649.";
        Aesi < 256 > forth = "-3928009971791325414131494380310354755848589661635836772157.";
        third |= forth; EXPECT_EQ(third, "-31300573833219677149575585177095791668203254392457242378285907334614981803901.");
    }
    {
        Aesi < 384 > first = "-11446583040090373605475078308686407661253780578535359300478961590761130945324.";
        Aesi < 256 > second = "5331833116310608968194825580652912015773026830142893852056.";
        EXPECT_EQ(first | second, "-11446583040090373606065102988175644356203462231333614468099675472071170453436.");

        Aesi < 320 > third = "-68707419666393645828147856613241702916226074757284942290924105507272351258859.";
        Aesi < 320 > forth = "634429482120836099755533226523752771030237282299204917147.";
        third |= forth; EXPECT_EQ(third, "-68707419666393645828543168708992238190447867181001106605380434830896771096571.");
    }
    {
        Aesi < 448 > first = "-70786232019316395353567532124018796960084696721551478628337602460190753675794.";
        Aesi < 320 > second = "-206420519447502052802903124914540137035050655734684886034.";
        EXPECT_EQ(first | second, "-70786232019316395353763991243690750667799688429456209867279356385737244671506.");

        Aesi < 352 > third = "-101387916587362736665062575606304030726554469973558360563727366205591261557626.";
        Aesi < 320 > forth = "3174669512453129202692757243325078657077915945890077758786.";
        third |= forth; EXPECT_EQ(third, "-101387916587362736665096392323081749964283085107813164330488857318513271142266.");
    }
    {
        Aesi < 448 > first = "-12257836845708347315435105376582819217967952140211556014384812246525055063506.";
        Aesi < 320 > second = "-2114561867474020185579086369377814952356109598680475475138.";
        EXPECT_EQ(first | second, "-12257836845708347317548514762049510668256071046372839342419935407542891961810.");

        Aesi < 384 > third = "41304790637645591219569765192040562669010598151199086713963155441207773778044.";
        Aesi < 256 > forth = "-3786999049347541291874128925066286530669327729211334335686.";
        third |= forth; EXPECT_EQ(third, "41304790637645591220158243912360564083549378493909034914451722796388430273790.");
    }
    {
        Aesi < 448 > first = "-92312462605157059075554709466350354490480144313883935153915729117943182472317.";
        Aesi < 288 > second = "-3424500456397332241035183176530331166080182427575291342382.";
        EXPECT_EQ(first | second, "-92312462605157059075619075588777857526568569481769256285651993851069992926847.");

        Aesi < 448 > third = "-115121142030112945715856414932761647187564920949771928182256448809279112350693.";
        Aesi < 192 > forth = "-4048105763242507991101179290551485826379516109316644289332.";
        third |= forth; EXPECT_EQ(third, "-115121142030112945719020269994317243596235416959044648362800664119638554459125.");
    }
    {
        Aesi < 416 > first = "-58842532344869953834913798431591433176671866797521520663556116509745843932542.";
        Aesi < 256 > second = "3841468437559600690099540509432251948926313197738666126268.";
        EXPECT_EQ(first | second, "-58842532344869953835417487088123362365977360497753335352539717595700647587838.");

        Aesi < 320 > third = "9077470965946562740576211980453256929958789141671945759112295916561828403954.";
        Aesi < 320 > forth = "1719779725084143624729572013995929580593409441317974414342.";
        third |= forth; EXPECT_EQ(third, "9077470965946562740579499983377940108333121427950324580530280816998740909814.");
    }
    {
        Aesi < 352 > first = "-51949188006887396075993669970515081249227344792528468327298119011573870811469.";
        Aesi < 320 > second = "4037737793948045605103363627697909704068440177177399188773.";
        EXPECT_EQ(first | second, "-51949188006887396076880434529543831365936166782326260567308031155647684730221.");

        Aesi < 320 > third = "43835758903955456399987053313422411319058636666226381064763263006017702455914.";
        Aesi < 320 > forth = "-4849348095756430172446247639060907034975299698442561091373.";
        third |= forth; EXPECT_EQ(third, "43835758903955456401660638137463799346212670859439313659807885955970224881519.");
    }
    {
        Aesi < 384 > first = "-75746795937618724268046620241498645056993814350984313049073698481989233847941.";
        Aesi < 192 > second = "-5367632638824209373537519320358535926501739130786208254095.";
        EXPECT_EQ(first | second, "-75746795937618724271442643478499095845288800813070028863249833165593344192143.");

        Aesi < 352 > third = "41393729978955478773669994604798191776815028444900719502047500428339442384104.";
        Aesi < 224 > forth = "1922195532580060323497057557255035354899963026682387949387.";
        third |= forth; EXPECT_EQ(third, "41393729978955478773967358746974150481999047572249480445886085743329643659243.");
    }
    {
        Aesi < 416 > first = "111863008462042417575723250523537763271336101872009592716033839401074108922375.";
        Aesi < 256 > second = "3975967503264040601026777178697328418658354909589064632092.";
        EXPECT_EQ(first | second, "111863008462042417578914388139928151969532970366557260406093708846846720081695.");

        Aesi < 416 > third = "-48681719552384552162546442502439738245450019072171964708233902059119163436636.";
        Aesi < 224 > forth = "2886640040143185538237483814058043223994119210293832038933.";
        third |= forth; EXPECT_EQ(third, "-48681719552384552164914244315330345107422442354506062224133939165479849950813.");
    }
    {
        Aesi < 448 > first = "-10580832856574864198027337408794828921005421157882618337126725368849199606536.";
        Aesi < 256 > second = "-6239589568521130352105003363482510413084902674322909948992.";
        EXPECT_EQ(first | second, "-10580832856574864201469329129823901598314681275331801094486317948143935549256.");

        Aesi < 416 > third = "30535789984260392016074862685596499396823165476021293317477511474592054510182.";
        Aesi < 256 > forth = "-2987088112933285911887083474739333233444982224787241199069.";
        third |= forth; EXPECT_EQ(third, "30535789984260392017461975270820487190068104017097498622511087067336377133055.");
    }
    {
        Aesi < 384 > first = "110242996963396357628456428079916663097795706280827152222256222696414433404339.";
        Aesi < 224 > second = "6081186169123916765088928103838213281718969807619852207447.";
        EXPECT_EQ(first | second, "110242996963396357629045101018070107919635302724919791789242681674985604277751.");

        Aesi < 384 > third = "111645584922399683188341396160241384529778335894428664867315798697862283131737.";
        Aesi < 320 > forth = "2511588231149330796006355168609249633602592616451335203793.";
        third |= forth; EXPECT_EQ(third, "111645584922399683188443690289872537333761263335336308661430540664903515765721.");
    }
    {
        Aesi < 448 > first = "-98702638832347071436629865249073568531588714741040267234836238263490835133731.";
        Aesi < 192 > second = "-1131340455245014915544397473522356645741877098212966557705.";
        EXPECT_EQ(first | second, "-98702638832347071436731304254478274993262172930476438560340492396095468240171.");

        Aesi < 448 > third = "40651179135155448432858702835385710555290544009091205358908779182914920083931.";
        Aesi < 192 > forth = "-3994844685394449537328714648200793993619805397935099100188.";
        third |= forth; EXPECT_EQ(third, "40651179135155448432907751531949085067073351358277665565741421136551302102495.");
    }
}