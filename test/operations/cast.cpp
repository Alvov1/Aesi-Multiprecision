#include <gtest/gtest.h>
#include "../../Multiprecision.h"

TEST(Casting, NumberCast) {
    {
        short v0 = 12660; Multiprecision o0 = v0; EXPECT_EQ(o0.integralCast<short>(), v0);
        short v1 = 1291; Multiprecision o1 = v1; EXPECT_EQ(o1.integralCast<short>(), v1);
        short v2 = 9379; Multiprecision o2 = v2; EXPECT_EQ(o2.integralCast<short>(), v2);
        short v3 = -5507; Multiprecision o3 = v3; EXPECT_EQ(o3.integralCast<short>(), v3);
        short v4 = 10503; Multiprecision o4 = v4; EXPECT_EQ(o4.integralCast<short>(), v4);
        short v5 = -15803; Multiprecision o5 = v5; EXPECT_EQ(o5.integralCast<short>(), v5);
        short v6 = -25434; Multiprecision o6 = v6; EXPECT_EQ(o6.integralCast<short>(), v6);
        short v7 = 31225; Multiprecision o7 = v7; EXPECT_EQ(o7.integralCast<short>(), v7);
        short v8 = 27368; Multiprecision o8 = v8; EXPECT_EQ(o8.integralCast<short>(), v8);
        short v9 = 3854; Multiprecision o9 = v9; EXPECT_EQ(o9.integralCast<short>(), v9);
        short v10 = -31821; Multiprecision o10 = v10; EXPECT_EQ(o10.integralCast<short>(), v10);
        short v11 = -14224; Multiprecision o11 = v11; EXPECT_EQ(o11.integralCast<short>(), v11);
        short v12 = 16812; Multiprecision o12 = v12; EXPECT_EQ(o12.integralCast<short>(), v12);
        short v13 = 24874; Multiprecision o13 = v13; EXPECT_EQ(o13.integralCast<short>(), v13);
        short v14 = 18866; Multiprecision o14 = v14; EXPECT_EQ(o14.integralCast<short>(), v14);
        short v15 = -14076; Multiprecision o15 = v15; EXPECT_EQ(o15.integralCast<short>(), v15);
        short v16 = 31911; Multiprecision o16 = v16; EXPECT_EQ(o16.integralCast<short>(), v16);
        short v17 = -8633; Multiprecision o17 = v17; EXPECT_EQ(o17.integralCast<short>(), v17);
        short v18 = 18120; Multiprecision o18 = v18; EXPECT_EQ(o18.integralCast<short>(), v18);
        short v19 = 3968; Multiprecision o19 = v19; EXPECT_EQ(o19.integralCast<short>(), v19);
    }
    {
        unsigned short v0 = 1941; Multiprecision o0 = v0; EXPECT_EQ(o0.integralCast<unsigned short>(), v0);
        unsigned short v1 = 65126; Multiprecision o1 = v1; EXPECT_EQ(o1.integralCast<unsigned short>(), v1);
        unsigned short v2 = 31635; Multiprecision o2 = v2; EXPECT_EQ(o2.integralCast<unsigned short>(), v2);
        unsigned short v3 = 57499; Multiprecision o3 = v3; EXPECT_EQ(o3.integralCast<unsigned short>(), v3);
        unsigned short v4 = 38058; Multiprecision o4 = v4; EXPECT_EQ(o4.integralCast<unsigned short>(), v4);
        unsigned short v5 = 63039; Multiprecision o5 = v5; EXPECT_EQ(o5.integralCast<unsigned short>(), v5);
        unsigned short v6 = 8296; Multiprecision o6 = v6; EXPECT_EQ(o6.integralCast<unsigned short>(), v6);
        unsigned short v7 = 24236; Multiprecision o7 = v7; EXPECT_EQ(o7.integralCast<unsigned short>(), v7);
        unsigned short v8 = 29619; Multiprecision o8 = v8; EXPECT_EQ(o8.integralCast<unsigned short>(), v8);
        unsigned short v9 = 60555; Multiprecision o9 = v9; EXPECT_EQ(o9.integralCast<unsigned short>(), v9);
        unsigned short v10 = 38254; Multiprecision o10 = v10; EXPECT_EQ(o10.integralCast<unsigned short>(), v10);
        unsigned short v11 = 59479; Multiprecision o11 = v11; EXPECT_EQ(o11.integralCast<unsigned short>(), v11);
        unsigned short v12 = 63392; Multiprecision o12 = v12; EXPECT_EQ(o12.integralCast<unsigned short>(), v12);
        unsigned short v13 = 25152; Multiprecision o13 = v13; EXPECT_EQ(o13.integralCast<unsigned short>(), v13);
        unsigned short v14 = 3758; Multiprecision o14 = v14; EXPECT_EQ(o14.integralCast<unsigned short>(), v14);
        unsigned short v15 = 18321; Multiprecision o15 = v15; EXPECT_EQ(o15.integralCast<unsigned short>(), v15);
        unsigned short v16 = 39353; Multiprecision o16 = v16; EXPECT_EQ(o16.integralCast<unsigned short>(), v16);
        unsigned short v17 = 8739; Multiprecision o17 = v17; EXPECT_EQ(o17.integralCast<unsigned short>(), v17);
        unsigned short v18 = 42112; Multiprecision o18 = v18; EXPECT_EQ(o18.integralCast<unsigned short>(), v18);
        unsigned short v19 = 19620; Multiprecision o19 = v19; EXPECT_EQ(o19.integralCast<unsigned short>(), v19);
    }
    {
        int v0 = 1633072347; Multiprecision o0 = v0; EXPECT_EQ(o0.integralCast<int>(), v0);
        int v1 = 1373045468; Multiprecision o1 = v1; EXPECT_EQ(o1.integralCast<int>(), v1);
        int v2 = -168273879; Multiprecision o2 = v2; EXPECT_EQ(o2.integralCast<int>(), v2);
        int v3 = 1946447499; Multiprecision o3 = v3; EXPECT_EQ(o3.integralCast<int>(), v3);
        int v4 = -182071435; Multiprecision o4 = v4; EXPECT_EQ(o4.integralCast<int>(), v4);
        int v5 = 1000724497; Multiprecision o5 = v5; EXPECT_EQ(o5.integralCast<int>(), v5);
        int v6 = 690522175; Multiprecision o6 = v6; EXPECT_EQ(o6.integralCast<int>(), v6);
        int v7 = 2085869570; Multiprecision o7 = v7; EXPECT_EQ(o7.integralCast<int>(), v7);
        int v8 = 1710099476; Multiprecision o8 = v8; EXPECT_EQ(o8.integralCast<int>(), v8);
        int v9 = -1830783184; Multiprecision o9 = v9; EXPECT_EQ(o9.integralCast<int>(), v9);
        int v10 = 1171383165; Multiprecision o10 = v10; EXPECT_EQ(o10.integralCast<int>(), v10);
        int v11 = -1751249685; Multiprecision o11 = v11; EXPECT_EQ(o11.integralCast<int>(), v11);
        int v12 = -1668504170; Multiprecision o12 = v12; EXPECT_EQ(o12.integralCast<int>(), v12);
        int v13 = -241908811; Multiprecision o13 = v13; EXPECT_EQ(o13.integralCast<int>(), v13);
        int v14 = 540291974; Multiprecision o14 = v14; EXPECT_EQ(o14.integralCast<int>(), v14);
        int v15 = -1643457674; Multiprecision o15 = v15; EXPECT_EQ(o15.integralCast<int>(), v15);
        int v16 = 902106685; Multiprecision o16 = v16; EXPECT_EQ(o16.integralCast<int>(), v16);
        int v17 = -39871497; Multiprecision o17 = v17; EXPECT_EQ(o17.integralCast<int>(), v17);
        int v18 = 888304759; Multiprecision o18 = v18; EXPECT_EQ(o18.integralCast<int>(), v18);
        int v19 = 1568953909; Multiprecision o19 = v19; EXPECT_EQ(o19.integralCast<int>(), v19);
    }
    {
        unsigned v0 = 2005690205; Multiprecision o0 = v0; EXPECT_EQ(o0.integralCast<unsigned>(), v0);
        unsigned v1 = 1449627982; Multiprecision o1 = v1; EXPECT_EQ(o1.integralCast<unsigned>(), v1);
        unsigned v2 = 1101302552; Multiprecision o2 = v2; EXPECT_EQ(o2.integralCast<unsigned>(), v2);
        unsigned v3 = 595765511; Multiprecision o3 = v3; EXPECT_EQ(o3.integralCast<unsigned>(), v3);
        unsigned v4 = 3553780719; Multiprecision o4 = v4; EXPECT_EQ(o4.integralCast<unsigned>(), v4);
        unsigned v5 = 1594854713; Multiprecision o5 = v5; EXPECT_EQ(o5.integralCast<unsigned>(), v5);
        unsigned v6 = 1443312011; Multiprecision o6 = v6; EXPECT_EQ(o6.integralCast<unsigned>(), v6);
        unsigned v7 = 4050435786; Multiprecision o7 = v7; EXPECT_EQ(o7.integralCast<unsigned>(), v7);
        unsigned v8 = 2322528604; Multiprecision o8 = v8; EXPECT_EQ(o8.integralCast<unsigned>(), v8);
        unsigned v9 = 659532747; Multiprecision o9 = v9; EXPECT_EQ(o9.integralCast<unsigned>(), v9);
        unsigned v10 = 4162423578; Multiprecision o10 = v10; EXPECT_EQ(o10.integralCast<unsigned>(), v10);
        unsigned v11 = 2641111926; Multiprecision o11 = v11; EXPECT_EQ(o11.integralCast<unsigned>(), v11);
        unsigned v12 = 4253288915; Multiprecision o12 = v12; EXPECT_EQ(o12.integralCast<unsigned>(), v12);
        unsigned v13 = 3110878193; Multiprecision o13 = v13; EXPECT_EQ(o13.integralCast<unsigned>(), v13);
        unsigned v14 = 587696870; Multiprecision o14 = v14; EXPECT_EQ(o14.integralCast<unsigned>(), v14);
        unsigned v15 = 1153568748; Multiprecision o15 = v15; EXPECT_EQ(o15.integralCast<unsigned>(), v15);
        unsigned v16 = 4175266679; Multiprecision o16 = v16; EXPECT_EQ(o16.integralCast<unsigned>(), v16);
        unsigned v17 = 346810525; Multiprecision o17 = v17; EXPECT_EQ(o17.integralCast<unsigned>(), v17);
        unsigned v18 = 274355692; Multiprecision o18 = v18; EXPECT_EQ(o18.integralCast<unsigned>(), v18);
        unsigned v19 = 3087867327; Multiprecision o19 = v19; EXPECT_EQ(o19.integralCast<unsigned>(), v19);
    }
    {
        long v0 = 634812322182757621; Multiprecision o0 = v0; EXPECT_EQ(o0.integralCast<long>(), v0);
        long v1 = -927449906461017615; Multiprecision o1 = v1; EXPECT_EQ(o1.integralCast<long>(), v1);
        long v2 = -3504628980837960044; Multiprecision o2 = v2; EXPECT_EQ(o2.integralCast<long>(), v2);
        long v3 = -2155375560745272488; Multiprecision o3 = v3; EXPECT_EQ(o3.integralCast<long>(), v3);
        long v4 = -6526318726215051137; Multiprecision o4 = v4; EXPECT_EQ(o4.integralCast<long>(), v4);
        long v5 = -2694746162811842986; Multiprecision o5 = v5; EXPECT_EQ(o5.integralCast<long>(), v5);
        long v6 = 1631485759082327833; Multiprecision o6 = v6; EXPECT_EQ(o6.integralCast<long>(), v6);
        long v7 = 8407645366062381801; Multiprecision o7 = v7; EXPECT_EQ(o7.integralCast<long>(), v7);
        long v8 = 5841857305468410835; Multiprecision o8 = v8; EXPECT_EQ(o8.integralCast<long>(), v8);
        long v9 = 9165801435086311778; Multiprecision o9 = v9; EXPECT_EQ(o9.integralCast<long>(), v9);
        long v10 = 7049900736852272389; Multiprecision o10 = v10; EXPECT_EQ(o10.integralCast<long>(), v10);
        long v11 = -1116039134974475934; Multiprecision o11 = v11; EXPECT_EQ(o11.integralCast<long>(), v11);
        long v12 = 6134624124941989956; Multiprecision o12 = v12; EXPECT_EQ(o12.integralCast<long>(), v12);
        long v13 = -1989814207699167888; Multiprecision o13 = v13; EXPECT_EQ(o13.integralCast<long>(), v13);
        long v14 = -5162674662935837073; Multiprecision o14 = v14; EXPECT_EQ(o14.integralCast<long>(), v14);
        long v15 = 361978787381322157; Multiprecision o15 = v15; EXPECT_EQ(o15.integralCast<long>(), v15);
        long v16 = 2358318038699402381; Multiprecision o16 = v16; EXPECT_EQ(o16.integralCast<long>(), v16);
        long v17 = 6721455152406516439; Multiprecision o17 = v17; EXPECT_EQ(o17.integralCast<long>(), v17);
        long v18 = 6240728962025263647; Multiprecision o18 = v18; EXPECT_EQ(o18.integralCast<long>(), v18);
        long v19 = -7516849453343319431; Multiprecision o19 = v19; EXPECT_EQ(o19.integralCast<long>(), v19);
    }
    {
        unsigned long v0 = 1458432336447339549UL; Multiprecision o0 = v0; EXPECT_EQ(o0.integralCast<unsigned long>(), v0);
        unsigned long v1 = 15166986956279526495UL; Multiprecision o1 = v1; EXPECT_EQ(o1.integralCast<unsigned long>(), v1);
        unsigned long v2 = 843907744079905873UL; Multiprecision o2 = v2; EXPECT_EQ(o2.integralCast<unsigned long>(), v2);
        unsigned long v3 = 16831641047459684976UL; Multiprecision o3 = v3; EXPECT_EQ(o3.integralCast<unsigned long>(), v3);
        unsigned long v4 = 8233716564210995814UL; Multiprecision o4 = v4; EXPECT_EQ(o4.integralCast<unsigned long>(), v4);
        unsigned long v5 = 12056099167404879864UL; Multiprecision o5 = v5; EXPECT_EQ(o5.integralCast<unsigned long>(), v5);
        unsigned long v6 = 9195468763811385966UL; Multiprecision o6 = v6; EXPECT_EQ(o6.integralCast<unsigned long>(), v6);
        unsigned long v7 = 6838015201710322987UL; Multiprecision o7 = v7; EXPECT_EQ(o7.integralCast<unsigned long>(), v7);
        unsigned long v8 = 15909623765770348916UL; Multiprecision o8 = v8; EXPECT_EQ(o8.integralCast<unsigned long>(), v8);
        unsigned long v9 = 580359803260052445UL; Multiprecision o9 = v9; EXPECT_EQ(o9.integralCast<unsigned long>(), v9);
        unsigned long v10 = 1562144106091796071UL; Multiprecision o10 = v10; EXPECT_EQ(o10.integralCast<unsigned long>(), v10);
        unsigned long v11 = 9499213611466063262UL; Multiprecision o11 = v11; EXPECT_EQ(o11.integralCast<unsigned long>(), v11);
        unsigned long v12 = 12515821306029840702UL; Multiprecision o12 = v12; EXPECT_EQ(o12.integralCast<unsigned long>(), v12);
        unsigned long v13 = 5666922082602738570UL; Multiprecision o13 = v13; EXPECT_EQ(o13.integralCast<unsigned long>(), v13);
        unsigned long v14 = 12546046344820481203UL; Multiprecision o14 = v14; EXPECT_EQ(o14.integralCast<unsigned long>(), v14);
        unsigned long v15 = 13562221276840283927UL; Multiprecision o15 = v15; EXPECT_EQ(o15.integralCast<unsigned long>(), v15);
        unsigned long v16 = 12646097440568072921UL; Multiprecision o16 = v16; EXPECT_EQ(o16.integralCast<unsigned long>(), v16);
        unsigned long v17 = 14099814405939634110UL; Multiprecision o17 = v17; EXPECT_EQ(o17.integralCast<unsigned long>(), v17);
        unsigned long v18 = 13036786147570218084UL; Multiprecision o18 = v18; EXPECT_EQ(o18.integralCast<unsigned long>(), v18);
        unsigned long v19 = 14809679552796761321UL; Multiprecision o19 = v19; EXPECT_EQ(o19.integralCast<unsigned long>(), v19);
    }
    {
        long long v0 = -966320832989284844; Multiprecision o0 = v0; EXPECT_EQ(o0.integralCast<long long>(), v0);
        long long v1 = -2539299047194883024; Multiprecision o1 = v1; EXPECT_EQ(o1.integralCast<long long>(), v1);
        long long v2 = -4072657627411227702; Multiprecision o2 = v2; EXPECT_EQ(o2.integralCast<long long>(), v2);
        long long v3 = 6676151703625450453; Multiprecision o3 = v3; EXPECT_EQ(o3.integralCast<long long>(), v3);
        long long v4 = 6024657647873919769; Multiprecision o4 = v4; EXPECT_EQ(o4.integralCast<long long>(), v4);
        long long v5 = -3436387051813180897; Multiprecision o5 = v5; EXPECT_EQ(o5.integralCast<long long>(), v5);
        long long v6 = -930863918495184864; Multiprecision o6 = v6; EXPECT_EQ(o6.integralCast<long long>(), v6);
        long long v7 = -2648679724062892078; Multiprecision o7 = v7; EXPECT_EQ(o7.integralCast<long long>(), v7);
        long long v8 = 163799229998191542; Multiprecision o8 = v8; EXPECT_EQ(o8.integralCast<long long>(), v8);
        long long v9 = 7096985823516738807; Multiprecision o9 = v9; EXPECT_EQ(o9.integralCast<long long>(), v9);
        long long v10 = 9021468662796673265; Multiprecision o10 = v10; EXPECT_EQ(o10.integralCast<long long>(), v10);
        long long v11 = 7329596342632026139; Multiprecision o11 = v11; EXPECT_EQ(o11.integralCast<long long>(), v11);
        long long v12 = 2830097210272796867; Multiprecision o12 = v12; EXPECT_EQ(o12.integralCast<long long>(), v12);
        long long v13 = -8165373913969389069; Multiprecision o13 = v13; EXPECT_EQ(o13.integralCast<long long>(), v13);
        long long v14 = 1849790133061638302; Multiprecision o14 = v14; EXPECT_EQ(o14.integralCast<long long>(), v14);
        long long v15 = -8953474544892738048; Multiprecision o15 = v15; EXPECT_EQ(o15.integralCast<long long>(), v15);
        long long v16 = -4256943809212026391; Multiprecision o16 = v16; EXPECT_EQ(o16.integralCast<long long>(), v16);
        long long v17 = 5358224103492354631; Multiprecision o17 = v17; EXPECT_EQ(o17.integralCast<long long>(), v17);
        long long v18 = -1353061819987477708; Multiprecision o18 = v18; EXPECT_EQ(o18.integralCast<long long>(), v18);
        long long v19 = 5852270391142351733; Multiprecision o19 = v19; EXPECT_EQ(o19.integralCast<long long>(), v19);
    }
    {
        unsigned long long v0 = 7056752338713246902ULL; Multiprecision o0 = v0; EXPECT_EQ(o0.integralCast<unsigned long long>(), v0);
        unsigned long long v1 = 1752111911588599450ULL; Multiprecision o1 = v1; EXPECT_EQ(o1.integralCast<unsigned long long>(), v1);
        unsigned long long v2 = 9479044036425476019ULL; Multiprecision o2 = v2; EXPECT_EQ(o2.integralCast<unsigned long long>(), v2);
        unsigned long long v3 = 5193170295485823118ULL; Multiprecision o3 = v3; EXPECT_EQ(o3.integralCast<unsigned long long>(), v3);
        unsigned long long v4 = 7329594209829406618ULL; Multiprecision o4 = v4; EXPECT_EQ(o4.integralCast<unsigned long long>(), v4);
        unsigned long long v5 = 17408589177287406633ULL; Multiprecision o5 = v5; EXPECT_EQ(o5.integralCast<unsigned long long>(), v5);
        unsigned long long v6 = 10049699202423151920ULL; Multiprecision o6 = v6; EXPECT_EQ(o6.integralCast<unsigned long long>(), v6);
        unsigned long long v7 = 14893142280413871075ULL; Multiprecision o7 = v7; EXPECT_EQ(o7.integralCast<unsigned long long>(), v7);
        unsigned long long v8 = 14477514342547457937ULL; Multiprecision o8 = v8; EXPECT_EQ(o8.integralCast<unsigned long long>(), v8);
        unsigned long long v9 = 14158198760418317512ULL; Multiprecision o9 = v9; EXPECT_EQ(o9.integralCast<unsigned long long>(), v9);
        unsigned long long v10 = 1737450455060823428ULL; Multiprecision o10 = v10; EXPECT_EQ(o10.integralCast<unsigned long long>(), v10);
        unsigned long long v11 = 12956079072589122739ULL; Multiprecision o11 = v11; EXPECT_EQ(o11.integralCast<unsigned long long>(), v11);
        unsigned long long v12 = 15263083373573044880ULL; Multiprecision o12 = v12; EXPECT_EQ(o12.integralCast<unsigned long long>(), v12);
        unsigned long long v13 = 11589228342409016981ULL; Multiprecision o13 = v13; EXPECT_EQ(o13.integralCast<unsigned long long>(), v13);
        unsigned long long v14 = 12279481383607276825ULL; Multiprecision o14 = v14; EXPECT_EQ(o14.integralCast<unsigned long long>(), v14);
        unsigned long long v15 = 11338438110938113974ULL; Multiprecision o15 = v15; EXPECT_EQ(o15.integralCast<unsigned long long>(), v15);
        unsigned long long v16 = 14528990072287077479ULL; Multiprecision o16 = v16; EXPECT_EQ(o16.integralCast<unsigned long long>(), v16);
        unsigned long long v17 = 13754684638618159726ULL; Multiprecision o17 = v17; EXPECT_EQ(o17.integralCast<unsigned long long>(), v17);
        unsigned long long v18 = 6821012615472018674ULL; Multiprecision o18 = v18; EXPECT_EQ(o18.integralCast<unsigned long long>(), v18);
        unsigned long long v19 = 5576836686719978266ULL; Multiprecision o19 = v19; EXPECT_EQ(o19.integralCast<unsigned long long>(), v19);
    }
}

TEST(Casting, PrecisionCast) {
    long long init0 = 7524839891475014690; Multiprecision<256> m0 = init0;
    EXPECT_EQ(m0.precisionCast<288>(), init0); EXPECT_EQ(m0.precisionCast<320>(), init0); EXPECT_EQ(m0.precisionCast<352>(), init0); EXPECT_EQ(m0.precisionCast<384>(), init0);
    EXPECT_EQ(m0.precisionCast<416>(), init0); EXPECT_EQ(m0.precisionCast<448>(), init0); EXPECT_EQ(m0.precisionCast<480>(), init0); EXPECT_EQ(m0.precisionCast<512>(), init0);
    EXPECT_EQ(m0.precisionCast<544>(), init0); EXPECT_EQ(m0.precisionCast<576>(), init0); EXPECT_EQ(m0.precisionCast<608>(), init0); EXPECT_EQ(m0.precisionCast<640>(), init0);
    EXPECT_EQ(m0.precisionCast<672>(), init0); EXPECT_EQ(m0.precisionCast<704>(), init0); EXPECT_EQ(m0.precisionCast<736>(), init0); EXPECT_EQ(m0.precisionCast<768>(), init0);
    EXPECT_EQ(m0.precisionCast<800>(), init0); EXPECT_EQ(m0.precisionCast<832>(), init0); EXPECT_EQ(m0.precisionCast<864>(), init0); EXPECT_EQ(m0.precisionCast<896>(), init0);


    long long init1 = 8504238977266685499; Multiprecision<256> m1 = init1;
    EXPECT_EQ(m1.precisionCast<288>(), init1); EXPECT_EQ(m1.precisionCast<320>(), init1); EXPECT_EQ(m1.precisionCast<352>(), init1); EXPECT_EQ(m1.precisionCast<384>(), init1);
    EXPECT_EQ(m1.precisionCast<416>(), init1); EXPECT_EQ(m1.precisionCast<448>(), init1); EXPECT_EQ(m1.precisionCast<480>(), init1); EXPECT_EQ(m1.precisionCast<512>(), init1);
    EXPECT_EQ(m1.precisionCast<544>(), init1); EXPECT_EQ(m1.precisionCast<576>(), init1); EXPECT_EQ(m1.precisionCast<608>(), init1); EXPECT_EQ(m1.precisionCast<640>(), init1);
    EXPECT_EQ(m1.precisionCast<672>(), init1); EXPECT_EQ(m1.precisionCast<704>(), init1); EXPECT_EQ(m1.precisionCast<736>(), init1); EXPECT_EQ(m1.precisionCast<768>(), init1);
    EXPECT_EQ(m1.precisionCast<800>(), init1); EXPECT_EQ(m1.precisionCast<832>(), init1); EXPECT_EQ(m1.precisionCast<864>(), init1); EXPECT_EQ(m1.precisionCast<896>(), init1);


    long long init2 = -7698721489817646536; Multiprecision<256> m2 = init2;
    EXPECT_EQ(m2.precisionCast<288>(), init2); EXPECT_EQ(m2.precisionCast<320>(), init2); EXPECT_EQ(m2.precisionCast<352>(), init2); EXPECT_EQ(m2.precisionCast<384>(), init2);
    EXPECT_EQ(m2.precisionCast<416>(), init2); EXPECT_EQ(m2.precisionCast<448>(), init2); EXPECT_EQ(m2.precisionCast<480>(), init2); EXPECT_EQ(m2.precisionCast<512>(), init2);
    EXPECT_EQ(m2.precisionCast<544>(), init2); EXPECT_EQ(m2.precisionCast<576>(), init2); EXPECT_EQ(m2.precisionCast<608>(), init2); EXPECT_EQ(m2.precisionCast<640>(), init2);
    EXPECT_EQ(m2.precisionCast<672>(), init2); EXPECT_EQ(m2.precisionCast<704>(), init2); EXPECT_EQ(m2.precisionCast<736>(), init2); EXPECT_EQ(m2.precisionCast<768>(), init2);
    EXPECT_EQ(m2.precisionCast<800>(), init2); EXPECT_EQ(m2.precisionCast<832>(), init2); EXPECT_EQ(m2.precisionCast<864>(), init2); EXPECT_EQ(m2.precisionCast<896>(), init2);


    long long init3 = -6306961477277496307; Multiprecision<256> m3 = init3;
    EXPECT_EQ(m3.precisionCast<288>(), init3); EXPECT_EQ(m3.precisionCast<320>(), init3); EXPECT_EQ(m3.precisionCast<352>(), init3); EXPECT_EQ(m3.precisionCast<384>(), init3);
    EXPECT_EQ(m3.precisionCast<416>(), init3); EXPECT_EQ(m3.precisionCast<448>(), init3); EXPECT_EQ(m3.precisionCast<480>(), init3); EXPECT_EQ(m3.precisionCast<512>(), init3);
    EXPECT_EQ(m3.precisionCast<544>(), init3); EXPECT_EQ(m3.precisionCast<576>(), init3); EXPECT_EQ(m3.precisionCast<608>(), init3); EXPECT_EQ(m3.precisionCast<640>(), init3);
    EXPECT_EQ(m3.precisionCast<672>(), init3); EXPECT_EQ(m3.precisionCast<704>(), init3); EXPECT_EQ(m3.precisionCast<736>(), init3); EXPECT_EQ(m3.precisionCast<768>(), init3);
    EXPECT_EQ(m3.precisionCast<800>(), init3); EXPECT_EQ(m3.precisionCast<832>(), init3); EXPECT_EQ(m3.precisionCast<864>(), init3); EXPECT_EQ(m3.precisionCast<896>(), init3);


    long long init4 = -1439223510727390957; Multiprecision<256> m4 = init4;
    EXPECT_EQ(m4.precisionCast<288>(), init4); EXPECT_EQ(m4.precisionCast<320>(), init4); EXPECT_EQ(m4.precisionCast<352>(), init4); EXPECT_EQ(m4.precisionCast<384>(), init4);
    EXPECT_EQ(m4.precisionCast<416>(), init4); EXPECT_EQ(m4.precisionCast<448>(), init4); EXPECT_EQ(m4.precisionCast<480>(), init4); EXPECT_EQ(m4.precisionCast<512>(), init4);
    EXPECT_EQ(m4.precisionCast<544>(), init4); EXPECT_EQ(m4.precisionCast<576>(), init4); EXPECT_EQ(m4.precisionCast<608>(), init4); EXPECT_EQ(m4.precisionCast<640>(), init4);
    EXPECT_EQ(m4.precisionCast<672>(), init4); EXPECT_EQ(m4.precisionCast<704>(), init4); EXPECT_EQ(m4.precisionCast<736>(), init4); EXPECT_EQ(m4.precisionCast<768>(), init4);
    EXPECT_EQ(m4.precisionCast<800>(), init4); EXPECT_EQ(m4.precisionCast<832>(), init4); EXPECT_EQ(m4.precisionCast<864>(), init4); EXPECT_EQ(m4.precisionCast<896>(), init4);


    long long init5 = -9108790223466577088; Multiprecision<256> m5 = init5;
    EXPECT_EQ(m5.precisionCast<288>(), init5); EXPECT_EQ(m5.precisionCast<320>(), init5); EXPECT_EQ(m5.precisionCast<352>(), init5); EXPECT_EQ(m5.precisionCast<384>(), init5);
    EXPECT_EQ(m5.precisionCast<416>(), init5); EXPECT_EQ(m5.precisionCast<448>(), init5); EXPECT_EQ(m5.precisionCast<480>(), init5); EXPECT_EQ(m5.precisionCast<512>(), init5);
    EXPECT_EQ(m5.precisionCast<544>(), init5); EXPECT_EQ(m5.precisionCast<576>(), init5); EXPECT_EQ(m5.precisionCast<608>(), init5); EXPECT_EQ(m5.precisionCast<640>(), init5);
    EXPECT_EQ(m5.precisionCast<672>(), init5); EXPECT_EQ(m5.precisionCast<704>(), init5); EXPECT_EQ(m5.precisionCast<736>(), init5); EXPECT_EQ(m5.precisionCast<768>(), init5);
    EXPECT_EQ(m5.precisionCast<800>(), init5); EXPECT_EQ(m5.precisionCast<832>(), init5); EXPECT_EQ(m5.precisionCast<864>(), init5); EXPECT_EQ(m5.precisionCast<896>(), init5);


    long long init6 = 7336368703164388051; Multiprecision<256> m6 = init6;
    EXPECT_EQ(m6.precisionCast<288>(), init6); EXPECT_EQ(m6.precisionCast<320>(), init6); EXPECT_EQ(m6.precisionCast<352>(), init6); EXPECT_EQ(m6.precisionCast<384>(), init6);
    EXPECT_EQ(m6.precisionCast<416>(), init6); EXPECT_EQ(m6.precisionCast<448>(), init6); EXPECT_EQ(m6.precisionCast<480>(), init6); EXPECT_EQ(m6.precisionCast<512>(), init6);
    EXPECT_EQ(m6.precisionCast<544>(), init6); EXPECT_EQ(m6.precisionCast<576>(), init6); EXPECT_EQ(m6.precisionCast<608>(), init6); EXPECT_EQ(m6.precisionCast<640>(), init6);
    EXPECT_EQ(m6.precisionCast<672>(), init6); EXPECT_EQ(m6.precisionCast<704>(), init6); EXPECT_EQ(m6.precisionCast<736>(), init6); EXPECT_EQ(m6.precisionCast<768>(), init6);
    EXPECT_EQ(m6.precisionCast<800>(), init6); EXPECT_EQ(m6.precisionCast<832>(), init6); EXPECT_EQ(m6.precisionCast<864>(), init6); EXPECT_EQ(m6.precisionCast<896>(), init6);


    long long init7 = 2314166570054255036; Multiprecision<256> m7 = init7;
    EXPECT_EQ(m7.precisionCast<288>(), init7); EXPECT_EQ(m7.precisionCast<320>(), init7); EXPECT_EQ(m7.precisionCast<352>(), init7); EXPECT_EQ(m7.precisionCast<384>(), init7);
    EXPECT_EQ(m7.precisionCast<416>(), init7); EXPECT_EQ(m7.precisionCast<448>(), init7); EXPECT_EQ(m7.precisionCast<480>(), init7); EXPECT_EQ(m7.precisionCast<512>(), init7);
    EXPECT_EQ(m7.precisionCast<544>(), init7); EXPECT_EQ(m7.precisionCast<576>(), init7); EXPECT_EQ(m7.precisionCast<608>(), init7); EXPECT_EQ(m7.precisionCast<640>(), init7);
    EXPECT_EQ(m7.precisionCast<672>(), init7); EXPECT_EQ(m7.precisionCast<704>(), init7); EXPECT_EQ(m7.precisionCast<736>(), init7); EXPECT_EQ(m7.precisionCast<768>(), init7);
    EXPECT_EQ(m7.precisionCast<800>(), init7); EXPECT_EQ(m7.precisionCast<832>(), init7); EXPECT_EQ(m7.precisionCast<864>(), init7); EXPECT_EQ(m7.precisionCast<896>(), init7);


    long long init8 = 7305419937287767522; Multiprecision<256> m8 = init8;
    EXPECT_EQ(m8.precisionCast<288>(), init8); EXPECT_EQ(m8.precisionCast<320>(), init8); EXPECT_EQ(m8.precisionCast<352>(), init8); EXPECT_EQ(m8.precisionCast<384>(), init8);
    EXPECT_EQ(m8.precisionCast<416>(), init8); EXPECT_EQ(m8.precisionCast<448>(), init8); EXPECT_EQ(m8.precisionCast<480>(), init8); EXPECT_EQ(m8.precisionCast<512>(), init8);
    EXPECT_EQ(m8.precisionCast<544>(), init8); EXPECT_EQ(m8.precisionCast<576>(), init8); EXPECT_EQ(m8.precisionCast<608>(), init8); EXPECT_EQ(m8.precisionCast<640>(), init8);
    EXPECT_EQ(m8.precisionCast<672>(), init8); EXPECT_EQ(m8.precisionCast<704>(), init8); EXPECT_EQ(m8.precisionCast<736>(), init8); EXPECT_EQ(m8.precisionCast<768>(), init8);
    EXPECT_EQ(m8.precisionCast<800>(), init8); EXPECT_EQ(m8.precisionCast<832>(), init8); EXPECT_EQ(m8.precisionCast<864>(), init8); EXPECT_EQ(m8.precisionCast<896>(), init8);


    long long init9 = -2561303725389650172; Multiprecision<256> m9 = init9;
    EXPECT_EQ(m9.precisionCast<288>(), init9); EXPECT_EQ(m9.precisionCast<320>(), init9); EXPECT_EQ(m9.precisionCast<352>(), init9); EXPECT_EQ(m9.precisionCast<384>(), init9);
    EXPECT_EQ(m9.precisionCast<416>(), init9); EXPECT_EQ(m9.precisionCast<448>(), init9); EXPECT_EQ(m9.precisionCast<480>(), init9); EXPECT_EQ(m9.precisionCast<512>(), init9);
    EXPECT_EQ(m9.precisionCast<544>(), init9); EXPECT_EQ(m9.precisionCast<576>(), init9); EXPECT_EQ(m9.precisionCast<608>(), init9); EXPECT_EQ(m9.precisionCast<640>(), init9);
    EXPECT_EQ(m9.precisionCast<672>(), init9); EXPECT_EQ(m9.precisionCast<704>(), init9); EXPECT_EQ(m9.precisionCast<736>(), init9); EXPECT_EQ(m9.precisionCast<768>(), init9);
    EXPECT_EQ(m9.precisionCast<800>(), init9); EXPECT_EQ(m9.precisionCast<832>(), init9); EXPECT_EQ(m9.precisionCast<864>(), init9); EXPECT_EQ(m9.precisionCast<896>(), init9);


    long long init10 = 3383802386880547846; Multiprecision<256> m10 = init10;
    EXPECT_EQ(m10.precisionCast<288>(), init10); EXPECT_EQ(m10.precisionCast<320>(), init10); EXPECT_EQ(m10.precisionCast<352>(), init10); EXPECT_EQ(m10.precisionCast<384>(), init10);
    EXPECT_EQ(m10.precisionCast<416>(), init10); EXPECT_EQ(m10.precisionCast<448>(), init10); EXPECT_EQ(m10.precisionCast<480>(), init10); EXPECT_EQ(m10.precisionCast<512>(), init10);
    EXPECT_EQ(m10.precisionCast<544>(), init10); EXPECT_EQ(m10.precisionCast<576>(), init10); EXPECT_EQ(m10.precisionCast<608>(), init10); EXPECT_EQ(m10.precisionCast<640>(), init10);
    EXPECT_EQ(m10.precisionCast<672>(), init10); EXPECT_EQ(m10.precisionCast<704>(), init10); EXPECT_EQ(m10.precisionCast<736>(), init10); EXPECT_EQ(m10.precisionCast<768>(), init10);
    EXPECT_EQ(m10.precisionCast<800>(), init10); EXPECT_EQ(m10.precisionCast<832>(), init10); EXPECT_EQ(m10.precisionCast<864>(), init10); EXPECT_EQ(m10.precisionCast<896>(), init10);


    long long init11 = 5703252986956204740; Multiprecision<256> m11 = init11;
    EXPECT_EQ(m11.precisionCast<288>(), init11); EXPECT_EQ(m11.precisionCast<320>(), init11); EXPECT_EQ(m11.precisionCast<352>(), init11); EXPECT_EQ(m11.precisionCast<384>(), init11);
    EXPECT_EQ(m11.precisionCast<416>(), init11); EXPECT_EQ(m11.precisionCast<448>(), init11); EXPECT_EQ(m11.precisionCast<480>(), init11); EXPECT_EQ(m11.precisionCast<512>(), init11);
    EXPECT_EQ(m11.precisionCast<544>(), init11); EXPECT_EQ(m11.precisionCast<576>(), init11); EXPECT_EQ(m11.precisionCast<608>(), init11); EXPECT_EQ(m11.precisionCast<640>(), init11);
    EXPECT_EQ(m11.precisionCast<672>(), init11); EXPECT_EQ(m11.precisionCast<704>(), init11); EXPECT_EQ(m11.precisionCast<736>(), init11); EXPECT_EQ(m11.precisionCast<768>(), init11);
    EXPECT_EQ(m11.precisionCast<800>(), init11); EXPECT_EQ(m11.precisionCast<832>(), init11); EXPECT_EQ(m11.precisionCast<864>(), init11); EXPECT_EQ(m11.precisionCast<896>(), init11);


    long long init12 = 1248358330450460190; Multiprecision<256> m12 = init12;
    EXPECT_EQ(m12.precisionCast<288>(), init12); EXPECT_EQ(m12.precisionCast<320>(), init12); EXPECT_EQ(m12.precisionCast<352>(), init12); EXPECT_EQ(m12.precisionCast<384>(), init12);
    EXPECT_EQ(m12.precisionCast<416>(), init12); EXPECT_EQ(m12.precisionCast<448>(), init12); EXPECT_EQ(m12.precisionCast<480>(), init12); EXPECT_EQ(m12.precisionCast<512>(), init12);
    EXPECT_EQ(m12.precisionCast<544>(), init12); EXPECT_EQ(m12.precisionCast<576>(), init12); EXPECT_EQ(m12.precisionCast<608>(), init12); EXPECT_EQ(m12.precisionCast<640>(), init12);
    EXPECT_EQ(m12.precisionCast<672>(), init12); EXPECT_EQ(m12.precisionCast<704>(), init12); EXPECT_EQ(m12.precisionCast<736>(), init12); EXPECT_EQ(m12.precisionCast<768>(), init12);
    EXPECT_EQ(m12.precisionCast<800>(), init12); EXPECT_EQ(m12.precisionCast<832>(), init12); EXPECT_EQ(m12.precisionCast<864>(), init12); EXPECT_EQ(m12.precisionCast<896>(), init12);


    long long init13 = 4175697953158773741; Multiprecision<256> m13 = init13;
    EXPECT_EQ(m13.precisionCast<288>(), init13); EXPECT_EQ(m13.precisionCast<320>(), init13); EXPECT_EQ(m13.precisionCast<352>(), init13); EXPECT_EQ(m13.precisionCast<384>(), init13);
    EXPECT_EQ(m13.precisionCast<416>(), init13); EXPECT_EQ(m13.precisionCast<448>(), init13); EXPECT_EQ(m13.precisionCast<480>(), init13); EXPECT_EQ(m13.precisionCast<512>(), init13);
    EXPECT_EQ(m13.precisionCast<544>(), init13); EXPECT_EQ(m13.precisionCast<576>(), init13); EXPECT_EQ(m13.precisionCast<608>(), init13); EXPECT_EQ(m13.precisionCast<640>(), init13);
    EXPECT_EQ(m13.precisionCast<672>(), init13); EXPECT_EQ(m13.precisionCast<704>(), init13); EXPECT_EQ(m13.precisionCast<736>(), init13); EXPECT_EQ(m13.precisionCast<768>(), init13);
    EXPECT_EQ(m13.precisionCast<800>(), init13); EXPECT_EQ(m13.precisionCast<832>(), init13); EXPECT_EQ(m13.precisionCast<864>(), init13); EXPECT_EQ(m13.precisionCast<896>(), init13);


    long long init14 = -2938517079411100489; Multiprecision<256> m14 = init14;
    EXPECT_EQ(m14.precisionCast<288>(), init14); EXPECT_EQ(m14.precisionCast<320>(), init14); EXPECT_EQ(m14.precisionCast<352>(), init14); EXPECT_EQ(m14.precisionCast<384>(), init14);
    EXPECT_EQ(m14.precisionCast<416>(), init14); EXPECT_EQ(m14.precisionCast<448>(), init14); EXPECT_EQ(m14.precisionCast<480>(), init14); EXPECT_EQ(m14.precisionCast<512>(), init14);
    EXPECT_EQ(m14.precisionCast<544>(), init14); EXPECT_EQ(m14.precisionCast<576>(), init14); EXPECT_EQ(m14.precisionCast<608>(), init14); EXPECT_EQ(m14.precisionCast<640>(), init14);
    EXPECT_EQ(m14.precisionCast<672>(), init14); EXPECT_EQ(m14.precisionCast<704>(), init14); EXPECT_EQ(m14.precisionCast<736>(), init14); EXPECT_EQ(m14.precisionCast<768>(), init14);
    EXPECT_EQ(m14.precisionCast<800>(), init14); EXPECT_EQ(m14.precisionCast<832>(), init14); EXPECT_EQ(m14.precisionCast<864>(), init14); EXPECT_EQ(m14.precisionCast<896>(), init14);


    long long init15 = -6902692574326511305; Multiprecision<256> m15 = init15;
    EXPECT_EQ(m15.precisionCast<288>(), init15); EXPECT_EQ(m15.precisionCast<320>(), init15); EXPECT_EQ(m15.precisionCast<352>(), init15); EXPECT_EQ(m15.precisionCast<384>(), init15);
    EXPECT_EQ(m15.precisionCast<416>(), init15); EXPECT_EQ(m15.precisionCast<448>(), init15); EXPECT_EQ(m15.precisionCast<480>(), init15); EXPECT_EQ(m15.precisionCast<512>(), init15);
    EXPECT_EQ(m15.precisionCast<544>(), init15); EXPECT_EQ(m15.precisionCast<576>(), init15); EXPECT_EQ(m15.precisionCast<608>(), init15); EXPECT_EQ(m15.precisionCast<640>(), init15);
    EXPECT_EQ(m15.precisionCast<672>(), init15); EXPECT_EQ(m15.precisionCast<704>(), init15); EXPECT_EQ(m15.precisionCast<736>(), init15); EXPECT_EQ(m15.precisionCast<768>(), init15);
    EXPECT_EQ(m15.precisionCast<800>(), init15); EXPECT_EQ(m15.precisionCast<832>(), init15); EXPECT_EQ(m15.precisionCast<864>(), init15); EXPECT_EQ(m15.precisionCast<896>(), init15);


    long long init16 = -2409078837645875887; Multiprecision<256> m16 = init16;
    EXPECT_EQ(m16.precisionCast<288>(), init16); EXPECT_EQ(m16.precisionCast<320>(), init16); EXPECT_EQ(m16.precisionCast<352>(), init16); EXPECT_EQ(m16.precisionCast<384>(), init16);
    EXPECT_EQ(m16.precisionCast<416>(), init16); EXPECT_EQ(m16.precisionCast<448>(), init16); EXPECT_EQ(m16.precisionCast<480>(), init16); EXPECT_EQ(m16.precisionCast<512>(), init16);
    EXPECT_EQ(m16.precisionCast<544>(), init16); EXPECT_EQ(m16.precisionCast<576>(), init16); EXPECT_EQ(m16.precisionCast<608>(), init16); EXPECT_EQ(m16.precisionCast<640>(), init16);
    EXPECT_EQ(m16.precisionCast<672>(), init16); EXPECT_EQ(m16.precisionCast<704>(), init16); EXPECT_EQ(m16.precisionCast<736>(), init16); EXPECT_EQ(m16.precisionCast<768>(), init16);
    EXPECT_EQ(m16.precisionCast<800>(), init16); EXPECT_EQ(m16.precisionCast<832>(), init16); EXPECT_EQ(m16.precisionCast<864>(), init16); EXPECT_EQ(m16.precisionCast<896>(), init16);


    long long init17 = -7506774356715976030; Multiprecision<256> m17 = init17;
    EXPECT_EQ(m17.precisionCast<288>(), init17); EXPECT_EQ(m17.precisionCast<320>(), init17); EXPECT_EQ(m17.precisionCast<352>(), init17); EXPECT_EQ(m17.precisionCast<384>(), init17);
    EXPECT_EQ(m17.precisionCast<416>(), init17); EXPECT_EQ(m17.precisionCast<448>(), init17); EXPECT_EQ(m17.precisionCast<480>(), init17); EXPECT_EQ(m17.precisionCast<512>(), init17);
    EXPECT_EQ(m17.precisionCast<544>(), init17); EXPECT_EQ(m17.precisionCast<576>(), init17); EXPECT_EQ(m17.precisionCast<608>(), init17); EXPECT_EQ(m17.precisionCast<640>(), init17);
    EXPECT_EQ(m17.precisionCast<672>(), init17); EXPECT_EQ(m17.precisionCast<704>(), init17); EXPECT_EQ(m17.precisionCast<736>(), init17); EXPECT_EQ(m17.precisionCast<768>(), init17);
    EXPECT_EQ(m17.precisionCast<800>(), init17); EXPECT_EQ(m17.precisionCast<832>(), init17); EXPECT_EQ(m17.precisionCast<864>(), init17); EXPECT_EQ(m17.precisionCast<896>(), init17);


    long long init18 = 227163478686314711; Multiprecision<256> m18 = init18;
    EXPECT_EQ(m18.precisionCast<288>(), init18); EXPECT_EQ(m18.precisionCast<320>(), init18); EXPECT_EQ(m18.precisionCast<352>(), init18); EXPECT_EQ(m18.precisionCast<384>(), init18);
    EXPECT_EQ(m18.precisionCast<416>(), init18); EXPECT_EQ(m18.precisionCast<448>(), init18); EXPECT_EQ(m18.precisionCast<480>(), init18); EXPECT_EQ(m18.precisionCast<512>(), init18);
    EXPECT_EQ(m18.precisionCast<544>(), init18); EXPECT_EQ(m18.precisionCast<576>(), init18); EXPECT_EQ(m18.precisionCast<608>(), init18); EXPECT_EQ(m18.precisionCast<640>(), init18);
    EXPECT_EQ(m18.precisionCast<672>(), init18); EXPECT_EQ(m18.precisionCast<704>(), init18); EXPECT_EQ(m18.precisionCast<736>(), init18); EXPECT_EQ(m18.precisionCast<768>(), init18);
    EXPECT_EQ(m18.precisionCast<800>(), init18); EXPECT_EQ(m18.precisionCast<832>(), init18); EXPECT_EQ(m18.precisionCast<864>(), init18); EXPECT_EQ(m18.precisionCast<896>(), init18);


    long long init19 = -4966632676207774604; Multiprecision<256> m19 = init19;
    EXPECT_EQ(m19.precisionCast<288>(), init19); EXPECT_EQ(m19.precisionCast<320>(), init19); EXPECT_EQ(m19.precisionCast<352>(), init19); EXPECT_EQ(m19.precisionCast<384>(), init19);
    EXPECT_EQ(m19.precisionCast<416>(), init19); EXPECT_EQ(m19.precisionCast<448>(), init19); EXPECT_EQ(m19.precisionCast<480>(), init19); EXPECT_EQ(m19.precisionCast<512>(), init19);
    EXPECT_EQ(m19.precisionCast<544>(), init19); EXPECT_EQ(m19.precisionCast<576>(), init19); EXPECT_EQ(m19.precisionCast<608>(), init19); EXPECT_EQ(m19.precisionCast<640>(), init19);
    EXPECT_EQ(m19.precisionCast<672>(), init19); EXPECT_EQ(m19.precisionCast<704>(), init19); EXPECT_EQ(m19.precisionCast<736>(), init19); EXPECT_EQ(m19.precisionCast<768>(), init19);
    EXPECT_EQ(m19.precisionCast<800>(), init19); EXPECT_EQ(m19.precisionCast<832>(), init19); EXPECT_EQ(m19.precisionCast<864>(), init19); EXPECT_EQ(m19.precisionCast<896>(), init19);


    long long init20 = -4021786551316732423; Multiprecision<256> m20 = init20;
    EXPECT_EQ(m20.precisionCast<288>(), init20); EXPECT_EQ(m20.precisionCast<320>(), init20); EXPECT_EQ(m20.precisionCast<352>(), init20); EXPECT_EQ(m20.precisionCast<384>(), init20);
    EXPECT_EQ(m20.precisionCast<416>(), init20); EXPECT_EQ(m20.precisionCast<448>(), init20); EXPECT_EQ(m20.precisionCast<480>(), init20); EXPECT_EQ(m20.precisionCast<512>(), init20);
    EXPECT_EQ(m20.precisionCast<544>(), init20); EXPECT_EQ(m20.precisionCast<576>(), init20); EXPECT_EQ(m20.precisionCast<608>(), init20); EXPECT_EQ(m20.precisionCast<640>(), init20);
    EXPECT_EQ(m20.precisionCast<672>(), init20); EXPECT_EQ(m20.precisionCast<704>(), init20); EXPECT_EQ(m20.precisionCast<736>(), init20); EXPECT_EQ(m20.precisionCast<768>(), init20);
    EXPECT_EQ(m20.precisionCast<800>(), init20); EXPECT_EQ(m20.precisionCast<832>(), init20); EXPECT_EQ(m20.precisionCast<864>(), init20); EXPECT_EQ(m20.precisionCast<896>(), init20);


    long long init21 = 7309039963756908723; Multiprecision<256> m21 = init21;
    EXPECT_EQ(m21.precisionCast<288>(), init21); EXPECT_EQ(m21.precisionCast<320>(), init21); EXPECT_EQ(m21.precisionCast<352>(), init21); EXPECT_EQ(m21.precisionCast<384>(), init21);
    EXPECT_EQ(m21.precisionCast<416>(), init21); EXPECT_EQ(m21.precisionCast<448>(), init21); EXPECT_EQ(m21.precisionCast<480>(), init21); EXPECT_EQ(m21.precisionCast<512>(), init21);
    EXPECT_EQ(m21.precisionCast<544>(), init21); EXPECT_EQ(m21.precisionCast<576>(), init21); EXPECT_EQ(m21.precisionCast<608>(), init21); EXPECT_EQ(m21.precisionCast<640>(), init21);
    EXPECT_EQ(m21.precisionCast<672>(), init21); EXPECT_EQ(m21.precisionCast<704>(), init21); EXPECT_EQ(m21.precisionCast<736>(), init21); EXPECT_EQ(m21.precisionCast<768>(), init21);
    EXPECT_EQ(m21.precisionCast<800>(), init21); EXPECT_EQ(m21.precisionCast<832>(), init21); EXPECT_EQ(m21.precisionCast<864>(), init21); EXPECT_EQ(m21.precisionCast<896>(), init21);


    long long init22 = -4037333522000484564; Multiprecision<256> m22 = init22;
    EXPECT_EQ(m22.precisionCast<288>(), init22); EXPECT_EQ(m22.precisionCast<320>(), init22); EXPECT_EQ(m22.precisionCast<352>(), init22); EXPECT_EQ(m22.precisionCast<384>(), init22);
    EXPECT_EQ(m22.precisionCast<416>(), init22); EXPECT_EQ(m22.precisionCast<448>(), init22); EXPECT_EQ(m22.precisionCast<480>(), init22); EXPECT_EQ(m22.precisionCast<512>(), init22);
    EXPECT_EQ(m22.precisionCast<544>(), init22); EXPECT_EQ(m22.precisionCast<576>(), init22); EXPECT_EQ(m22.precisionCast<608>(), init22); EXPECT_EQ(m22.precisionCast<640>(), init22);
    EXPECT_EQ(m22.precisionCast<672>(), init22); EXPECT_EQ(m22.precisionCast<704>(), init22); EXPECT_EQ(m22.precisionCast<736>(), init22); EXPECT_EQ(m22.precisionCast<768>(), init22);
    EXPECT_EQ(m22.precisionCast<800>(), init22); EXPECT_EQ(m22.precisionCast<832>(), init22); EXPECT_EQ(m22.precisionCast<864>(), init22); EXPECT_EQ(m22.precisionCast<896>(), init22);


    long long init23 = 3067962240659289255; Multiprecision<256> m23 = init23;
    EXPECT_EQ(m23.precisionCast<288>(), init23); EXPECT_EQ(m23.precisionCast<320>(), init23); EXPECT_EQ(m23.precisionCast<352>(), init23); EXPECT_EQ(m23.precisionCast<384>(), init23);
    EXPECT_EQ(m23.precisionCast<416>(), init23); EXPECT_EQ(m23.precisionCast<448>(), init23); EXPECT_EQ(m23.precisionCast<480>(), init23); EXPECT_EQ(m23.precisionCast<512>(), init23);
    EXPECT_EQ(m23.precisionCast<544>(), init23); EXPECT_EQ(m23.precisionCast<576>(), init23); EXPECT_EQ(m23.precisionCast<608>(), init23); EXPECT_EQ(m23.precisionCast<640>(), init23);
    EXPECT_EQ(m23.precisionCast<672>(), init23); EXPECT_EQ(m23.precisionCast<704>(), init23); EXPECT_EQ(m23.precisionCast<736>(), init23); EXPECT_EQ(m23.precisionCast<768>(), init23);
    EXPECT_EQ(m23.precisionCast<800>(), init23); EXPECT_EQ(m23.precisionCast<832>(), init23); EXPECT_EQ(m23.precisionCast<864>(), init23); EXPECT_EQ(m23.precisionCast<896>(), init23);


    long long init24 = -3707800888550614839; Multiprecision<256> m24 = init24;
    EXPECT_EQ(m24.precisionCast<288>(), init24); EXPECT_EQ(m24.precisionCast<320>(), init24); EXPECT_EQ(m24.precisionCast<352>(), init24); EXPECT_EQ(m24.precisionCast<384>(), init24);
    EXPECT_EQ(m24.precisionCast<416>(), init24); EXPECT_EQ(m24.precisionCast<448>(), init24); EXPECT_EQ(m24.precisionCast<480>(), init24); EXPECT_EQ(m24.precisionCast<512>(), init24);
    EXPECT_EQ(m24.precisionCast<544>(), init24); EXPECT_EQ(m24.precisionCast<576>(), init24); EXPECT_EQ(m24.precisionCast<608>(), init24); EXPECT_EQ(m24.precisionCast<640>(), init24);
    EXPECT_EQ(m24.precisionCast<672>(), init24); EXPECT_EQ(m24.precisionCast<704>(), init24); EXPECT_EQ(m24.precisionCast<736>(), init24); EXPECT_EQ(m24.precisionCast<768>(), init24);
    EXPECT_EQ(m24.precisionCast<800>(), init24); EXPECT_EQ(m24.precisionCast<832>(), init24); EXPECT_EQ(m24.precisionCast<864>(), init24); EXPECT_EQ(m24.precisionCast<896>(), init24);


    long long init25 = -6878944643378492415; Multiprecision<256> m25 = init25;
    EXPECT_EQ(m25.precisionCast<288>(), init25); EXPECT_EQ(m25.precisionCast<320>(), init25); EXPECT_EQ(m25.precisionCast<352>(), init25); EXPECT_EQ(m25.precisionCast<384>(), init25);
    EXPECT_EQ(m25.precisionCast<416>(), init25); EXPECT_EQ(m25.precisionCast<448>(), init25); EXPECT_EQ(m25.precisionCast<480>(), init25); EXPECT_EQ(m25.precisionCast<512>(), init25);
    EXPECT_EQ(m25.precisionCast<544>(), init25); EXPECT_EQ(m25.precisionCast<576>(), init25); EXPECT_EQ(m25.precisionCast<608>(), init25); EXPECT_EQ(m25.precisionCast<640>(), init25);
    EXPECT_EQ(m25.precisionCast<672>(), init25); EXPECT_EQ(m25.precisionCast<704>(), init25); EXPECT_EQ(m25.precisionCast<736>(), init25); EXPECT_EQ(m25.precisionCast<768>(), init25);
    EXPECT_EQ(m25.precisionCast<800>(), init25); EXPECT_EQ(m25.precisionCast<832>(), init25); EXPECT_EQ(m25.precisionCast<864>(), init25); EXPECT_EQ(m25.precisionCast<896>(), init25);


    long long init26 = -590213465179246446; Multiprecision<256> m26 = init26;
    EXPECT_EQ(m26.precisionCast<288>(), init26); EXPECT_EQ(m26.precisionCast<320>(), init26); EXPECT_EQ(m26.precisionCast<352>(), init26); EXPECT_EQ(m26.precisionCast<384>(), init26);
    EXPECT_EQ(m26.precisionCast<416>(), init26); EXPECT_EQ(m26.precisionCast<448>(), init26); EXPECT_EQ(m26.precisionCast<480>(), init26); EXPECT_EQ(m26.precisionCast<512>(), init26);
    EXPECT_EQ(m26.precisionCast<544>(), init26); EXPECT_EQ(m26.precisionCast<576>(), init26); EXPECT_EQ(m26.precisionCast<608>(), init26); EXPECT_EQ(m26.precisionCast<640>(), init26);
    EXPECT_EQ(m26.precisionCast<672>(), init26); EXPECT_EQ(m26.precisionCast<704>(), init26); EXPECT_EQ(m26.precisionCast<736>(), init26); EXPECT_EQ(m26.precisionCast<768>(), init26);
    EXPECT_EQ(m26.precisionCast<800>(), init26); EXPECT_EQ(m26.precisionCast<832>(), init26); EXPECT_EQ(m26.precisionCast<864>(), init26); EXPECT_EQ(m26.precisionCast<896>(), init26);


    long long init27 = 6023706187360170892; Multiprecision<256> m27 = init27;
    EXPECT_EQ(m27.precisionCast<288>(), init27); EXPECT_EQ(m27.precisionCast<320>(), init27); EXPECT_EQ(m27.precisionCast<352>(), init27); EXPECT_EQ(m27.precisionCast<384>(), init27);
    EXPECT_EQ(m27.precisionCast<416>(), init27); EXPECT_EQ(m27.precisionCast<448>(), init27); EXPECT_EQ(m27.precisionCast<480>(), init27); EXPECT_EQ(m27.precisionCast<512>(), init27);
    EXPECT_EQ(m27.precisionCast<544>(), init27); EXPECT_EQ(m27.precisionCast<576>(), init27); EXPECT_EQ(m27.precisionCast<608>(), init27); EXPECT_EQ(m27.precisionCast<640>(), init27);
    EXPECT_EQ(m27.precisionCast<672>(), init27); EXPECT_EQ(m27.precisionCast<704>(), init27); EXPECT_EQ(m27.precisionCast<736>(), init27); EXPECT_EQ(m27.precisionCast<768>(), init27);
    EXPECT_EQ(m27.precisionCast<800>(), init27); EXPECT_EQ(m27.precisionCast<832>(), init27); EXPECT_EQ(m27.precisionCast<864>(), init27); EXPECT_EQ(m27.precisionCast<896>(), init27);


    long long init28 = 5869361540369536229; Multiprecision<256> m28 = init28;
    EXPECT_EQ(m28.precisionCast<288>(), init28); EXPECT_EQ(m28.precisionCast<320>(), init28); EXPECT_EQ(m28.precisionCast<352>(), init28); EXPECT_EQ(m28.precisionCast<384>(), init28);
    EXPECT_EQ(m28.precisionCast<416>(), init28); EXPECT_EQ(m28.precisionCast<448>(), init28); EXPECT_EQ(m28.precisionCast<480>(), init28); EXPECT_EQ(m28.precisionCast<512>(), init28);
    EXPECT_EQ(m28.precisionCast<544>(), init28); EXPECT_EQ(m28.precisionCast<576>(), init28); EXPECT_EQ(m28.precisionCast<608>(), init28); EXPECT_EQ(m28.precisionCast<640>(), init28);
    EXPECT_EQ(m28.precisionCast<672>(), init28); EXPECT_EQ(m28.precisionCast<704>(), init28); EXPECT_EQ(m28.precisionCast<736>(), init28); EXPECT_EQ(m28.precisionCast<768>(), init28);
    EXPECT_EQ(m28.precisionCast<800>(), init28); EXPECT_EQ(m28.precisionCast<832>(), init28); EXPECT_EQ(m28.precisionCast<864>(), init28); EXPECT_EQ(m28.precisionCast<896>(), init28);


    long long init29 = 1220581244790115876; Multiprecision<256> m29 = init29;
    EXPECT_EQ(m29.precisionCast<288>(), init29); EXPECT_EQ(m29.precisionCast<320>(), init29); EXPECT_EQ(m29.precisionCast<352>(), init29); EXPECT_EQ(m29.precisionCast<384>(), init29);
    EXPECT_EQ(m29.precisionCast<416>(), init29); EXPECT_EQ(m29.precisionCast<448>(), init29); EXPECT_EQ(m29.precisionCast<480>(), init29); EXPECT_EQ(m29.precisionCast<512>(), init29);
    EXPECT_EQ(m29.precisionCast<544>(), init29); EXPECT_EQ(m29.precisionCast<576>(), init29); EXPECT_EQ(m29.precisionCast<608>(), init29); EXPECT_EQ(m29.precisionCast<640>(), init29);
    EXPECT_EQ(m29.precisionCast<672>(), init29); EXPECT_EQ(m29.precisionCast<704>(), init29); EXPECT_EQ(m29.precisionCast<736>(), init29); EXPECT_EQ(m29.precisionCast<768>(), init29);
    EXPECT_EQ(m29.precisionCast<800>(), init29); EXPECT_EQ(m29.precisionCast<832>(), init29); EXPECT_EQ(m29.precisionCast<864>(), init29); EXPECT_EQ(m29.precisionCast<896>(), init29);
}