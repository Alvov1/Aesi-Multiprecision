#include <gtest/gtest.h>
#include "../../AesiMultiprecision.h"

TEST(Casting, IntegralCast) {
    {
        short v0 = 12660; Aesi512 o0 = v0; EXPECT_EQ(o0.integralCast<short>(), v0);
        short v1 = 1291; Aesi512 o1 = v1; EXPECT_EQ(o1.integralCast<short>(), v1);
        short v2 = 9379; Aesi512 o2 = v2; EXPECT_EQ(o2.integralCast<short>(), v2);
        short v3 = -5507; Aesi512 o3 = v3; EXPECT_EQ(o3.integralCast<short>(), v3);
        short v4 = 10503; Aesi512 o4 = v4; EXPECT_EQ(o4.integralCast<short>(), v4);
        short v5 = -15803; Aesi512 o5 = v5; EXPECT_EQ(o5.integralCast<short>(), v5);
        short v6 = -25434; Aesi512 o6 = v6; EXPECT_EQ(o6.integralCast<short>(), v6);
        short v7 = 31225; Aesi512 o7 = v7; EXPECT_EQ(o7.integralCast<short>(), v7);
        short v8 = 27368; Aesi512 o8 = v8; EXPECT_EQ(o8.integralCast<short>(), v8);
        short v9 = 3854; Aesi512 o9 = v9; EXPECT_EQ(o9.integralCast<short>(), v9);
        short v10 = -31821; Aesi512 o10 = v10; EXPECT_EQ(o10.integralCast<short>(), v10);
        short v11 = -14224; Aesi512 o11 = v11; EXPECT_EQ(o11.integralCast<short>(), v11);
        short v12 = 16812; Aesi512 o12 = v12; EXPECT_EQ(o12.integralCast<short>(), v12);
        short v13 = 24874; Aesi512 o13 = v13; EXPECT_EQ(o13.integralCast<short>(), v13);
        short v14 = 18866; Aesi512 o14 = v14; EXPECT_EQ(o14.integralCast<short>(), v14);
        short v15 = -14076; Aesi512 o15 = v15; EXPECT_EQ(o15.integralCast<short>(), v15);
        short v16 = 31911; Aesi512 o16 = v16; EXPECT_EQ(o16.integralCast<short>(), v16);
        short v17 = -8633; Aesi512 o17 = v17; EXPECT_EQ(o17.integralCast<short>(), v17);
        short v18 = 18120; Aesi512 o18 = v18; EXPECT_EQ(o18.integralCast<short>(), v18);
        short v19 = 3968; Aesi512 o19 = v19; EXPECT_EQ(o19.integralCast<short>(), v19);
    }
    {
        unsigned short v0 = 1941; Aesi512 o0 = v0; EXPECT_EQ(o0.integralCast<unsigned short>(), v0);
        unsigned short v1 = 65126; Aesi512 o1 = v1; EXPECT_EQ(o1.integralCast<unsigned short>(), v1);
        unsigned short v2 = 31635; Aesi512 o2 = v2; EXPECT_EQ(o2.integralCast<unsigned short>(), v2);
        unsigned short v3 = 57499; Aesi512 o3 = v3; EXPECT_EQ(o3.integralCast<unsigned short>(), v3);
        unsigned short v4 = 38058; Aesi512 o4 = v4; EXPECT_EQ(o4.integralCast<unsigned short>(), v4);
        unsigned short v5 = 63039; Aesi512 o5 = v5; EXPECT_EQ(o5.integralCast<unsigned short>(), v5);
        unsigned short v6 = 8296; Aesi512 o6 = v6; EXPECT_EQ(o6.integralCast<unsigned short>(), v6);
        unsigned short v7 = 24236; Aesi512 o7 = v7; EXPECT_EQ(o7.integralCast<unsigned short>(), v7);
        unsigned short v8 = 29619; Aesi512 o8 = v8; EXPECT_EQ(o8.integralCast<unsigned short>(), v8);
        unsigned short v9 = 60555; Aesi512 o9 = v9; EXPECT_EQ(o9.integralCast<unsigned short>(), v9);
        unsigned short v10 = 38254; Aesi512 o10 = v10; EXPECT_EQ(o10.integralCast<unsigned short>(), v10);
        unsigned short v11 = 59479; Aesi512 o11 = v11; EXPECT_EQ(o11.integralCast<unsigned short>(), v11);
        unsigned short v12 = 63392; Aesi512 o12 = v12; EXPECT_EQ(o12.integralCast<unsigned short>(), v12);
        unsigned short v13 = 25152; Aesi512 o13 = v13; EXPECT_EQ(o13.integralCast<unsigned short>(), v13);
        unsigned short v14 = 3758; Aesi512 o14 = v14; EXPECT_EQ(o14.integralCast<unsigned short>(), v14);
        unsigned short v15 = 18321; Aesi512 o15 = v15; EXPECT_EQ(o15.integralCast<unsigned short>(), v15);
        unsigned short v16 = 39353; Aesi512 o16 = v16; EXPECT_EQ(o16.integralCast<unsigned short>(), v16);
        unsigned short v17 = 8739; Aesi512 o17 = v17; EXPECT_EQ(o17.integralCast<unsigned short>(), v17);
        unsigned short v18 = 42112; Aesi512 o18 = v18; EXPECT_EQ(o18.integralCast<unsigned short>(), v18);
        unsigned short v19 = 19620; Aesi512 o19 = v19; EXPECT_EQ(o19.integralCast<unsigned short>(), v19);
    }
    {
        int v0 = 1633072347; Aesi512 o0 = v0; EXPECT_EQ(o0.integralCast<int>(), v0);
        int v1 = 1373045468; Aesi512 o1 = v1; EXPECT_EQ(o1.integralCast<int>(), v1);
        int v2 = -168273879; Aesi512 o2 = v2; EXPECT_EQ(o2.integralCast<int>(), v2);
        int v3 = 1946447499; Aesi512 o3 = v3; EXPECT_EQ(o3.integralCast<int>(), v3);
        int v4 = -182071435; Aesi512 o4 = v4; EXPECT_EQ(o4.integralCast<int>(), v4);
        int v5 = 1000724497; Aesi512 o5 = v5; EXPECT_EQ(o5.integralCast<int>(), v5);
        int v6 = 690522175; Aesi512 o6 = v6; EXPECT_EQ(o6.integralCast<int>(), v6);
        int v7 = 2085869570; Aesi512 o7 = v7; EXPECT_EQ(o7.integralCast<int>(), v7);
        int v8 = 1710099476; Aesi512 o8 = v8; EXPECT_EQ(o8.integralCast<int>(), v8);
        int v9 = -1830783184; Aesi512 o9 = v9; EXPECT_EQ(o9.integralCast<int>(), v9);
        int v10 = 1171383165; Aesi512 o10 = v10; EXPECT_EQ(o10.integralCast<int>(), v10);
        int v11 = -1751249685; Aesi512 o11 = v11; EXPECT_EQ(o11.integralCast<int>(), v11);
        int v12 = -1668504170; Aesi512 o12 = v12; EXPECT_EQ(o12.integralCast<int>(), v12);
        int v13 = -241908811; Aesi512 o13 = v13; EXPECT_EQ(o13.integralCast<int>(), v13);
        int v14 = 540291974; Aesi512 o14 = v14; EXPECT_EQ(o14.integralCast<int>(), v14);
        int v15 = -1643457674; Aesi512 o15 = v15; EXPECT_EQ(o15.integralCast<int>(), v15);
        int v16 = 902106685; Aesi512 o16 = v16; EXPECT_EQ(o16.integralCast<int>(), v16);
        int v17 = -39871497; Aesi512 o17 = v17; EXPECT_EQ(o17.integralCast<int>(), v17);
        int v18 = 888304759; Aesi512 o18 = v18; EXPECT_EQ(o18.integralCast<int>(), v18);
        int v19 = 1568953909; Aesi512 o19 = v19; EXPECT_EQ(o19.integralCast<int>(), v19);
    }
    {
        unsigned v0 = 2005690205; Aesi512 o0 = v0; EXPECT_EQ(o0.integralCast<unsigned>(), v0);
        unsigned v1 = 1449627982; Aesi512 o1 = v1; EXPECT_EQ(o1.integralCast<unsigned>(), v1);
        unsigned v2 = 1101302552; Aesi512 o2 = v2; EXPECT_EQ(o2.integralCast<unsigned>(), v2);
        unsigned v3 = 595765511; Aesi512 o3 = v3; EXPECT_EQ(o3.integralCast<unsigned>(), v3);
        unsigned v4 = 3553780719; Aesi512 o4 = v4; EXPECT_EQ(o4.integralCast<unsigned>(), v4);
        unsigned v5 = 1594854713; Aesi512 o5 = v5; EXPECT_EQ(o5.integralCast<unsigned>(), v5);
        unsigned v6 = 1443312011; Aesi512 o6 = v6; EXPECT_EQ(o6.integralCast<unsigned>(), v6);
        unsigned v7 = 4050435786; Aesi512 o7 = v7; EXPECT_EQ(o7.integralCast<unsigned>(), v7);
        unsigned v8 = 2322528604; Aesi512 o8 = v8; EXPECT_EQ(o8.integralCast<unsigned>(), v8);
        unsigned v9 = 659532747; Aesi512 o9 = v9; EXPECT_EQ(o9.integralCast<unsigned>(), v9);
        unsigned v10 = 4162423578; Aesi512 o10 = v10; EXPECT_EQ(o10.integralCast<unsigned>(), v10);
        unsigned v11 = 2641111926; Aesi512 o11 = v11; EXPECT_EQ(o11.integralCast<unsigned>(), v11);
        unsigned v12 = 4253288915; Aesi512 o12 = v12; EXPECT_EQ(o12.integralCast<unsigned>(), v12);
        unsigned v13 = 3110878193; Aesi512 o13 = v13; EXPECT_EQ(o13.integralCast<unsigned>(), v13);
        unsigned v14 = 587696870; Aesi512 o14 = v14; EXPECT_EQ(o14.integralCast<unsigned>(), v14);
        unsigned v15 = 1153568748; Aesi512 o15 = v15; EXPECT_EQ(o15.integralCast<unsigned>(), v15);
        unsigned v16 = 4175266679; Aesi512 o16 = v16; EXPECT_EQ(o16.integralCast<unsigned>(), v16);
        unsigned v17 = 346810525; Aesi512 o17 = v17; EXPECT_EQ(o17.integralCast<unsigned>(), v17);
        unsigned v18 = 274355692; Aesi512 o18 = v18; EXPECT_EQ(o18.integralCast<unsigned>(), v18);
        unsigned v19 = 3087867327; Aesi512 o19 = v19; EXPECT_EQ(o19.integralCast<unsigned>(), v19);
    }
    {
        long v0 = 634812322182757621; Aesi512 o0 = v0; EXPECT_EQ(o0.integralCast<long>(), v0);
        long v1 = -927449906461017615; Aesi512 o1 = v1; EXPECT_EQ(o1.integralCast<long>(), v1);
        long v2 = -3504628980837960044; Aesi512 o2 = v2; EXPECT_EQ(o2.integralCast<long>(), v2);
        long v3 = -2155375560745272488; Aesi512 o3 = v3; EXPECT_EQ(o3.integralCast<long>(), v3);
        long v4 = -6526318726215051137; Aesi512 o4 = v4; EXPECT_EQ(o4.integralCast<long>(), v4);
        long v5 = -2694746162811842986; Aesi512 o5 = v5; EXPECT_EQ(o5.integralCast<long>(), v5);
        long v6 = 1631485759082327833; Aesi512 o6 = v6; EXPECT_EQ(o6.integralCast<long>(), v6);
        long v7 = 8407645366062381801; Aesi512 o7 = v7; EXPECT_EQ(o7.integralCast<long>(), v7);
        long v8 = 5841857305468410835; Aesi512 o8 = v8; EXPECT_EQ(o8.integralCast<long>(), v8);
        long v9 = 9165801435086311778; Aesi512 o9 = v9; EXPECT_EQ(o9.integralCast<long>(), v9);
        long v10 = 7049900736852272389; Aesi512 o10 = v10; EXPECT_EQ(o10.integralCast<long>(), v10);
        long v11 = -1116039134974475934; Aesi512 o11 = v11; EXPECT_EQ(o11.integralCast<long>(), v11);
        long v12 = 6134624124941989956; Aesi512 o12 = v12; EXPECT_EQ(o12.integralCast<long>(), v12);
        long v13 = -1989814207699167888; Aesi512 o13 = v13; EXPECT_EQ(o13.integralCast<long>(), v13);
        long v14 = -5162674662935837073; Aesi512 o14 = v14; EXPECT_EQ(o14.integralCast<long>(), v14);
        long v15 = 361978787381322157; Aesi512 o15 = v15; EXPECT_EQ(o15.integralCast<long>(), v15);
        long v16 = 2358318038699402381; Aesi512 o16 = v16; EXPECT_EQ(o16.integralCast<long>(), v16);
        long v17 = 6721455152406516439; Aesi512 o17 = v17; EXPECT_EQ(o17.integralCast<long>(), v17);
        long v18 = 6240728962025263647; Aesi512 o18 = v18; EXPECT_EQ(o18.integralCast<long>(), v18);
        long v19 = -7516849453343319431; Aesi512 o19 = v19; EXPECT_EQ(o19.integralCast<long>(), v19);
    }
    {
        unsigned long v0 = 1458432336447339549UL; Aesi512 o0 = v0; EXPECT_EQ(o0.integralCast<unsigned long>(), v0);
        unsigned long v1 = 15166986956279526495UL; Aesi512 o1 = v1; EXPECT_EQ(o1.integralCast<unsigned long>(), v1);
        unsigned long v2 = 843907744079905873UL; Aesi512 o2 = v2; EXPECT_EQ(o2.integralCast<unsigned long>(), v2);
        unsigned long v3 = 16831641047459684976UL; Aesi512 o3 = v3; EXPECT_EQ(o3.integralCast<unsigned long>(), v3);
        unsigned long v4 = 8233716564210995814UL; Aesi512 o4 = v4; EXPECT_EQ(o4.integralCast<unsigned long>(), v4);
        unsigned long v5 = 12056099167404879864UL; Aesi512 o5 = v5; EXPECT_EQ(o5.integralCast<unsigned long>(), v5);
        unsigned long v6 = 9195468763811385966UL; Aesi512 o6 = v6; EXPECT_EQ(o6.integralCast<unsigned long>(), v6);
        unsigned long v7 = 6838015201710322987UL; Aesi512 o7 = v7; EXPECT_EQ(o7.integralCast<unsigned long>(), v7);
        unsigned long v8 = 15909623765770348916UL; Aesi512 o8 = v8; EXPECT_EQ(o8.integralCast<unsigned long>(), v8);
        unsigned long v9 = 580359803260052445UL; Aesi512 o9 = v9; EXPECT_EQ(o9.integralCast<unsigned long>(), v9);
        unsigned long v10 = 1562144106091796071UL; Aesi512 o10 = v10; EXPECT_EQ(o10.integralCast<unsigned long>(), v10);
        unsigned long v11 = 9499213611466063262UL; Aesi512 o11 = v11; EXPECT_EQ(o11.integralCast<unsigned long>(), v11);
        unsigned long v12 = 12515821306029840702UL; Aesi512 o12 = v12; EXPECT_EQ(o12.integralCast<unsigned long>(), v12);
        unsigned long v13 = 5666922082602738570UL; Aesi512 o13 = v13; EXPECT_EQ(o13.integralCast<unsigned long>(), v13);
        unsigned long v14 = 12546046344820481203UL; Aesi512 o14 = v14; EXPECT_EQ(o14.integralCast<unsigned long>(), v14);
        unsigned long v15 = 13562221276840283927UL; Aesi512 o15 = v15; EXPECT_EQ(o15.integralCast<unsigned long>(), v15);
        unsigned long v16 = 12646097440568072921UL; Aesi512 o16 = v16; EXPECT_EQ(o16.integralCast<unsigned long>(), v16);
        unsigned long v17 = 14099814405939634110UL; Aesi512 o17 = v17; EXPECT_EQ(o17.integralCast<unsigned long>(), v17);
        unsigned long v18 = 13036786147570218084UL; Aesi512 o18 = v18; EXPECT_EQ(o18.integralCast<unsigned long>(), v18);
        unsigned long v19 = 14809679552796761321UL; Aesi512 o19 = v19; EXPECT_EQ(o19.integralCast<unsigned long>(), v19);
    }
    {
        long long v0 = -966320832989284844; Aesi512 o0 = v0; EXPECT_EQ(o0.integralCast<long long>(), v0);
        long long v1 = -2539299047194883024; Aesi512 o1 = v1; EXPECT_EQ(o1.integralCast<long long>(), v1);
        long long v2 = -4072657627411227702; Aesi512 o2 = v2; EXPECT_EQ(o2.integralCast<long long>(), v2);
        long long v3 = 6676151703625450453; Aesi512 o3 = v3; EXPECT_EQ(o3.integralCast<long long>(), v3);
        long long v4 = 6024657647873919769; Aesi512 o4 = v4; EXPECT_EQ(o4.integralCast<long long>(), v4);
        long long v5 = -3436387051813180897; Aesi512 o5 = v5; EXPECT_EQ(o5.integralCast<long long>(), v5);
        long long v6 = -930863918495184864; Aesi512 o6 = v6; EXPECT_EQ(o6.integralCast<long long>(), v6);
        long long v7 = -2648679724062892078; Aesi512 o7 = v7; EXPECT_EQ(o7.integralCast<long long>(), v7);
        long long v8 = 163799229998191542; Aesi512 o8 = v8; EXPECT_EQ(o8.integralCast<long long>(), v8);
        long long v9 = 7096985823516738807; Aesi512 o9 = v9; EXPECT_EQ(o9.integralCast<long long>(), v9);
        long long v10 = 9021468662796673265; Aesi512 o10 = v10; EXPECT_EQ(o10.integralCast<long long>(), v10);
        long long v11 = 7329596342632026139; Aesi512 o11 = v11; EXPECT_EQ(o11.integralCast<long long>(), v11);
        long long v12 = 2830097210272796867; Aesi512 o12 = v12; EXPECT_EQ(o12.integralCast<long long>(), v12);
        long long v13 = -8165373913969389069; Aesi512 o13 = v13; EXPECT_EQ(o13.integralCast<long long>(), v13);
        long long v14 = 1849790133061638302; Aesi512 o14 = v14; EXPECT_EQ(o14.integralCast<long long>(), v14);
        long long v15 = -8953474544892738048; Aesi512 o15 = v15; EXPECT_EQ(o15.integralCast<long long>(), v15);
        long long v16 = -4256943809212026391; Aesi512 o16 = v16; EXPECT_EQ(o16.integralCast<long long>(), v16);
        long long v17 = 5358224103492354631; Aesi512 o17 = v17; EXPECT_EQ(o17.integralCast<long long>(), v17);
        long long v18 = -1353061819987477708; Aesi512 o18 = v18; EXPECT_EQ(o18.integralCast<long long>(), v18);
        long long v19 = 5852270391142351733; Aesi512 o19 = v19; EXPECT_EQ(o19.integralCast<long long>(), v19);
    }
    {
        unsigned long long v0 = 7056752338713246902ULL; Aesi512 o0 = v0; EXPECT_EQ(o0.integralCast<unsigned long long>(), v0);
        unsigned long long v1 = 1752111911588599450ULL; Aesi512 o1 = v1; EXPECT_EQ(o1.integralCast<unsigned long long>(), v1);
        unsigned long long v2 = 9479044036425476019ULL; Aesi512 o2 = v2; EXPECT_EQ(o2.integralCast<unsigned long long>(), v2);
        unsigned long long v3 = 5193170295485823118ULL; Aesi512 o3 = v3; EXPECT_EQ(o3.integralCast<unsigned long long>(), v3);
        unsigned long long v4 = 7329594209829406618ULL; Aesi512 o4 = v4; EXPECT_EQ(o4.integralCast<unsigned long long>(), v4);
        unsigned long long v5 = 17408589177287406633ULL; Aesi512 o5 = v5; EXPECT_EQ(o5.integralCast<unsigned long long>(), v5);
        unsigned long long v6 = 10049699202423151920ULL; Aesi512 o6 = v6; EXPECT_EQ(o6.integralCast<unsigned long long>(), v6);
        unsigned long long v7 = 14893142280413871075ULL; Aesi512 o7 = v7; EXPECT_EQ(o7.integralCast<unsigned long long>(), v7);
        unsigned long long v8 = 14477514342547457937ULL; Aesi512 o8 = v8; EXPECT_EQ(o8.integralCast<unsigned long long>(), v8);
        unsigned long long v9 = 14158198760418317512ULL; Aesi512 o9 = v9; EXPECT_EQ(o9.integralCast<unsigned long long>(), v9);
        unsigned long long v10 = 1737450455060823428ULL; Aesi512 o10 = v10; EXPECT_EQ(o10.integralCast<unsigned long long>(), v10);
        unsigned long long v11 = 12956079072589122739ULL; Aesi512 o11 = v11; EXPECT_EQ(o11.integralCast<unsigned long long>(), v11);
        unsigned long long v12 = 15263083373573044880ULL; Aesi512 o12 = v12; EXPECT_EQ(o12.integralCast<unsigned long long>(), v12);
        unsigned long long v13 = 11589228342409016981ULL; Aesi512 o13 = v13; EXPECT_EQ(o13.integralCast<unsigned long long>(), v13);
        unsigned long long v14 = 12279481383607276825ULL; Aesi512 o14 = v14; EXPECT_EQ(o14.integralCast<unsigned long long>(), v14);
        unsigned long long v15 = 11338438110938113974ULL; Aesi512 o15 = v15; EXPECT_EQ(o15.integralCast<unsigned long long>(), v15);
        unsigned long long v16 = 14528990072287077479ULL; Aesi512 o16 = v16; EXPECT_EQ(o16.integralCast<unsigned long long>(), v16);
        unsigned long long v17 = 13754684638618159726ULL; Aesi512 o17 = v17; EXPECT_EQ(o17.integralCast<unsigned long long>(), v17);
        unsigned long long v18 = 6821012615472018674ULL; Aesi512 o18 = v18; EXPECT_EQ(o18.integralCast<unsigned long long>(), v18);
        unsigned long long v19 = 5576836686719978266ULL; Aesi512 o19 = v19; EXPECT_EQ(o19.integralCast<unsigned long long>(), v19);
    }
}