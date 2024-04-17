#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../../Aesi.h"
#include "../../benchmarks/benchmarks.h"

TEST(Signed_IntegralCast, IntegralCast) { EXPECT_TRUE(false); }

TEST(Unsigned_IntegralCast, IntegralCast) {
    unsigned long long v0 = 9563922511731576ULL; Aeu128 o0 = v0; EXPECT_EQ(o0.integralCast<uint64_t>(), v0);
    unsigned long long v1 = 2705626355847155206ULL; Aeu128 o1 = v1; EXPECT_EQ(o1.integralCast<uint64_t>(), v1);
    unsigned long long v2 = 2204869012896022340ULL; Aeu128 o2 = v2; EXPECT_EQ(o2.integralCast<uint64_t>(), v2);
    unsigned long long v3 = 2039975111301504882ULL; Aeu128 o3 = v3; EXPECT_EQ(o3.integralCast<uint64_t>(), v3);
    unsigned long long v4 = 3089255815364648991ULL; Aeu128 o4 = v4; EXPECT_EQ(o4.integralCast<uint64_t>(), v4);
    unsigned long long v5 = 3999747261013330564ULL; Aeu128 o5 = v5; EXPECT_EQ(o5.integralCast<uint64_t>(), v5);
    unsigned long long v6 = 3383934553841059293ULL; Aeu128 o6 = v6; EXPECT_EQ(o6.integralCast<uint64_t>(), v6);
    unsigned long long v7 = 2129812885622137057ULL; Aeu128 o7 = v7; EXPECT_EQ(o7.integralCast<uint64_t>(), v7);
    unsigned long long v8 = 1783048835844072892ULL; Aeu128 o8 = v8; EXPECT_EQ(o8.integralCast<uint64_t>(), v8);
    unsigned long long v9 = 2632389050335074205ULL; Aeu128 o9 = v9; EXPECT_EQ(o9.integralCast<uint64_t>(), v9);
    unsigned long long v10 = 2390630534804768022ULL; Aeu128 o10 = v10; EXPECT_EQ(o10.integralCast<uint64_t>(), v10);
    unsigned long long v11 = 4590625373074199170ULL; Aeu128 o11 = v11; EXPECT_EQ(o11.integralCast<uint64_t>(), v11);
    unsigned long long v12 = 714001976969997177ULL; Aeu128 o12 = v12; EXPECT_EQ(o12.integralCast<uint64_t>(), v12);
    unsigned long long v13 = 216088742529590610ULL; Aeu128 o13 = v13; EXPECT_EQ(o13.integralCast<uint64_t>(), v13);
    unsigned long long v14 = 4150049197491184509ULL; Aeu128 o14 = v14; EXPECT_EQ(o14.integralCast<uint64_t>(), v14);
    unsigned long long v15 = 4590450137326545981ULL; Aeu128 o15 = v15; EXPECT_EQ(o15.integralCast<uint64_t>(), v15);
    unsigned long long v16 = 1065800204708636973ULL; Aeu128 o16 = v16; EXPECT_EQ(o16.integralCast<uint64_t>(), v16);
    unsigned long long v17 = 2824033025738424769ULL; Aeu128 o17 = v17; EXPECT_EQ(o17.integralCast<uint64_t>(), v17);
    unsigned long long v18 = 1005889755414257807ULL; Aeu128 o18 = v18; EXPECT_EQ(o18.integralCast<uint64_t>(), v18);
    unsigned long long v19 = 2934533201305098799ULL; Aeu128 o19 = v19; EXPECT_EQ(o19.integralCast<uint64_t>(), v19);
    unsigned long long v20 = 632419563509435174ULL; Aeu128 o20 = v20; EXPECT_EQ(o20.integralCast<uint64_t>(), v20);
    unsigned long long v21 = 1051028575943916536ULL; Aeu128 o21 = v21; EXPECT_EQ(o21.integralCast<uint64_t>(), v21);
    unsigned long long v22 = 3735063102417190551ULL; Aeu128 o22 = v22; EXPECT_EQ(o22.integralCast<uint64_t>(), v22);
    unsigned long long v23 = 329740787736358964ULL; Aeu128 o23 = v23; EXPECT_EQ(o23.integralCast<uint64_t>(), v23);
    unsigned long long v24 = 918915557684813636ULL; Aeu128 o24 = v24; EXPECT_EQ(o24.integralCast<uint64_t>(), v24);
    unsigned long long v25 = 3891080963393089651ULL; Aeu128 o25 = v25; EXPECT_EQ(o25.integralCast<uint64_t>(), v25);
    unsigned long long v26 = 611698301902317705ULL; Aeu128 o26 = v26; EXPECT_EQ(o26.integralCast<uint64_t>(), v26);
    unsigned long long v27 = 2778995796009947229ULL; Aeu128 o27 = v27; EXPECT_EQ(o27.integralCast<uint64_t>(), v27);
    unsigned long long v28 = 2705973133489440138ULL; Aeu128 o28 = v28; EXPECT_EQ(o28.integralCast<uint64_t>(), v28);
    unsigned long long v29 = 1455913880124089952ULL; Aeu128 o29 = v29; EXPECT_EQ(o29.integralCast<uint64_t>(), v29);
    unsigned long long v30 = 1178035659177528756ULL; Aeu128 o30 = v30; EXPECT_EQ(o30.integralCast<uint64_t>(), v30);
    unsigned long long v31 = 2415626845061624691ULL; Aeu128 o31 = v31; EXPECT_EQ(o31.integralCast<uint64_t>(), v31);
    unsigned long long v32 = 235287766535184341ULL; Aeu128 o32 = v32; EXPECT_EQ(o32.integralCast<uint64_t>(), v32);
    unsigned long long v33 = 3430304044240828669ULL; Aeu128 o33 = v33; EXPECT_EQ(o33.integralCast<uint64_t>(), v33);
    unsigned long long v34 = 3186147322138926830ULL; Aeu128 o34 = v34; EXPECT_EQ(o34.integralCast<uint64_t>(), v34);
    unsigned long long v35 = 3089186871610794757ULL; Aeu128 o35 = v35; EXPECT_EQ(o35.integralCast<uint64_t>(), v35);
    unsigned long long v36 = 2651208519110722841ULL; Aeu128 o36 = v36; EXPECT_EQ(o36.integralCast<uint64_t>(), v36);
    unsigned long long v37 = 1542246942691383376ULL; Aeu128 o37 = v37; EXPECT_EQ(o37.integralCast<uint64_t>(), v37);
    unsigned long long v38 = 2645403377036016679ULL; Aeu128 o38 = v38; EXPECT_EQ(o38.integralCast<uint64_t>(), v38);
    unsigned long long v39 = 3569407294806241482ULL; Aeu128 o39 = v39; EXPECT_EQ(o39.integralCast<uint64_t>(), v39);
    unsigned long long v40 = 780189332546527807ULL; Aeu128 o40 = v40; EXPECT_EQ(o40.integralCast<uint64_t>(), v40);
    unsigned long long v41 = 217919967121748111ULL; Aeu128 o41 = v41; EXPECT_EQ(o41.integralCast<uint64_t>(), v41);
    unsigned long long v42 = 2244368658828994012ULL; Aeu128 o42 = v42; EXPECT_EQ(o42.integralCast<uint64_t>(), v42);
    unsigned long long v43 = 1925390167995875587ULL; Aeu128 o43 = v43; EXPECT_EQ(o43.integralCast<uint64_t>(), v43);
    unsigned long long v44 = 1330214324586281931ULL; Aeu128 o44 = v44; EXPECT_EQ(o44.integralCast<uint64_t>(), v44);
    unsigned long long v45 = 4299809719069193973ULL; Aeu128 o45 = v45; EXPECT_EQ(o45.integralCast<uint64_t>(), v45);
    unsigned long long v46 = 2436712190175650951ULL; Aeu128 o46 = v46; EXPECT_EQ(o46.integralCast<uint64_t>(), v46);
    unsigned long long v47 = 4178593420796969333ULL; Aeu128 o47 = v47; EXPECT_EQ(o47.integralCast<uint64_t>(), v47);
    unsigned long long v48 = 2473916782250088680ULL; Aeu128 o48 = v48; EXPECT_EQ(o48.integralCast<uint64_t>(), v48);
    unsigned long long v49 = 4427108935863985241ULL; Aeu128 o49 = v49; EXPECT_EQ(o49.integralCast<uint64_t>(), v49);
}