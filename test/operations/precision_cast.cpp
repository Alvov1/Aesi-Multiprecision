#include <gtest/gtest.h>
#include "../../AesiMultiprecision.h"

TEST(Casting, PrecisionCast) {
    long long init0 = 7524839891475014690; Aesi<256> m0 = init0;
    EXPECT_EQ(m0.precisionCast<288>(), init0); EXPECT_EQ(m0.precisionCast<320>(), init0); EXPECT_EQ(m0.precisionCast<352>(), init0); EXPECT_EQ(m0.precisionCast<384>(), init0);
    EXPECT_EQ(m0.precisionCast<416>(), init0); EXPECT_EQ(m0.precisionCast<448>(), init0); EXPECT_EQ(m0.precisionCast<480>(), init0); EXPECT_EQ(m0.precisionCast<512>(), init0);
    EXPECT_EQ(m0.precisionCast<544>(), init0); EXPECT_EQ(m0.precisionCast<576>(), init0); EXPECT_EQ(m0.precisionCast<608>(), init0); EXPECT_EQ(m0.precisionCast<640>(), init0);
    EXPECT_EQ(m0.precisionCast<672>(), init0); EXPECT_EQ(m0.precisionCast<704>(), init0); EXPECT_EQ(m0.precisionCast<736>(), init0); EXPECT_EQ(m0.precisionCast<768>(), init0);
    EXPECT_EQ(m0.precisionCast<800>(), init0); EXPECT_EQ(m0.precisionCast<832>(), init0); EXPECT_EQ(m0.precisionCast<864>(), init0); EXPECT_EQ(m0.precisionCast<896>(), init0);


    long long init1 = 8504238977266685499; Aesi<256> m1 = init1;
    EXPECT_EQ(m1.precisionCast<288>(), init1); EXPECT_EQ(m1.precisionCast<320>(), init1); EXPECT_EQ(m1.precisionCast<352>(), init1); EXPECT_EQ(m1.precisionCast<384>(), init1);
    EXPECT_EQ(m1.precisionCast<416>(), init1); EXPECT_EQ(m1.precisionCast<448>(), init1); EXPECT_EQ(m1.precisionCast<480>(), init1); EXPECT_EQ(m1.precisionCast<512>(), init1);
    EXPECT_EQ(m1.precisionCast<544>(), init1); EXPECT_EQ(m1.precisionCast<576>(), init1); EXPECT_EQ(m1.precisionCast<608>(), init1); EXPECT_EQ(m1.precisionCast<640>(), init1);
    EXPECT_EQ(m1.precisionCast<672>(), init1); EXPECT_EQ(m1.precisionCast<704>(), init1); EXPECT_EQ(m1.precisionCast<736>(), init1); EXPECT_EQ(m1.precisionCast<768>(), init1);
    EXPECT_EQ(m1.precisionCast<800>(), init1); EXPECT_EQ(m1.precisionCast<832>(), init1); EXPECT_EQ(m1.precisionCast<864>(), init1); EXPECT_EQ(m1.precisionCast<896>(), init1);


    long long init2 = -7698721489817646536; Aesi<256> m2 = init2;
    EXPECT_EQ(m2.precisionCast<288>(), init2); EXPECT_EQ(m2.precisionCast<320>(), init2); EXPECT_EQ(m2.precisionCast<352>(), init2); EXPECT_EQ(m2.precisionCast<384>(), init2);
    EXPECT_EQ(m2.precisionCast<416>(), init2); EXPECT_EQ(m2.precisionCast<448>(), init2); EXPECT_EQ(m2.precisionCast<480>(), init2); EXPECT_EQ(m2.precisionCast<512>(), init2);
    EXPECT_EQ(m2.precisionCast<544>(), init2); EXPECT_EQ(m2.precisionCast<576>(), init2); EXPECT_EQ(m2.precisionCast<608>(), init2); EXPECT_EQ(m2.precisionCast<640>(), init2);
    EXPECT_EQ(m2.precisionCast<672>(), init2); EXPECT_EQ(m2.precisionCast<704>(), init2); EXPECT_EQ(m2.precisionCast<736>(), init2); EXPECT_EQ(m2.precisionCast<768>(), init2);
    EXPECT_EQ(m2.precisionCast<800>(), init2); EXPECT_EQ(m2.precisionCast<832>(), init2); EXPECT_EQ(m2.precisionCast<864>(), init2); EXPECT_EQ(m2.precisionCast<896>(), init2);


    long long init3 = -6306961477277496307; Aesi<256> m3 = init3;
    EXPECT_EQ(m3.precisionCast<288>(), init3); EXPECT_EQ(m3.precisionCast<320>(), init3); EXPECT_EQ(m3.precisionCast<352>(), init3); EXPECT_EQ(m3.precisionCast<384>(), init3);
    EXPECT_EQ(m3.precisionCast<416>(), init3); EXPECT_EQ(m3.precisionCast<448>(), init3); EXPECT_EQ(m3.precisionCast<480>(), init3); EXPECT_EQ(m3.precisionCast<512>(), init3);
    EXPECT_EQ(m3.precisionCast<544>(), init3); EXPECT_EQ(m3.precisionCast<576>(), init3); EXPECT_EQ(m3.precisionCast<608>(), init3); EXPECT_EQ(m3.precisionCast<640>(), init3);
    EXPECT_EQ(m3.precisionCast<672>(), init3); EXPECT_EQ(m3.precisionCast<704>(), init3); EXPECT_EQ(m3.precisionCast<736>(), init3); EXPECT_EQ(m3.precisionCast<768>(), init3);
    EXPECT_EQ(m3.precisionCast<800>(), init3); EXPECT_EQ(m3.precisionCast<832>(), init3); EXPECT_EQ(m3.precisionCast<864>(), init3); EXPECT_EQ(m3.precisionCast<896>(), init3);


    long long init4 = -1439223510727390957; Aesi<256> m4 = init4;
    EXPECT_EQ(m4.precisionCast<288>(), init4); EXPECT_EQ(m4.precisionCast<320>(), init4); EXPECT_EQ(m4.precisionCast<352>(), init4); EXPECT_EQ(m4.precisionCast<384>(), init4);
    EXPECT_EQ(m4.precisionCast<416>(), init4); EXPECT_EQ(m4.precisionCast<448>(), init4); EXPECT_EQ(m4.precisionCast<480>(), init4); EXPECT_EQ(m4.precisionCast<512>(), init4);
    EXPECT_EQ(m4.precisionCast<544>(), init4); EXPECT_EQ(m4.precisionCast<576>(), init4); EXPECT_EQ(m4.precisionCast<608>(), init4); EXPECT_EQ(m4.precisionCast<640>(), init4);
    EXPECT_EQ(m4.precisionCast<672>(), init4); EXPECT_EQ(m4.precisionCast<704>(), init4); EXPECT_EQ(m4.precisionCast<736>(), init4); EXPECT_EQ(m4.precisionCast<768>(), init4);
    EXPECT_EQ(m4.precisionCast<800>(), init4); EXPECT_EQ(m4.precisionCast<832>(), init4); EXPECT_EQ(m4.precisionCast<864>(), init4); EXPECT_EQ(m4.precisionCast<896>(), init4);


    long long init5 = -9108790223466577088; Aesi<256> m5 = init5;
    EXPECT_EQ(m5.precisionCast<288>(), init5); EXPECT_EQ(m5.precisionCast<320>(), init5); EXPECT_EQ(m5.precisionCast<352>(), init5); EXPECT_EQ(m5.precisionCast<384>(), init5);
    EXPECT_EQ(m5.precisionCast<416>(), init5); EXPECT_EQ(m5.precisionCast<448>(), init5); EXPECT_EQ(m5.precisionCast<480>(), init5); EXPECT_EQ(m5.precisionCast<512>(), init5);
    EXPECT_EQ(m5.precisionCast<544>(), init5); EXPECT_EQ(m5.precisionCast<576>(), init5); EXPECT_EQ(m5.precisionCast<608>(), init5); EXPECT_EQ(m5.precisionCast<640>(), init5);
    EXPECT_EQ(m5.precisionCast<672>(), init5); EXPECT_EQ(m5.precisionCast<704>(), init5); EXPECT_EQ(m5.precisionCast<736>(), init5); EXPECT_EQ(m5.precisionCast<768>(), init5);
    EXPECT_EQ(m5.precisionCast<800>(), init5); EXPECT_EQ(m5.precisionCast<832>(), init5); EXPECT_EQ(m5.precisionCast<864>(), init5); EXPECT_EQ(m5.precisionCast<896>(), init5);


    long long init6 = 7336368703164388051; Aesi<256> m6 = init6;
    EXPECT_EQ(m6.precisionCast<288>(), init6); EXPECT_EQ(m6.precisionCast<320>(), init6); EXPECT_EQ(m6.precisionCast<352>(), init6); EXPECT_EQ(m6.precisionCast<384>(), init6);
    EXPECT_EQ(m6.precisionCast<416>(), init6); EXPECT_EQ(m6.precisionCast<448>(), init6); EXPECT_EQ(m6.precisionCast<480>(), init6); EXPECT_EQ(m6.precisionCast<512>(), init6);
    EXPECT_EQ(m6.precisionCast<544>(), init6); EXPECT_EQ(m6.precisionCast<576>(), init6); EXPECT_EQ(m6.precisionCast<608>(), init6); EXPECT_EQ(m6.precisionCast<640>(), init6);
    EXPECT_EQ(m6.precisionCast<672>(), init6); EXPECT_EQ(m6.precisionCast<704>(), init6); EXPECT_EQ(m6.precisionCast<736>(), init6); EXPECT_EQ(m6.precisionCast<768>(), init6);
    EXPECT_EQ(m6.precisionCast<800>(), init6); EXPECT_EQ(m6.precisionCast<832>(), init6); EXPECT_EQ(m6.precisionCast<864>(), init6); EXPECT_EQ(m6.precisionCast<896>(), init6);


    long long init7 = 2314166570054255036; Aesi<256> m7 = init7;
    EXPECT_EQ(m7.precisionCast<288>(), init7); EXPECT_EQ(m7.precisionCast<320>(), init7); EXPECT_EQ(m7.precisionCast<352>(), init7); EXPECT_EQ(m7.precisionCast<384>(), init7);
    EXPECT_EQ(m7.precisionCast<416>(), init7); EXPECT_EQ(m7.precisionCast<448>(), init7); EXPECT_EQ(m7.precisionCast<480>(), init7); EXPECT_EQ(m7.precisionCast<512>(), init7);
    EXPECT_EQ(m7.precisionCast<544>(), init7); EXPECT_EQ(m7.precisionCast<576>(), init7); EXPECT_EQ(m7.precisionCast<608>(), init7); EXPECT_EQ(m7.precisionCast<640>(), init7);
    EXPECT_EQ(m7.precisionCast<672>(), init7); EXPECT_EQ(m7.precisionCast<704>(), init7); EXPECT_EQ(m7.precisionCast<736>(), init7); EXPECT_EQ(m7.precisionCast<768>(), init7);
    EXPECT_EQ(m7.precisionCast<800>(), init7); EXPECT_EQ(m7.precisionCast<832>(), init7); EXPECT_EQ(m7.precisionCast<864>(), init7); EXPECT_EQ(m7.precisionCast<896>(), init7);


    long long init8 = 7305419937287767522; Aesi<256> m8 = init8;
    EXPECT_EQ(m8.precisionCast<288>(), init8); EXPECT_EQ(m8.precisionCast<320>(), init8); EXPECT_EQ(m8.precisionCast<352>(), init8); EXPECT_EQ(m8.precisionCast<384>(), init8);
    EXPECT_EQ(m8.precisionCast<416>(), init8); EXPECT_EQ(m8.precisionCast<448>(), init8); EXPECT_EQ(m8.precisionCast<480>(), init8); EXPECT_EQ(m8.precisionCast<512>(), init8);
    EXPECT_EQ(m8.precisionCast<544>(), init8); EXPECT_EQ(m8.precisionCast<576>(), init8); EXPECT_EQ(m8.precisionCast<608>(), init8); EXPECT_EQ(m8.precisionCast<640>(), init8);
    EXPECT_EQ(m8.precisionCast<672>(), init8); EXPECT_EQ(m8.precisionCast<704>(), init8); EXPECT_EQ(m8.precisionCast<736>(), init8); EXPECT_EQ(m8.precisionCast<768>(), init8);
    EXPECT_EQ(m8.precisionCast<800>(), init8); EXPECT_EQ(m8.precisionCast<832>(), init8); EXPECT_EQ(m8.precisionCast<864>(), init8); EXPECT_EQ(m8.precisionCast<896>(), init8);


    long long init9 = -2561303725389650172; Aesi<256> m9 = init9;
    EXPECT_EQ(m9.precisionCast<288>(), init9); EXPECT_EQ(m9.precisionCast<320>(), init9); EXPECT_EQ(m9.precisionCast<352>(), init9); EXPECT_EQ(m9.precisionCast<384>(), init9);
    EXPECT_EQ(m9.precisionCast<416>(), init9); EXPECT_EQ(m9.precisionCast<448>(), init9); EXPECT_EQ(m9.precisionCast<480>(), init9); EXPECT_EQ(m9.precisionCast<512>(), init9);
    EXPECT_EQ(m9.precisionCast<544>(), init9); EXPECT_EQ(m9.precisionCast<576>(), init9); EXPECT_EQ(m9.precisionCast<608>(), init9); EXPECT_EQ(m9.precisionCast<640>(), init9);
    EXPECT_EQ(m9.precisionCast<672>(), init9); EXPECT_EQ(m9.precisionCast<704>(), init9); EXPECT_EQ(m9.precisionCast<736>(), init9); EXPECT_EQ(m9.precisionCast<768>(), init9);
    EXPECT_EQ(m9.precisionCast<800>(), init9); EXPECT_EQ(m9.precisionCast<832>(), init9); EXPECT_EQ(m9.precisionCast<864>(), init9); EXPECT_EQ(m9.precisionCast<896>(), init9);


    long long init10 = 3383802386880547846; Aesi<256> m10 = init10;
    EXPECT_EQ(m10.precisionCast<288>(), init10); EXPECT_EQ(m10.precisionCast<320>(), init10); EXPECT_EQ(m10.precisionCast<352>(), init10); EXPECT_EQ(m10.precisionCast<384>(), init10);
    EXPECT_EQ(m10.precisionCast<416>(), init10); EXPECT_EQ(m10.precisionCast<448>(), init10); EXPECT_EQ(m10.precisionCast<480>(), init10); EXPECT_EQ(m10.precisionCast<512>(), init10);
    EXPECT_EQ(m10.precisionCast<544>(), init10); EXPECT_EQ(m10.precisionCast<576>(), init10); EXPECT_EQ(m10.precisionCast<608>(), init10); EXPECT_EQ(m10.precisionCast<640>(), init10);
    EXPECT_EQ(m10.precisionCast<672>(), init10); EXPECT_EQ(m10.precisionCast<704>(), init10); EXPECT_EQ(m10.precisionCast<736>(), init10); EXPECT_EQ(m10.precisionCast<768>(), init10);
    EXPECT_EQ(m10.precisionCast<800>(), init10); EXPECT_EQ(m10.precisionCast<832>(), init10); EXPECT_EQ(m10.precisionCast<864>(), init10); EXPECT_EQ(m10.precisionCast<896>(), init10);


    long long init11 = 5703252986956204740; Aesi<256> m11 = init11;
    EXPECT_EQ(m11.precisionCast<288>(), init11); EXPECT_EQ(m11.precisionCast<320>(), init11); EXPECT_EQ(m11.precisionCast<352>(), init11); EXPECT_EQ(m11.precisionCast<384>(), init11);
    EXPECT_EQ(m11.precisionCast<416>(), init11); EXPECT_EQ(m11.precisionCast<448>(), init11); EXPECT_EQ(m11.precisionCast<480>(), init11); EXPECT_EQ(m11.precisionCast<512>(), init11);
    EXPECT_EQ(m11.precisionCast<544>(), init11); EXPECT_EQ(m11.precisionCast<576>(), init11); EXPECT_EQ(m11.precisionCast<608>(), init11); EXPECT_EQ(m11.precisionCast<640>(), init11);
    EXPECT_EQ(m11.precisionCast<672>(), init11); EXPECT_EQ(m11.precisionCast<704>(), init11); EXPECT_EQ(m11.precisionCast<736>(), init11); EXPECT_EQ(m11.precisionCast<768>(), init11);
    EXPECT_EQ(m11.precisionCast<800>(), init11); EXPECT_EQ(m11.precisionCast<832>(), init11); EXPECT_EQ(m11.precisionCast<864>(), init11); EXPECT_EQ(m11.precisionCast<896>(), init11);


    long long init12 = 1248358330450460190; Aesi<256> m12 = init12;
    EXPECT_EQ(m12.precisionCast<288>(), init12); EXPECT_EQ(m12.precisionCast<320>(), init12); EXPECT_EQ(m12.precisionCast<352>(), init12); EXPECT_EQ(m12.precisionCast<384>(), init12);
    EXPECT_EQ(m12.precisionCast<416>(), init12); EXPECT_EQ(m12.precisionCast<448>(), init12); EXPECT_EQ(m12.precisionCast<480>(), init12); EXPECT_EQ(m12.precisionCast<512>(), init12);
    EXPECT_EQ(m12.precisionCast<544>(), init12); EXPECT_EQ(m12.precisionCast<576>(), init12); EXPECT_EQ(m12.precisionCast<608>(), init12); EXPECT_EQ(m12.precisionCast<640>(), init12);
    EXPECT_EQ(m12.precisionCast<672>(), init12); EXPECT_EQ(m12.precisionCast<704>(), init12); EXPECT_EQ(m12.precisionCast<736>(), init12); EXPECT_EQ(m12.precisionCast<768>(), init12);
    EXPECT_EQ(m12.precisionCast<800>(), init12); EXPECT_EQ(m12.precisionCast<832>(), init12); EXPECT_EQ(m12.precisionCast<864>(), init12); EXPECT_EQ(m12.precisionCast<896>(), init12);


    long long init13 = 4175697953158773741; Aesi<256> m13 = init13;
    EXPECT_EQ(m13.precisionCast<288>(), init13); EXPECT_EQ(m13.precisionCast<320>(), init13); EXPECT_EQ(m13.precisionCast<352>(), init13); EXPECT_EQ(m13.precisionCast<384>(), init13);
    EXPECT_EQ(m13.precisionCast<416>(), init13); EXPECT_EQ(m13.precisionCast<448>(), init13); EXPECT_EQ(m13.precisionCast<480>(), init13); EXPECT_EQ(m13.precisionCast<512>(), init13);
    EXPECT_EQ(m13.precisionCast<544>(), init13); EXPECT_EQ(m13.precisionCast<576>(), init13); EXPECT_EQ(m13.precisionCast<608>(), init13); EXPECT_EQ(m13.precisionCast<640>(), init13);
    EXPECT_EQ(m13.precisionCast<672>(), init13); EXPECT_EQ(m13.precisionCast<704>(), init13); EXPECT_EQ(m13.precisionCast<736>(), init13); EXPECT_EQ(m13.precisionCast<768>(), init13);
    EXPECT_EQ(m13.precisionCast<800>(), init13); EXPECT_EQ(m13.precisionCast<832>(), init13); EXPECT_EQ(m13.precisionCast<864>(), init13); EXPECT_EQ(m13.precisionCast<896>(), init13);


    long long init14 = -2938517079411100489; Aesi<256> m14 = init14;
    EXPECT_EQ(m14.precisionCast<288>(), init14); EXPECT_EQ(m14.precisionCast<320>(), init14); EXPECT_EQ(m14.precisionCast<352>(), init14); EXPECT_EQ(m14.precisionCast<384>(), init14);
    EXPECT_EQ(m14.precisionCast<416>(), init14); EXPECT_EQ(m14.precisionCast<448>(), init14); EXPECT_EQ(m14.precisionCast<480>(), init14); EXPECT_EQ(m14.precisionCast<512>(), init14);
    EXPECT_EQ(m14.precisionCast<544>(), init14); EXPECT_EQ(m14.precisionCast<576>(), init14); EXPECT_EQ(m14.precisionCast<608>(), init14); EXPECT_EQ(m14.precisionCast<640>(), init14);
    EXPECT_EQ(m14.precisionCast<672>(), init14); EXPECT_EQ(m14.precisionCast<704>(), init14); EXPECT_EQ(m14.precisionCast<736>(), init14); EXPECT_EQ(m14.precisionCast<768>(), init14);
    EXPECT_EQ(m14.precisionCast<800>(), init14); EXPECT_EQ(m14.precisionCast<832>(), init14); EXPECT_EQ(m14.precisionCast<864>(), init14); EXPECT_EQ(m14.precisionCast<896>(), init14);


    long long init15 = -6902692574326511305; Aesi<256> m15 = init15;
    EXPECT_EQ(m15.precisionCast<288>(), init15); EXPECT_EQ(m15.precisionCast<320>(), init15); EXPECT_EQ(m15.precisionCast<352>(), init15); EXPECT_EQ(m15.precisionCast<384>(), init15);
    EXPECT_EQ(m15.precisionCast<416>(), init15); EXPECT_EQ(m15.precisionCast<448>(), init15); EXPECT_EQ(m15.precisionCast<480>(), init15); EXPECT_EQ(m15.precisionCast<512>(), init15);
    EXPECT_EQ(m15.precisionCast<544>(), init15); EXPECT_EQ(m15.precisionCast<576>(), init15); EXPECT_EQ(m15.precisionCast<608>(), init15); EXPECT_EQ(m15.precisionCast<640>(), init15);
    EXPECT_EQ(m15.precisionCast<672>(), init15); EXPECT_EQ(m15.precisionCast<704>(), init15); EXPECT_EQ(m15.precisionCast<736>(), init15); EXPECT_EQ(m15.precisionCast<768>(), init15);
    EXPECT_EQ(m15.precisionCast<800>(), init15); EXPECT_EQ(m15.precisionCast<832>(), init15); EXPECT_EQ(m15.precisionCast<864>(), init15); EXPECT_EQ(m15.precisionCast<896>(), init15);


    long long init16 = -2409078837645875887; Aesi<256> m16 = init16;
    EXPECT_EQ(m16.precisionCast<288>(), init16); EXPECT_EQ(m16.precisionCast<320>(), init16); EXPECT_EQ(m16.precisionCast<352>(), init16); EXPECT_EQ(m16.precisionCast<384>(), init16);
    EXPECT_EQ(m16.precisionCast<416>(), init16); EXPECT_EQ(m16.precisionCast<448>(), init16); EXPECT_EQ(m16.precisionCast<480>(), init16); EXPECT_EQ(m16.precisionCast<512>(), init16);
    EXPECT_EQ(m16.precisionCast<544>(), init16); EXPECT_EQ(m16.precisionCast<576>(), init16); EXPECT_EQ(m16.precisionCast<608>(), init16); EXPECT_EQ(m16.precisionCast<640>(), init16);
    EXPECT_EQ(m16.precisionCast<672>(), init16); EXPECT_EQ(m16.precisionCast<704>(), init16); EXPECT_EQ(m16.precisionCast<736>(), init16); EXPECT_EQ(m16.precisionCast<768>(), init16);
    EXPECT_EQ(m16.precisionCast<800>(), init16); EXPECT_EQ(m16.precisionCast<832>(), init16); EXPECT_EQ(m16.precisionCast<864>(), init16); EXPECT_EQ(m16.precisionCast<896>(), init16);


    long long init17 = -7506774356715976030; Aesi<256> m17 = init17;
    EXPECT_EQ(m17.precisionCast<288>(), init17); EXPECT_EQ(m17.precisionCast<320>(), init17); EXPECT_EQ(m17.precisionCast<352>(), init17); EXPECT_EQ(m17.precisionCast<384>(), init17);
    EXPECT_EQ(m17.precisionCast<416>(), init17); EXPECT_EQ(m17.precisionCast<448>(), init17); EXPECT_EQ(m17.precisionCast<480>(), init17); EXPECT_EQ(m17.precisionCast<512>(), init17);
    EXPECT_EQ(m17.precisionCast<544>(), init17); EXPECT_EQ(m17.precisionCast<576>(), init17); EXPECT_EQ(m17.precisionCast<608>(), init17); EXPECT_EQ(m17.precisionCast<640>(), init17);
    EXPECT_EQ(m17.precisionCast<672>(), init17); EXPECT_EQ(m17.precisionCast<704>(), init17); EXPECT_EQ(m17.precisionCast<736>(), init17); EXPECT_EQ(m17.precisionCast<768>(), init17);
    EXPECT_EQ(m17.precisionCast<800>(), init17); EXPECT_EQ(m17.precisionCast<832>(), init17); EXPECT_EQ(m17.precisionCast<864>(), init17); EXPECT_EQ(m17.precisionCast<896>(), init17);


    long long init18 = 227163478686314711; Aesi<256> m18 = init18;
    EXPECT_EQ(m18.precisionCast<288>(), init18); EXPECT_EQ(m18.precisionCast<320>(), init18); EXPECT_EQ(m18.precisionCast<352>(), init18); EXPECT_EQ(m18.precisionCast<384>(), init18);
    EXPECT_EQ(m18.precisionCast<416>(), init18); EXPECT_EQ(m18.precisionCast<448>(), init18); EXPECT_EQ(m18.precisionCast<480>(), init18); EXPECT_EQ(m18.precisionCast<512>(), init18);
    EXPECT_EQ(m18.precisionCast<544>(), init18); EXPECT_EQ(m18.precisionCast<576>(), init18); EXPECT_EQ(m18.precisionCast<608>(), init18); EXPECT_EQ(m18.precisionCast<640>(), init18);
    EXPECT_EQ(m18.precisionCast<672>(), init18); EXPECT_EQ(m18.precisionCast<704>(), init18); EXPECT_EQ(m18.precisionCast<736>(), init18); EXPECT_EQ(m18.precisionCast<768>(), init18);
    EXPECT_EQ(m18.precisionCast<800>(), init18); EXPECT_EQ(m18.precisionCast<832>(), init18); EXPECT_EQ(m18.precisionCast<864>(), init18); EXPECT_EQ(m18.precisionCast<896>(), init18);


    long long init19 = -4966632676207774604; Aesi<256> m19 = init19;
    EXPECT_EQ(m19.precisionCast<288>(), init19); EXPECT_EQ(m19.precisionCast<320>(), init19); EXPECT_EQ(m19.precisionCast<352>(), init19); EXPECT_EQ(m19.precisionCast<384>(), init19);
    EXPECT_EQ(m19.precisionCast<416>(), init19); EXPECT_EQ(m19.precisionCast<448>(), init19); EXPECT_EQ(m19.precisionCast<480>(), init19); EXPECT_EQ(m19.precisionCast<512>(), init19);
    EXPECT_EQ(m19.precisionCast<544>(), init19); EXPECT_EQ(m19.precisionCast<576>(), init19); EXPECT_EQ(m19.precisionCast<608>(), init19); EXPECT_EQ(m19.precisionCast<640>(), init19);
    EXPECT_EQ(m19.precisionCast<672>(), init19); EXPECT_EQ(m19.precisionCast<704>(), init19); EXPECT_EQ(m19.precisionCast<736>(), init19); EXPECT_EQ(m19.precisionCast<768>(), init19);
    EXPECT_EQ(m19.precisionCast<800>(), init19); EXPECT_EQ(m19.precisionCast<832>(), init19); EXPECT_EQ(m19.precisionCast<864>(), init19); EXPECT_EQ(m19.precisionCast<896>(), init19);


    long long init20 = -4021786551316732423; Aesi<256> m20 = init20;
    EXPECT_EQ(m20.precisionCast<288>(), init20); EXPECT_EQ(m20.precisionCast<320>(), init20); EXPECT_EQ(m20.precisionCast<352>(), init20); EXPECT_EQ(m20.precisionCast<384>(), init20);
    EXPECT_EQ(m20.precisionCast<416>(), init20); EXPECT_EQ(m20.precisionCast<448>(), init20); EXPECT_EQ(m20.precisionCast<480>(), init20); EXPECT_EQ(m20.precisionCast<512>(), init20);
    EXPECT_EQ(m20.precisionCast<544>(), init20); EXPECT_EQ(m20.precisionCast<576>(), init20); EXPECT_EQ(m20.precisionCast<608>(), init20); EXPECT_EQ(m20.precisionCast<640>(), init20);
    EXPECT_EQ(m20.precisionCast<672>(), init20); EXPECT_EQ(m20.precisionCast<704>(), init20); EXPECT_EQ(m20.precisionCast<736>(), init20); EXPECT_EQ(m20.precisionCast<768>(), init20);
    EXPECT_EQ(m20.precisionCast<800>(), init20); EXPECT_EQ(m20.precisionCast<832>(), init20); EXPECT_EQ(m20.precisionCast<864>(), init20); EXPECT_EQ(m20.precisionCast<896>(), init20);


    long long init21 = 7309039963756908723; Aesi<256> m21 = init21;
    EXPECT_EQ(m21.precisionCast<288>(), init21); EXPECT_EQ(m21.precisionCast<320>(), init21); EXPECT_EQ(m21.precisionCast<352>(), init21); EXPECT_EQ(m21.precisionCast<384>(), init21);
    EXPECT_EQ(m21.precisionCast<416>(), init21); EXPECT_EQ(m21.precisionCast<448>(), init21); EXPECT_EQ(m21.precisionCast<480>(), init21); EXPECT_EQ(m21.precisionCast<512>(), init21);
    EXPECT_EQ(m21.precisionCast<544>(), init21); EXPECT_EQ(m21.precisionCast<576>(), init21); EXPECT_EQ(m21.precisionCast<608>(), init21); EXPECT_EQ(m21.precisionCast<640>(), init21);
    EXPECT_EQ(m21.precisionCast<672>(), init21); EXPECT_EQ(m21.precisionCast<704>(), init21); EXPECT_EQ(m21.precisionCast<736>(), init21); EXPECT_EQ(m21.precisionCast<768>(), init21);
    EXPECT_EQ(m21.precisionCast<800>(), init21); EXPECT_EQ(m21.precisionCast<832>(), init21); EXPECT_EQ(m21.precisionCast<864>(), init21); EXPECT_EQ(m21.precisionCast<896>(), init21);


    long long init22 = -4037333522000484564; Aesi<256> m22 = init22;
    EXPECT_EQ(m22.precisionCast<288>(), init22); EXPECT_EQ(m22.precisionCast<320>(), init22); EXPECT_EQ(m22.precisionCast<352>(), init22); EXPECT_EQ(m22.precisionCast<384>(), init22);
    EXPECT_EQ(m22.precisionCast<416>(), init22); EXPECT_EQ(m22.precisionCast<448>(), init22); EXPECT_EQ(m22.precisionCast<480>(), init22); EXPECT_EQ(m22.precisionCast<512>(), init22);
    EXPECT_EQ(m22.precisionCast<544>(), init22); EXPECT_EQ(m22.precisionCast<576>(), init22); EXPECT_EQ(m22.precisionCast<608>(), init22); EXPECT_EQ(m22.precisionCast<640>(), init22);
    EXPECT_EQ(m22.precisionCast<672>(), init22); EXPECT_EQ(m22.precisionCast<704>(), init22); EXPECT_EQ(m22.precisionCast<736>(), init22); EXPECT_EQ(m22.precisionCast<768>(), init22);
    EXPECT_EQ(m22.precisionCast<800>(), init22); EXPECT_EQ(m22.precisionCast<832>(), init22); EXPECT_EQ(m22.precisionCast<864>(), init22); EXPECT_EQ(m22.precisionCast<896>(), init22);


    long long init23 = 3067962240659289255; Aesi<256> m23 = init23;
    EXPECT_EQ(m23.precisionCast<288>(), init23); EXPECT_EQ(m23.precisionCast<320>(), init23); EXPECT_EQ(m23.precisionCast<352>(), init23); EXPECT_EQ(m23.precisionCast<384>(), init23);
    EXPECT_EQ(m23.precisionCast<416>(), init23); EXPECT_EQ(m23.precisionCast<448>(), init23); EXPECT_EQ(m23.precisionCast<480>(), init23); EXPECT_EQ(m23.precisionCast<512>(), init23);
    EXPECT_EQ(m23.precisionCast<544>(), init23); EXPECT_EQ(m23.precisionCast<576>(), init23); EXPECT_EQ(m23.precisionCast<608>(), init23); EXPECT_EQ(m23.precisionCast<640>(), init23);
    EXPECT_EQ(m23.precisionCast<672>(), init23); EXPECT_EQ(m23.precisionCast<704>(), init23); EXPECT_EQ(m23.precisionCast<736>(), init23); EXPECT_EQ(m23.precisionCast<768>(), init23);
    EXPECT_EQ(m23.precisionCast<800>(), init23); EXPECT_EQ(m23.precisionCast<832>(), init23); EXPECT_EQ(m23.precisionCast<864>(), init23); EXPECT_EQ(m23.precisionCast<896>(), init23);


    long long init24 = -3707800888550614839; Aesi<256> m24 = init24;
    EXPECT_EQ(m24.precisionCast<288>(), init24); EXPECT_EQ(m24.precisionCast<320>(), init24); EXPECT_EQ(m24.precisionCast<352>(), init24); EXPECT_EQ(m24.precisionCast<384>(), init24);
    EXPECT_EQ(m24.precisionCast<416>(), init24); EXPECT_EQ(m24.precisionCast<448>(), init24); EXPECT_EQ(m24.precisionCast<480>(), init24); EXPECT_EQ(m24.precisionCast<512>(), init24);
    EXPECT_EQ(m24.precisionCast<544>(), init24); EXPECT_EQ(m24.precisionCast<576>(), init24); EXPECT_EQ(m24.precisionCast<608>(), init24); EXPECT_EQ(m24.precisionCast<640>(), init24);
    EXPECT_EQ(m24.precisionCast<672>(), init24); EXPECT_EQ(m24.precisionCast<704>(), init24); EXPECT_EQ(m24.precisionCast<736>(), init24); EXPECT_EQ(m24.precisionCast<768>(), init24);
    EXPECT_EQ(m24.precisionCast<800>(), init24); EXPECT_EQ(m24.precisionCast<832>(), init24); EXPECT_EQ(m24.precisionCast<864>(), init24); EXPECT_EQ(m24.precisionCast<896>(), init24);


    long long init25 = -6878944643378492415; Aesi<256> m25 = init25;
    EXPECT_EQ(m25.precisionCast<288>(), init25); EXPECT_EQ(m25.precisionCast<320>(), init25); EXPECT_EQ(m25.precisionCast<352>(), init25); EXPECT_EQ(m25.precisionCast<384>(), init25);
    EXPECT_EQ(m25.precisionCast<416>(), init25); EXPECT_EQ(m25.precisionCast<448>(), init25); EXPECT_EQ(m25.precisionCast<480>(), init25); EXPECT_EQ(m25.precisionCast<512>(), init25);
    EXPECT_EQ(m25.precisionCast<544>(), init25); EXPECT_EQ(m25.precisionCast<576>(), init25); EXPECT_EQ(m25.precisionCast<608>(), init25); EXPECT_EQ(m25.precisionCast<640>(), init25);
    EXPECT_EQ(m25.precisionCast<672>(), init25); EXPECT_EQ(m25.precisionCast<704>(), init25); EXPECT_EQ(m25.precisionCast<736>(), init25); EXPECT_EQ(m25.precisionCast<768>(), init25);
    EXPECT_EQ(m25.precisionCast<800>(), init25); EXPECT_EQ(m25.precisionCast<832>(), init25); EXPECT_EQ(m25.precisionCast<864>(), init25); EXPECT_EQ(m25.precisionCast<896>(), init25);


    long long init26 = -590213465179246446; Aesi<256> m26 = init26;
    EXPECT_EQ(m26.precisionCast<288>(), init26); EXPECT_EQ(m26.precisionCast<320>(), init26); EXPECT_EQ(m26.precisionCast<352>(), init26); EXPECT_EQ(m26.precisionCast<384>(), init26);
    EXPECT_EQ(m26.precisionCast<416>(), init26); EXPECT_EQ(m26.precisionCast<448>(), init26); EXPECT_EQ(m26.precisionCast<480>(), init26); EXPECT_EQ(m26.precisionCast<512>(), init26);
    EXPECT_EQ(m26.precisionCast<544>(), init26); EXPECT_EQ(m26.precisionCast<576>(), init26); EXPECT_EQ(m26.precisionCast<608>(), init26); EXPECT_EQ(m26.precisionCast<640>(), init26);
    EXPECT_EQ(m26.precisionCast<672>(), init26); EXPECT_EQ(m26.precisionCast<704>(), init26); EXPECT_EQ(m26.precisionCast<736>(), init26); EXPECT_EQ(m26.precisionCast<768>(), init26);
    EXPECT_EQ(m26.precisionCast<800>(), init26); EXPECT_EQ(m26.precisionCast<832>(), init26); EXPECT_EQ(m26.precisionCast<864>(), init26); EXPECT_EQ(m26.precisionCast<896>(), init26);


    long long init27 = 6023706187360170892; Aesi<256> m27 = init27;
    EXPECT_EQ(m27.precisionCast<288>(), init27); EXPECT_EQ(m27.precisionCast<320>(), init27); EXPECT_EQ(m27.precisionCast<352>(), init27); EXPECT_EQ(m27.precisionCast<384>(), init27);
    EXPECT_EQ(m27.precisionCast<416>(), init27); EXPECT_EQ(m27.precisionCast<448>(), init27); EXPECT_EQ(m27.precisionCast<480>(), init27); EXPECT_EQ(m27.precisionCast<512>(), init27);
    EXPECT_EQ(m27.precisionCast<544>(), init27); EXPECT_EQ(m27.precisionCast<576>(), init27); EXPECT_EQ(m27.precisionCast<608>(), init27); EXPECT_EQ(m27.precisionCast<640>(), init27);
    EXPECT_EQ(m27.precisionCast<672>(), init27); EXPECT_EQ(m27.precisionCast<704>(), init27); EXPECT_EQ(m27.precisionCast<736>(), init27); EXPECT_EQ(m27.precisionCast<768>(), init27);
    EXPECT_EQ(m27.precisionCast<800>(), init27); EXPECT_EQ(m27.precisionCast<832>(), init27); EXPECT_EQ(m27.precisionCast<864>(), init27); EXPECT_EQ(m27.precisionCast<896>(), init27);


    long long init28 = 5869361540369536229; Aesi<256> m28 = init28;
    EXPECT_EQ(m28.precisionCast<288>(), init28); EXPECT_EQ(m28.precisionCast<320>(), init28); EXPECT_EQ(m28.precisionCast<352>(), init28); EXPECT_EQ(m28.precisionCast<384>(), init28);
    EXPECT_EQ(m28.precisionCast<416>(), init28); EXPECT_EQ(m28.precisionCast<448>(), init28); EXPECT_EQ(m28.precisionCast<480>(), init28); EXPECT_EQ(m28.precisionCast<512>(), init28);
    EXPECT_EQ(m28.precisionCast<544>(), init28); EXPECT_EQ(m28.precisionCast<576>(), init28); EXPECT_EQ(m28.precisionCast<608>(), init28); EXPECT_EQ(m28.precisionCast<640>(), init28);
    EXPECT_EQ(m28.precisionCast<672>(), init28); EXPECT_EQ(m28.precisionCast<704>(), init28); EXPECT_EQ(m28.precisionCast<736>(), init28); EXPECT_EQ(m28.precisionCast<768>(), init28);
    EXPECT_EQ(m28.precisionCast<800>(), init28); EXPECT_EQ(m28.precisionCast<832>(), init28); EXPECT_EQ(m28.precisionCast<864>(), init28); EXPECT_EQ(m28.precisionCast<896>(), init28);


    long long init29 = 1220581244790115876; Aesi<256> m29 = init29;
    EXPECT_EQ(m29.precisionCast<288>(), init29); EXPECT_EQ(m29.precisionCast<320>(), init29); EXPECT_EQ(m29.precisionCast<352>(), init29); EXPECT_EQ(m29.precisionCast<384>(), init29);
    EXPECT_EQ(m29.precisionCast<416>(), init29); EXPECT_EQ(m29.precisionCast<448>(), init29); EXPECT_EQ(m29.precisionCast<480>(), init29); EXPECT_EQ(m29.precisionCast<512>(), init29);
    EXPECT_EQ(m29.precisionCast<544>(), init29); EXPECT_EQ(m29.precisionCast<576>(), init29); EXPECT_EQ(m29.precisionCast<608>(), init29); EXPECT_EQ(m29.precisionCast<640>(), init29);
    EXPECT_EQ(m29.precisionCast<672>(), init29); EXPECT_EQ(m29.precisionCast<704>(), init29); EXPECT_EQ(m29.precisionCast<736>(), init29); EXPECT_EQ(m29.precisionCast<768>(), init29);
    EXPECT_EQ(m29.precisionCast<800>(), init29); EXPECT_EQ(m29.precisionCast<832>(), init29); EXPECT_EQ(m29.precisionCast<864>(), init29); EXPECT_EQ(m29.precisionCast<896>(), init29);

    Aesi<448> l0 = "554241380131154142962417280417998404754929828012346097179687246704910303315350311367744175378443304128036758983824324734859979768793199.";
    EXPECT_EQ(l0.precisionCast<288>(), "408360527252105534419027813905394704348416525449517229626145092686873334890812834908271.");
    Aesi<576> l1 = "173149006578507408541558843749803097456182801623666290320265239338181642132588175569300761116459015677654013937947922994294373883893325690805618652779686048163576445320646483.";
    EXPECT_EQ(l1.precisionCast<128>(), "210443040993062611785785420324814540627.");
    Aesi<576> l2 = "66531456573720529442216757952933331047332848598711407403162907502778746277356003198417676048381460115126892609965612607567352416913344652080416627158294298245466839069775512.";
    EXPECT_EQ(l2.precisionCast<160>(), "1443229828299692348357792670919834022979176257176.");
    Aesi<384> l3 = "1098897099174270538478178379535802020226067483295913532002005072123857444388599717512365115154922424475266850503120.";
    EXPECT_EQ(l3.precisionCast<320>(), "1913199418936122088346837076981965348887692948209008066197015782466463820230886895375591805565392.");
    Aesi<512> l4 = "3069476797928086052610870903467024437789501028462434594919744663711449127505497769599003817092647117494652657649787519339440892062605145266124294683385280.";
    EXPECT_EQ(l4.precisionCast<288>(), "404126981474586070624526050760237057921124714973395197115667617772232279058850682265024.");
    Aesi<352> l5 = "2948016904402708200913240914482790524802448562129494737510300849124666954825609112501202095235581771983067.";
    EXPECT_EQ(l5.precisionCast<192>(), "1120442763073472009948111228858560283233140945287272409307.");
    Aesi<544> l6 = "10812628969630480520296504856105212223514943626905130339806055726389997460671652738109321632680730144583156832174036283780045422846847056560092948355830105461452561.";
    EXPECT_EQ(l6.precisionCast<128>(), "330342928733391374787712324367724277521.");
    Aesi<544> l7 = "4112600136041780850943347403651201772866193144792995503451651200434798923608930918174270859878921981299563448120913143826254282006786082095303150341646975970259785.";
    EXPECT_EQ(l7.precisionCast<288>(), "257304491601243764544329915240262128843475594535859368626521507218543067528530890011465.");
    Aesi<384> l8 = "28048835949300858284373720359733481364253247860667645736041649560388756406925265368724245525050388699493492127215396.";
    EXPECT_EQ(l8.precisionCast<256>(), "41878091393227431195219434526290235492405672892345774495477168807404321681188.");
    Aesi<640> l9 = "3848583613533234298953998803964435640019755235139553979562211441430877304606643444685829881598731682613890268054287149783980908651187506403654762887423297734191981318623098337988025363561114940.";
    EXPECT_EQ(l9.precisionCast<160>(), "541300877713023860093328693961789219540436446524.");
    Aesi<512> l10 = "8488462826013523989070679682331534259557436274695014765129319667636222109541019846966725069632607848463488852770772093964627525568328843684570072600088832.";
    EXPECT_EQ(l10.precisionCast<160>(), "523538838524052847229876918926034085392282358016.");
    Aesi<576> l11 = "11236969385546154746878679488297868602806916111627187642798923353498840149284473359875405738278850824908974037092662116610981773121895588572197349961886691945064651776229050.";
    EXPECT_EQ(l11.precisionCast<128>(), "29179789164106523943541955618292268730.");
    Aesi<480> l12 = "924082864945322453283057577897421592990774691137290429034141806501722486590884960785337810624006801980116699350354269354398363040536296529326681.";
    EXPECT_EQ(l12.precisionCast<256>(), "15790246827509817793328866413084671789409373095293932456575888129892184748633.");
    Aesi<512> l13 = "9607535504179656540242447115527397529604623905521687360978734063119313561780639573012578165648934865049773849480530263572594554247909207343890208529325178.";
    EXPECT_EQ(l13.precisionCast<96>(), "43716316362024562473758891130.");
    Aesi<352> l14 = "504397667162110386921588331778312416154169943905827797809093816111203720055808585599403524127547394508185.";
    EXPECT_EQ(l14.precisionCast<320>(), "366270022681278488140630850294606110994384776947473041868263123438838372271196545770306825766297.");
    Aesi<544> l15 = "27608764507031531319380382106462863564782632077695501707054751614193288192180423340509534352068491112085645377731761780973224450169988873068318780343686563718281728.";
    EXPECT_EQ(l15.precisionCast<128>(), "299026899257808842617251361117814438400.");
    Aesi<352> l16 = "1978129624964056450214026876059300487877996643140932950061364608660420356925042564405231900323677600676732.";
    EXPECT_EQ(l16.precisionCast<320>(), "365332643823514689801663993143371567677009732484894048068501453581388238598858847283260927441788.");
    Aesi<576> l17 = "195265355728943716865923619155375969530111891393015085388776834717930166073822658430177349287072569812662023940174467801397134386843815407817766067594355521616265929622366898.";
    EXPECT_EQ(l17.precisionCast<160>(), "221431119546753700832094394260270558954283936434.");
    Aesi<448> l18 = "414657978233561248876686250413054606900177295904410399578041220021615424324810017863916222257119299661647319122459751028412186388057654.";
    EXPECT_EQ(l18.precisionCast<160>(), "287129154900729631574915186396319442248636812854.");
    Aesi<416> l19 = "56687510612260333551549640577823250078355837097233108084669493198465659199087411861462621495608437125573216962137258061197924.";
    EXPECT_EQ(l19.precisionCast<192>(), "3671611332121323818239920054049877596259686932276771811940.");
    Aesi<640> l20 = "1546223449542899971558886608789690118697134740086024592798235620930020323290348765670723997823619247765209481873838363731256837657816604584457400460176692587426042570895554473755247436115230189.";
    EXPECT_EQ(l20.precisionCast<320>(), "1705058499460706178048346990812902342294295809567394699482134087317553182108908996873089872575981.");
    Aesi<544> l21 = "174065680196633578590894771435895688468079098486287540664921962829963159709119456409138894766839898574579608494679948491219544731370758383681896937747086278361071.";
    EXPECT_EQ(l21.precisionCast<160>(), "220794710368648048438000780675654519053309411311.");
    Aesi<448> l22 = "136337341513559528961653942676228706381771186833849367631461720301064785651836843206251270469997623786977461200400063656318366284038665.";
    EXPECT_EQ(l22.precisionCast<224>(), "816945909177209805889026909383334983853515920833335341477807047177.");
    Aesi<352> l23 = "5882616107094423670384170997721509360066167036113934904312412848395219509023523677500755059444938780072409.";
    EXPECT_EQ(l23.precisionCast<192>(), "2482927072274059692659564052439247617034525043020273441241.");
    Aesi<576> l24 = "49857691326617957339080952718292050690932576020199908481126156219200801064319573633766216649464944745473827870799580369859357610319478396095070633417622983092014069297900742.";
    EXPECT_EQ(l24.precisionCast<96>(), "20676605872175966156536081606.");
    Aesi<384> l25 = "28597517267806223597843711502305823530466336309097073757475161768363138442370308787956725565043616828782360287501138.";
    EXPECT_EQ(l25.precisionCast<192>(), "2349989946988515365971908154524797542461532293278087240530.");
    Aesi<352> l26 = "5359709770582948688991943818180669926832250789635927200530270394005210447928718134456374737387522683948933.";
    EXPECT_EQ(l26.precisionCast<96>(), "30780870628112076616018546565.");
    Aesi<608> l27 = "800374186897442994345991532287383427669343624463618411247954314190792985454915883373717659254388118542800958906763592267710368929085300529376043722066941548681087911311229845689540460.";
    EXPECT_EQ(l27.precisionCast<128>(), "2854867648611582803837292073272156012.");
    Aesi<384> l28 = "33405800918446002208040817593884350731087014451427547055891417316377644807769010652297688512536252488106030729965818.";
    EXPECT_EQ(l28.precisionCast<224>(), "459273051701248468242064512442751256897801283813240021045830922490.");
    Aesi<448> l29 = "545136889659423149046640496483142589267039543174772529930583944491083582156213563276447863450923726927200116011820859957964687342869311.";
    EXPECT_EQ(l29.precisionCast<224>(), "11564578527812652389272649650310409698199102565460528063263161490239.");
}