#include <gtest/gtest.h>
#include "../../Aesi.h"
#include "../benchmarks/benchmarks.h"

TEST(Division, SmallPositive) {
    Aesi512 small1 = -8492, small2 = 4243, small3 = -678, small4 = 2323;
    EXPECT_EQ(small1 / small3, 12);
    EXPECT_EQ(small3 / small1, 0);
    EXPECT_EQ(small2 / small4, 1);
    EXPECT_EQ(small4 / small2, 0);
}

TEST(Division, SmallNegative) {
    Aesi512 small1 = -8492, small2 = 4243, small3 = -678, small4 = 2323;
    EXPECT_EQ(small2 / small3, -6);
    EXPECT_EQ(small3 / small2, 0);
    EXPECT_EQ(small1 / small4, -3);
    EXPECT_EQ(small4 / small1, 0);
}

TEST(Division, MixedDivision) {
    Aesi512 m0 = -67602113121365, m1 = -70814915;
    EXPECT_EQ(m0 / m1, 954631);
    Aesi512 m2 = 15113324630360, m3 = 57523730;
    EXPECT_EQ(m2 / m3, 262732);
    Aesi512 m4 = 34122183947714, m5 = -44752342;
    EXPECT_EQ(m4 / m5, -762467);
    Aesi512 m6 = 2220619718304, m7 = -6807624;
    EXPECT_EQ(m6 / m7, -326196);
    Aesi512 m8 = -56544544247220, m9 = -64028010;
    EXPECT_EQ(m8 / m9, 883122);
    Aesi512 m10 = 27311885651556, m11 = 40063554;
    EXPECT_EQ(m10 / m11, 681714);
    Aesi512 m12 = 65477763597661, m13 = -85123723;
    EXPECT_EQ(m12 / m13, -769207);
    Aesi512 m14 = -40861948934173, m15 = -56440949;
    EXPECT_EQ(m14 / m15, 723977);
    Aesi512 m16 = 176062774875, m17 = -14641395;
    EXPECT_EQ(m16 / m17, -12025);
    Aesi512 m18 = -329952273599, m19 = -3832777;
    EXPECT_EQ(m18 / m19, 86087);
    Aesi512 m20 = -1088200002846, m21 = -1883706;
    EXPECT_EQ(m20 / m21, 577691);
    Aesi512 m22 = 6974639128232, m23 = -10309004;
    EXPECT_EQ(m22 / m23, -676558);
    Aesi512 m24 = 31834533607632, m25 = -55868976;
    EXPECT_EQ(m24 / m25, -569807);
    Aesi512 m26 = 31126617188568, m27 = -54848568;
    EXPECT_EQ(m26 / m27, -567501);
    Aesi512 m28 = 6290940249900, m29 = 21494628;
    EXPECT_EQ(m28 / m29, 292675);
    Aesi512 m30 = 42187650007176, m31 = -78019518;
    EXPECT_EQ(m30 / m31, -540732);
    Aesi512 m32 = -20062099158147, m33 = -64832101;
    EXPECT_EQ(m32 / m33, 309447);
    Aesi512 m34 = -9216212918125, m35 = 26711725;
    EXPECT_EQ(m34 / m35, -345025);
    Aesi512 m36 = -5364545160432, m37 = -31267749;
    EXPECT_EQ(m36 / m37, 171568);
    Aesi512 m38 = -19899125279059, m39 = 26608409;
    EXPECT_EQ(m38 / m39, -747851);
    Aesi512 m40 = -5126477678000, m41 = -7908790;
    EXPECT_EQ(m40 / m41, 648200);
    Aesi512 m42 = -54479717879628, m43 = -56742436;
    EXPECT_EQ(m42 / m43, 960123);
    Aesi512 m44 = -11035329100380, m45 = 85511380;
    EXPECT_EQ(m44 / m45, -129051);
    Aesi512 m46 = -77240330776750, m47 = 77342810;
    EXPECT_EQ(m46 / m47, -998675);
    Aesi512 m48 = -25940256135750, m49 = 32187375;
    EXPECT_EQ(m48 / m49, -805914);
    Aesi512 m50 = 68475350540736, m51 = -72959904;
    EXPECT_EQ(m50 / m51, -938534);
    Aesi512 m52 = -20926969579065, m53 = -77395215;
    EXPECT_EQ(m52 / m53, 270391);
    Aesi512 m54 = -1875340132380, m55 = -17179580;
    EXPECT_EQ(m54 / m55, 109161);
    Aesi512 m56 = -287061254477, m57 = -22981447;
    EXPECT_EQ(m56 / m57, 12491);
    Aesi512 m58 = -22187933279568, m59 = -49281768;
    EXPECT_EQ(m58 / m59, 450226);
    Aesi512 m60 = -979862770236, m61 = 2721327;
    EXPECT_EQ(m60 / m61, -360068);
    Aesi512 m62 = -1404488473664, m63 = -39927464;
    EXPECT_EQ(m62 / m63, 35176);
    Aesi512 m64 = -1465619419940, m65 = 26127452;
    EXPECT_EQ(m64 / m65, -56095);
    Aesi512 m66 = 49073567920942, m67 = 77753978;
    EXPECT_EQ(m66 / m67, 631139);
    Aesi512 m68 = 8897687795196, m69 = -49216686;
    EXPECT_EQ(m68 / m69, -180786);
    Aesi512 m70 = 6085886956392, m71 = -16070088;
    EXPECT_EQ(m70 / m71, -378709);
    Aesi512 m72 = 50224482468354, m73 = 75335443;
    EXPECT_EQ(m72 / m73, 666678);
    Aesi512 m74 = 7740677726208, m75 = 15847496;
    EXPECT_EQ(m74 / m75, 488448);
    Aesi512 m76 = 1063246222290, m77 = 8595570;
    EXPECT_EQ(m76 / m77, 123697);
    Aesi512 m78 = -10332555271144, m79 = 29755036;
    EXPECT_EQ(m78 / m79, -347254);
}

TEST(Division, MixedDivisionAssignment) {
    Aesi512 m0 = -3174698500920, m1 = 37446314;
    m0 /= m1; EXPECT_EQ(m0, -84780);
    Aesi512 m2 = 156000466440, m3 = -66666866;
    m2 /= m3; EXPECT_EQ(m2, -2340);
    Aesi512 m4 = -9664086627036, m5 = -51407724;
    m4 /= m5; EXPECT_EQ(m4, 187989);
    Aesi512 m6 = 683516898168, m7 = 1805609;
    m6 /= m7; EXPECT_EQ(m6, 378552);
    Aesi512 m8 = -24751231123115, m9 = -89301431;
    m8 /= m9; EXPECT_EQ(m8, 277165);
    Aesi512 m10 = 26354396493600, m11 = -49311248;
    m10 /= m11; EXPECT_EQ(m10, -534450);
    Aesi512 m12 = -1203507178300, m13 = 67055225;
    m12 /= m13; EXPECT_EQ(m12, -17948);
    Aesi512 m14 = -9860315571000, m15 = -32244750;
    m14 /= m15; EXPECT_EQ(m14, 305796);
    Aesi512 m16 = -13210718718360, m17 = 32872297;
    m16 /= m17; EXPECT_EQ(m16, -401880);
    Aesi512 m18 = -1166273423250, m19 = 5292222;
    m18 /= m19; EXPECT_EQ(m18, -220375);
    Aesi512 m20 = -17154168020562, m21 = -32960958;
    m20 /= m21; EXPECT_EQ(m20, 520439);
    Aesi512 m22 = -24461813157015, m23 = -33125755;
    m22 /= m23; EXPECT_EQ(m22, 738453);
    Aesi512 m24 = -3915356462184, m25 = -67492182;
    m24 /= m25; EXPECT_EQ(m24, 58012);
    Aesi512 m26 = 6239712095136, m27 = 46955376;
    m26 /= m27; EXPECT_EQ(m26, 132886);
    Aesi512 m28 = -41815750742532, m29 = 55578484;
    m28 /= m29; EXPECT_EQ(m28, -752373);
    Aesi512 m30 = -58740024955500, m31 = -66670100;
    m30 /= m31; EXPECT_EQ(m30, 881055);
    Aesi512 m32 = 25823556551045, m33 = 37417039;
    m32 /= m33; EXPECT_EQ(m32, 690155);
    Aesi512 m34 = 5480709811479, m35 = -54757269;
    m34 /= m35; EXPECT_EQ(m34, -100091);
    Aesi512 m36 = -22711656353225, m37 = -79507295;
    m36 /= m37; EXPECT_EQ(m36, 285655);
    Aesi512 m38 = 12409303686885, m39 = 17897733;
    m38 /= m39; EXPECT_EQ(m38, 693345);
    Aesi512 m40 = 13872981907785, m41 = -36991955;
    m40 /= m41; EXPECT_EQ(m40, -375027);
    Aesi512 m42 = -643299920236, m43 = 67038341;
    m42 /= m43; EXPECT_EQ(m42, -9596);
    Aesi512 m44 = 3282278192935, m45 = 4741265;
    m44 /= m45; EXPECT_EQ(m44, 692279);
    Aesi512 m46 = -13386640546113, m47 = -57296749;
    m46 /= m47; EXPECT_EQ(m46, 233637);
    Aesi512 m48 = -368301167920, m49 = 12409069;
    m48 /= m49; EXPECT_EQ(m48, -29680);
    Aesi512 m50 = 4999702954272, m51 = -74294207;
    m50 /= m51; EXPECT_EQ(m50, -67296);
    Aesi512 m52 = -13934020168638, m53 = -18893766;
    m52 /= m53; EXPECT_EQ(m52, 737493);
    Aesi512 m54 = 18163193044176, m55 = 18950076;
    m54 /= m55; EXPECT_EQ(m54, 958476);
    Aesi512 m56 = -80852334667440, m57 = 82227173;
    m56 /= m57; EXPECT_EQ(m56, -983280);
    Aesi512 m58 = -13961168504878, m59 = -14066626;
    m58 /= m59; EXPECT_EQ(m58, 992503);
    Aesi512 m60 = 35163604583850, m61 = 77817450;
    m60 /= m61; EXPECT_EQ(m60, 451873);
    Aesi512 m62 = -71110375898000, m63 = 74848300;
    m62 /= m63; EXPECT_EQ(m62, -950060);
    Aesi512 m64 = -15288429937525, m65 = 45382825;
    m64 /= m65; EXPECT_EQ(m64, -336877);
    Aesi512 m66 = -34335775193105, m67 = 56421811;
    m66 /= m67; EXPECT_EQ(m66, -608555);
    Aesi512 m68 = -20503984546510, m69 = -22501931;
    m68 /= m69; EXPECT_EQ(m68, 911210);
    Aesi512 m70 = -37499367266592, m71 = 43862808;
    m70 /= m71; EXPECT_EQ(m70, -854924);
    Aesi512 m72 = 44304906945178, m73 = -47076238;
    m72 /= m73; EXPECT_EQ(m72, -941131);
    Aesi512 m74 = -39150837269096, m75 = 90134744;
    m74 /= m75; EXPECT_EQ(m74, -434359);
    Aesi512 m76 = -22502676678930, m77 = -89545785;
    m76 /= m77; EXPECT_EQ(m76, 251298);
    Aesi512 m78 = -34136928128166, m79 = -79394298;
    m78 /= m79; EXPECT_EQ(m78, 429967);
}

TEST(Division, DifferentPrecision) {
    {
        Aesi < 256 > first = 11921542526290; // Aesi<8 blocks>
        Aesi < 352 > second = 88322770; // Aesi<11 blocks>
        EXPECT_EQ(first / second, 134977);

        Aesi < 320 > third = -4102459315740; // Aesi<10 blocks>
        Aesi < 448 > forth = 57049914; // Aesi<14 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Aesi < 320 > first = 13488766159653; // Aesi<10 blocks>
        Aesi < 352 > second = 48109389; // Aesi<11 blocks>
        EXPECT_EQ(first / second, 280377);

        Aesi < 320 > third = -991821198108; // Aesi<10 blocks>
        Aesi < 416 > forth = 4233378; // Aesi<13 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Aesi < 480 > first = -73311512255046; // Aesi<15 blocks>
        Aesi < 384 > second = 86942434; // Aesi<12 blocks>
        EXPECT_EQ(first / second, -843219);

        Aesi < 224 > third = 721422850272; // Aesi<7 blocks>
        Aesi < 416 > forth = 15138768; // Aesi<13 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Aesi < 256 > first = 57155718149216; // Aesi<8 blocks>
        Aesi < 256 > second = 80590001; // Aesi<8 blocks>
        EXPECT_EQ(first / second, 709216);

        Aesi < 192 > third = 657315248892; // Aesi<6 blocks>
        Aesi < 448 > forth = 4018458; // Aesi<14 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Aesi < 224 > first = -24101170189740; // Aesi<7 blocks>
        Aesi < 224 > second = -64694181; // Aesi<7 blocks>
        EXPECT_EQ(first / second, 372540);

        Aesi < 288 > third = 27530629582832; // Aesi<9 blocks>
        Aesi < 384 > forth = 49146568; // Aesi<12 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Aesi < 448 > first = 10440263901354; // Aesi<14 blocks>
        Aesi < 352 > second = -25782631; // Aesi<11 blocks>
        EXPECT_EQ(first / second, -404934);

        Aesi < 192 > third = 49691966023092; // Aesi<6 blocks>
        Aesi < 416 > forth = -57556284; // Aesi<13 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Aesi < 416 > first = -19076120667460; // Aesi<13 blocks>
        Aesi < 384 > second = 43657645; // Aesi<12 blocks>
        EXPECT_EQ(first / second, -436948);

        Aesi < 224 > third = 3594292228818; // Aesi<7 blocks>
        Aesi < 384 > forth = 8218869; // Aesi<12 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Aesi < 192 > first = -80928002017452; // Aesi<6 blocks>
        Aesi < 480 > second = -83298854; // Aesi<15 blocks>
        EXPECT_EQ(first / second, 971538);

        Aesi < 256 > third = 17240509748020; // Aesi<8 blocks>
        Aesi < 352 > forth = 35593090; // Aesi<11 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Aesi < 192 > first = -34515574677948; // Aesi<6 blocks>
        Aesi < 224 > second = 72218589; // Aesi<7 blocks>
        EXPECT_EQ(first / second, -477932);

        Aesi < 320 > third = -14100957360246; // Aesi<10 blocks>
        Aesi < 448 > forth = -35406678; // Aesi<14 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
    {
        Aesi < 480 > first = -1093297575920; // Aesi<15 blocks>
        Aesi < 448 > second = -22640248; // Aesi<14 blocks>
        EXPECT_EQ(first / second, 48290);

        Aesi < 320 > third = 6343088533611; // Aesi<10 blocks>
        Aesi < 352 > forth = 8322199; // Aesi<11 blocks>
        forth /= third; EXPECT_EQ(forth, 0);
    }
}

TEST(Division, Huge) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi512 o0 = "2713006190353794843001574597263830299754718406554841786815057226557898480996304062155538953890163418239849100307320134528405798210465559594879025314495448.";
    Aesi512 o1 = "11661561043881735325505841521207566651613627725292988229211140876317241794187341201621018221552442573993825724096572.";
    EXPECT_EQ(o0 / o1, "232645199055677005273199249378767621034."); o0 /= o1; EXPECT_EQ(o0, "232645199055677005273199249378767621034.");

    Aesi512 o2 = "691657383591844216575014611357784034870295742903613141672512174103876528903985094696620754198093979498076876293271199527919688738035345736374122646767573.";
    Aesi512 o3 = "2409602862744372806307702972540060975260544871590287474520202002693463055313506708820654306720331381850013047928613.";
    EXPECT_EQ(o2 / o3, "287042065846524508523959704981051655921."); o2 /= o3; EXPECT_EQ(o2, "287042065846524508523959704981051655921.");

    Aesi512 o4 = "862961919221122926503864097354238936894245414515494708267585978566644549468173258415739298482605508015607858266296250399665996022362011764438685647463830.";
    Aesi512 o5 = "5873557982994106495137882685361231858689698157978917893282787595467180282270338468781173472015561545927736172685733.";
    EXPECT_EQ(o4 / o5, "146923197441769226692308450902282780510."); o4 /= o5; EXPECT_EQ(o4, "146923197441769226692308450902282780510.");

    Aesi512 o6 = "909751349496540626286980189308431084499585627279790605720322255669065341735403163846977171861141445150947510387797653148247515123747860052126284330722202.";
    Aesi512 o7 = "15561991402458290936100771430419188589420359597137447746379656966063460915846460963239225132324610295230390971530646.";
    EXPECT_EQ(o6 / o7, "58459828563639314156347669206282086087."); o6 /= o7; EXPECT_EQ(o6, "58459828563639314156347669206282086087.");

    Aesi512 o8 = "25421649421044648256410363130202797785188994860967155201171938294895129536230522888280666734235813067662201778838420766660088549887626676430980696267314.";
    Aesi512 o9 = "3635380412137022286721052520386439723863526171105557476789276051209134573219745677378561810465517973946503152426611.";
    EXPECT_EQ(o8 / o9, "6992844362634606375592076092630167174."); o8 /= o9; EXPECT_EQ(o8, "6992844362634606375592076092630167174.");

    Aesi512 o10 = "4361619760408996497482816148465856683316591740863630138805361794028625200167831013334690318241884915315786020088677636841530018945682288958019700858180466.";
    Aesi512 o11 = "17946983294462140367128935734938575269994784724208162584309403592330215378458801492365395532452117805645008857836674.";
    EXPECT_EQ(o10 / o11, "243028016956746794445631914777091679609."); o10 /= o11; EXPECT_EQ(o10, "243028016956746794445631914777091679609.");

    Aesi512 o12 = "726172394863842026643532276912271267238672837581228323321295912865133609218915008946525867688511388382156657375838968785133147950886472758607057067639120.";
    Aesi512 o13 = "6963643245505486388144439615168604074915851162431054977755035976588546396925217547401715343982194118163867607903024.";
    EXPECT_EQ(o12 / o13, "104280528060154758712170321968763807255."); o12 /= o13; EXPECT_EQ(o12, "104280528060154758712170321968763807255.");

    Aesi512 o14 = "32637388470827471839204303046929852595530011254668835976250067681125216485847721748866065534498675747447674611564141358897997618547017563533021511035442.";
    Aesi512 o15 = "8495800752323116811536861037424141384042500227031207807221799280770094104642494244733378746232366201651484118711662.";
    EXPECT_EQ(o14 / o15, "3841590618977617629417109808541334191."); o14 /= o15; EXPECT_EQ(o14, "3841590618977617629417109808541334191.");

    Aesi512 o16 = "403514851571224992074670700527580311336780227973409735836910807900593428188675071364886201637165760162827231591462548327739712932076634497379216374526440.";
    Aesi512 o17 = "6733396156826693256195932070648508785650780701244861235797832783776481641285701279450526328066107183446532511729671.";
    EXPECT_EQ(o16 / o17, "59927389117320698400617346497363047640."); o16 /= o17; EXPECT_EQ(o16, "59927389117320698400617346497363047640.");

    Aesi512 o18 = "703085916010877168634991995210897878816519373048746252351843657754538695917849101366670841844148739057835894511089554007299580165956308796215533574899140.";
    Aesi512 o19 = "3996800316069871989396775978196722373576793944323962127524656845306336180885434055130830940605076849934807431146085.";
    EXPECT_EQ(o18 / o19, "175912194858469840458518768984666656884."); o18 /= o19; EXPECT_EQ(o18, "175912194858469840458518768984666656884.");

    Aesi512 o20 = "1392867926917594562735059626659923877830375057197975395188418567307349366664595102034516267516986764104663372211421060509069899981135348323082900543783280.";
    Aesi512 o21 = "15140392216976894545328794445234489480411523723344802715233457965502369948861382164443999932329525612133450561181640.";
    EXPECT_EQ(o20 / o21, "91996819300082217353120817415886665502."); o20 /= o21; EXPECT_EQ(o20, "91996819300082217353120817415886665502.");

    Aesi512 o22 = "564664911237330892616831807680330160443386923150487476298016715746609924327673916961717376746712620215121947484077945564140076272773087875101286543329616.";
    Aesi512 o23 = "4427829722673921894941501123797908019887127300578370497420583880233872474221991949803215803100919269031040076018448.";
    EXPECT_EQ(o22 / o23, "127526338320059748007000295678516004517."); o22 /= o23; EXPECT_EQ(o22, "127526338320059748007000295678516004517.");

    Aesi512 o24 = "2006539653770390597793031370613294726129181121914524896685945102724092816717630062966580381936170851502671619596802895674775645206455524594180074181318290.";
    Aesi512 o25 = "8753014661316809094787261020053253628573219286495039075648090426261494590517100527929867082773655162745489615312947.";
    EXPECT_EQ(o24 / o25, "229239836948762233149436135193819596070."); o24 /= o25; EXPECT_EQ(o24, "229239836948762233149436135193819596070.");

    Aesi512 o26 = "1131754130677725278471408653546150324551716089888751281301023211466577625215152808590539680048962256290057307680274655381055747926576436370522930724973990.";
    Aesi512 o27 = "7532806378389133345888841679648860985161470838383015575538102816646215263409191847447160680286644424674430036796270.";
    EXPECT_EQ(o26 / o27, "150243358693595851456437409905437725637."); o26 /= o27; EXPECT_EQ(o26, "150243358693595851456437409905437725637.");

    Aesi512 o28 = "392045816913861384110182837670146014418158786910171049652511119443803771652897458446761792569998529744097588316928192249202690178014492498093321730695252.";
    Aesi512 o29 = "7289150240379164352847466484948682652289378198225792835750634115753219010828372615066490117130487092093411611987078.";
    EXPECT_EQ(o28 / o29, "53784845144510025644355972800071310734."); o28 /= o29; EXPECT_EQ(o28, "53784845144510025644355972800071310734.");

    Aesi512 o30 = "649151807305208322819075147378512799332754074950549894308035215843567142251955013904248736157323651065409018116947670181945408611850384784226346936909125.";
    Aesi512 o31 = "14114873659930250008296013710596176333852853505711578625187785033492072690322254492516381480313827691754941847973375.";
    EXPECT_EQ(o30 / o31, "45990621166382878432642770774474120379."); o30 /= o31; EXPECT_EQ(o30, "45990621166382878432642770774474120379.");

    Aesi512 o32 = "491513195360149037141858565175280208443700729028629069139985697787173466595101428221812992328592164336076448530404068636684386866516329284657504623321344.";
    Aesi512 o33 = "2554046695314546944451784981669847873566566722812710372548737683298595173544983436313101217242797437402567900406728.";
    EXPECT_EQ(o32 / o33, "192444874348554574856789385679904322848."); o32 /= o33; EXPECT_EQ(o32, "192444874348554574856789385679904322848.");

    Aesi512 o34 = "1750111615128054584071209571169512721812101352190946637938110753081617382339234788467719485061656096907148315690036457646746677201438110604897942230502345.";
    Aesi512 o35 = "17583576419554686978419858424675588345342539289060158672542125590077526480096964009077379694043904132946027279265861.";
    EXPECT_EQ(o34 / o35, "99531038132933883049213867009174302645."); o34 /= o35; EXPECT_EQ(o34, "99531038132933883049213867009174302645.");

    Aesi512 o36 = "975058314705345060304615149116899781753795322425846036902427477482509651452222506388525256460347290469421379642225502384220625878009064793990547328185220.";
    Aesi512 o37 = "4709711523694740054104021806004566195987773630282623336315586126128147683851911129505467994549060052606350281875270.";
    EXPECT_EQ(o36 / o37, "207031430651280684176356796387804025686."); o36 /= o37; EXPECT_EQ(o36, "207031430651280684176356796387804025686.");

    Aesi512 o38 = "1424332986368404409944310221477738117356487171558651898706383584604542338320396149762082436191242347159948952981047537355884297272057017655564953647383312.";
    Aesi512 o39 = "5398100402588307676844060457667058824456130242156306208509550786794035241114781897022580844578292097188483328808084.";
    EXPECT_EQ(o38 / o39, "263858187166259123024781973845635300468."); o38 /= o39; EXPECT_EQ(o38, "263858187166259123024781973845635300468.");

#ifdef NDEBUG
    Logging::addRecord("Division",
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}