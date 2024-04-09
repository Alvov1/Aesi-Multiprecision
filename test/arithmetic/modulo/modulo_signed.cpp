#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../../Aesi-Multiprecision.h"
#include "../../benchmarks/benchmarks.h"

TEST(Modulo, Signed_MixedModulo) {
    Aesi512 m0 = -4825285950739, m1 = -26462400;
    EXPECT_EQ(m0 % m1, -26085139);
    Aesi512 m2 = -15187820465943, m3 = -73508543;
    EXPECT_EQ(m2 % m3, -73379627);
    Aesi512 m4 = 29744669116783, m5 = -59738688;
    EXPECT_EQ(m4 % m5, 59497327);
    Aesi512 m6 = 1445481773735, m7 = -10943575;
    EXPECT_EQ(m6 % m7, 10613435);
    Aesi512 m8 = -9608444938718, m9 = -14443665;
    EXPECT_EQ(m8 % m9, -13452443);
    Aesi512 m10 = -1827101039843, m11 = -5476397;
    EXPECT_EQ(m10 % m11, -5232336);
    Aesi512 m12 = -8555190917773, m13 = 26131496;
    EXPECT_EQ(m12 % m13, -442333);
    Aesi512 m14 = 3835023546766, m15 = -66353339;
    EXPECT_EQ(m14 % m15, 65965922);
    Aesi512 m16 = 4152238729728, m17 = 70516764;
    EXPECT_EQ(m16 % m17, 115116);
    Aesi512 m18 = 21152437469773, m19 = 27580050;
    EXPECT_EQ(m18 % m19, 862423);
    Aesi512 m20 = -16788008144183, m21 = 74026418;
    EXPECT_EQ(m20 % m21, -964471);
    Aesi512 m22 = -28451749301697, m23 = 53230588;
    EXPECT_EQ(m22 % m23, -15697);
    Aesi512 m24 = -3270181161316, m25 = -8281687;
    EXPECT_EQ(m24 % m25, -7979000);
    Aesi512 m26 = 49633340100976, m27 = 66209125;
    EXPECT_EQ(m26 % m27, 590351);
    Aesi512 m28 = -46340911398101, m29 = -51209841;
    EXPECT_EQ(m28 % m29, -50870540);
    Aesi512 m30 = 185622644633, m31 = 9521052;
    EXPECT_EQ(m30 % m31, 214841);
    Aesi512 m32 = -10811475069954, m33 = 59784751;
    EXPECT_EQ(m32 % m33, -699114);
    Aesi512 m34 = 12380907605205, m35 = 15918898;
    EXPECT_EQ(m34 % m35, 604603);
    Aesi512 m36 = 1257249026383, m37 = -61467173;
    EXPECT_EQ(m36 % m37, 60937014);
    Aesi512 m38 = 61847752758018, m39 = -72413117;
    EXPECT_EQ(m38 % m39, 71593903);
    Aesi512 m40 = -37469111279132, m41 = 74062214;
    EXPECT_EQ(m40 % m41, -345536);
    Aesi512 m42 = -31033946342900, m43 = 31210566;
    EXPECT_EQ(m42 % m43, -935894);
    Aesi512 m44 = 262286204362, m45 = 1046069;
    EXPECT_EQ(m44 % m45, 93647);
    Aesi512 m46 = -38784099239147, m47 = -66593692;
    EXPECT_EQ(m46 % m47, -66205731);
    Aesi512 m48 = -49611834841836, m49 = 50734228;
    EXPECT_EQ(m48 % m49, -167880);
    Aesi512 m50 = -5264331698408, m51 = -13698889;
    EXPECT_EQ(m50 % m51, -13042376);
    Aesi512 m52 = 8207558345771, m53 = 29861083;
    EXPECT_EQ(m52 % m53, 794557);
    Aesi512 m54 = 13581825744423, m55 = 71047285;
    EXPECT_EQ(m54 % m55, 460113);
    Aesi512 m56 = -10767453796459, m57 = 85680382;
    EXPECT_EQ(m56 % m57, -190519);
    Aesi512 m58 = -7637085665123, m59 = 12026053;
    EXPECT_EQ(m58 % m59, -837738);
    Aesi512 m60 = 11626011187006, m61 = 13548218;
    EXPECT_EQ(m60 % m61, 808628);
    Aesi512 m62 = -23116846973385, m63 = 70653532;
    EXPECT_EQ(m62 % m63, -452433);
    Aesi512 m64 = -10189597866429, m65 = 39212175;
    EXPECT_EQ(m64 % m65, -495279);
    Aesi512 m66 = 35873994210322, m67 = 66136687;
    EXPECT_EQ(m66 % m67, 174408);
    Aesi512 m68 = 4535653103788, m69 = 13603381;
    EXPECT_EQ(m68 % m69, 207387);
    Aesi512 m70 = 9262614882707, m71 = -68228371;
    EXPECT_EQ(m70 % m71, 67692489);
    Aesi512 m72 = -25362332495829, m73 = 31704461;
    EXPECT_EQ(m72 % m73, -169808);
    Aesi512 m74 = 46943441312445, m75 = -56547122;
    EXPECT_EQ(m74 % m75, 56324437);
    Aesi512 m76 = 6397582511119, m77 = 12738858;
    EXPECT_EQ(m76 % m77, 634939);
    Aesi512 m78 = -10713134329789, m79 = -20339953;
    EXPECT_EQ(m78 % m79, -20064830);
}

TEST(Modulo, Signed_MixedModuloAssignment) {
    Aesi512 m0 = 347971189167, m1 = -10468158;
    m0 %= m1; EXPECT_EQ(m0, 9617247);
    Aesi512 m2 = 82299524104295, m3 = -85161108;
    m2 %= m3; EXPECT_EQ(m2, 84816419);
    Aesi512 m4 = 15733272660162, m5 = 58445417;
    m4 %= m5; EXPECT_EQ(m4, 185430);
    Aesi512 m6 = 21441802118831, m7 = 32445743;
    m6 %= m7; EXPECT_EQ(m6, 411538);
    Aesi512 m8 = -225797197700, m9 = -791578;
    m8 %= m9; EXPECT_EQ(m8, -364778);
    Aesi512 m10 = 37158981415510, m11 = -85944342;
    m10 %= m11; EXPECT_EQ(m10, 85708390);
    Aesi512 m12 = -84978103484, m13 = -1337355;
    m12 %= m13; EXPECT_EQ(m12, -1229429);
    Aesi512 m14 = -36479221879833, m15 = -57772210;
    m14 %= m15; EXPECT_EQ(m14, -57547323);
    Aesi512 m16 = 19799281452078, m17 = -88989537;
    m16 %= m17; EXPECT_EQ(m16, 88354485);
    Aesi512 m18 = 62544905852708, m19 = 88569191;
    m18 %= m19; EXPECT_EQ(m18, 244238);
    Aesi512 m20 = 10563516105678, m21 = 33470367;
    m20 %= m21; EXPECT_EQ(m20, 517542);
    Aesi512 m22 = 14792471248628, m23 = 16633406;
    m22 %= m23; EXPECT_EQ(m22, 724490);
    Aesi512 m24 = -4474855045896, m25 = 30121937;
    m24 %= m25; EXPECT_EQ(m24, -329050);
    Aesi512 m26 = -67695068629974, m27 = 68710890;
    m26 %= m27; EXPECT_EQ(m26, -427734);
    Aesi512 m28 = -26367812003697, m29 = 89576749;
    m28 %= m29; EXPECT_EQ(m28, -168057);
    Aesi512 m30 = -73884520718086, m31 = 78543985;
    m30 %= m31; EXPECT_EQ(m30, -540241);
    Aesi512 m32 = 27121633606354, m33 = 33856377;
    m32 %= m33; EXPECT_EQ(m32, 975571);
    Aesi512 m34 = 7643784867640, m35 = 8394625;
    m34 %= m35; EXPECT_EQ(m34, 311515);
    Aesi512 m36 = 20823365064094, m37 = -26717860;
    m36 %= m37; EXPECT_EQ(m36, 26055154);
    Aesi512 m38 = 32317724233551, m39 = -42171193;
    m38 %= m39; EXPECT_EQ(m38, 41333966);
    Aesi512 m40 = -79175100071807, m41 = 86660675;
    m40 %= m41; EXPECT_EQ(m40, -856957);
    Aesi512 m42 = 49371377679669, m43 = -72600911;
    m42 %= m43; EXPECT_EQ(m42, 71965962);
    Aesi512 m44 = 4496522629453, m45 = -38748432;
    m44 %= m45; EXPECT_EQ(m44, 38334877);
    Aesi512 m46 = -56997212920295, m47 = 88560132;
    m46 %= m47; EXPECT_EQ(m46, -525227);
    Aesi512 m48 = 21973208639078, m49 = 86842729;
    m48 %= m49; EXPECT_EQ(m48, 819311);
    Aesi512 m50 = 14554092528935, m51 = 25879784;
    m50 %= m51; EXPECT_EQ(m50, 761503);
    Aesi512 m52 = 37798099783123, m53 = 48852995;
    m52 %= m53; EXPECT_EQ(m52, 168678);
    Aesi512 m54 = 57333872564438, m55 = -59196307;
    m54 %= m55; EXPECT_EQ(m54, 58971579);
    Aesi512 m56 = 71400434364443, m57 = 89222326;
    m56 %= m57; EXPECT_EQ(m56, 315965);
    Aesi512 m58 = -15529620519738, m59 = 70288538;
    m58 %= m59; EXPECT_EQ(m58, -645480);
    Aesi512 m60 = 19311829893256, m61 = -36260355;
    m60 %= m61; EXPECT_EQ(m60, 36204871);
    Aesi512 m62 = 21676016617116, m63 = 75122829;
    m62 %= m63; EXPECT_EQ(m62, 414627);
    Aesi512 m64 = -1999725666682, m65 = -4655668;
    m64 %= m65; EXPECT_EQ(m64, -4524650);
    Aesi512 m66 = -13442336259545, m67 = -79617243;
    m66 %= m67; EXPECT_EQ(m66, -79420397);
    Aesi512 m68 = 7141387217361, m69 = 18976547;
    m68 %= m69; EXPECT_EQ(m68, 214492);
    Aesi512 m70 = -70624997799901, m71 = -78849999;
    m70 %= m71; EXPECT_EQ(m70, -78745588);
    Aesi512 m72 = -848542427966, m73 = -1009884;
    m72 %= m73; EXPECT_EQ(m72, -525458);
    Aesi512 m74 = -18916578704681, m75 = 78085076;
    m74 %= m75; EXPECT_EQ(m74, -533225);
    Aesi512 m76 = 36446022312946, m77 = -64759312;
    m76 %= m77; EXPECT_EQ(m76, 64353154);
    Aesi512 m78 = 37344597379618, m79 = 79024810;
    m78 %= m79; EXPECT_EQ(m78, 967538);
}

TEST(Modulo, Signed_DifferentPrecision) {
    {
        Aesi < 384 > first = "-9212123903380763278101201131077887055248270472066703026833951090289752111125710803750532800554307780355950233624499.";
        Aesi < 384 > second = "-37087789512414492205196779606086372414501194510707902956528508145980596721567.";
        EXPECT_EQ(first % second, "-9052785355585321901678007264182200660636851749985107144462309299294274871465.");

        Aesi512 third = "12089899034527593196140344328636122649910515780008608308617745290839686195645435400927167283773708549570445484997504.";
        Aesi < 384 > forth = "38791081228502430712295259601297978608185027799939115358486031550869170231270.";
        third %= forth; EXPECT_EQ(third, "8229687390510874391492419228200312885571290359808778645138412734611230783154.");
    }
    {
        Aesi < 480 > first = "25414663572306423936490250923572266652416682314002638641807901374746684688007267878261403919132312814374630716851907.";
        Aesi < 352 > second = "-57748043104944908853467289109093066086320028315245852872075560040678510723143.";
        EXPECT_EQ(first % second, "7811518141519210546951227708961227216767377145666525439039853091819876995170.");

        Aesi < 448 > third = "22757490737086579577394309223959070847014595257477270224373020007725779776762142727489192667823520369419737745918275.";
        Aesi < 320 > forth = "-77937983826943278060297401266069340866369983469857999581207368639217128209499.";
        third %= forth; EXPECT_EQ(third, "50072887114738719811992856366839041528849040372290680330692851276184261984500.");
    }
    {
        Aesi < 448 > first = "-14440547008645304188997321069862130815751971449927948904946175806842433185837031351222212459083146785238909192902275.";
        Aesi < 320 > second = "112243590895343647310811440527884338254586019129397908671896596052123913308399.";
        EXPECT_EQ(first % second, "-105207684422391308462667250571554459875195292666975849711787192697916843753745.");

        Aesi < 416 > third = "-10268949206820969989650611477721896714787377575245565315296992625964313634488310621397554691862059871720214276194032.";
        Aesi < 256 > forth = "-97915859798192899741146385835304197695884521475350318896921504856785445282710.";
        third %= forth; EXPECT_EQ(third, "-21354871838365052150982584642258077127963991294614846633906216571672069408642.");
    }
    {
        Aesi < 448 > first = "1032860400567937369294772897465672303081805162378681352688737456777397449446292230974770359450531789943904891108387.";
        Aesi < 256 > second = "49992660152609979714616642494208374242393532898305184801038884289201332944987.";
        EXPECT_EQ(first % second, "113339249487640625811812408254081423215075210215409887918689878459009614131.");

        Aesi < 480 > third = "10980211111151232730221827844457700856371004188825896750254777877040050084717142885525432167659985530456887665107732.";
        Aesi < 288 > forth = "111621867721925962090303827881892225819466844466429605168925336016810664659780.";
        third %= forth; EXPECT_EQ(third, "44381282469867310094177193417659636271343100057435928554900709821497673605152.");
    }
    {
        Aesi < 448 > first = "-25312179451839675712380043563674090495413264683538724166215812895403185336921532657434465273628749843184136930807580.";
        Aesi < 256 > second = "-13895025758207086054244308634159781036094931846266326751823269030546314855586.";
        EXPECT_EQ(first % second, "-201748163201676557356245742421857800785977779813132610985415626669522669092.");

        Aesi < 416 > third = "-2317827191029444904787826232969645717508041352748511323648884219838964298888609893452064919399820318106223227173844.";
        Aesi < 320 > forth = "8436753361943231389439346120724592165859648921557619449921495675330933425699.";
        third %= forth; EXPECT_EQ(third, "-2263846971461383090136249614950370347898202632502562196203041317425754139606.");
    }
    {
        Aesi512 first = "31244392556242193721445195543558932714691196546472353207928976569317835494440164003717806353246559324551981369843869.";
        Aesi < 384 > second = "112655450252104889127696699479658308940753477957066554290717216582011363805116.";
        EXPECT_EQ(first % second, "12345422148729111675954511994026012126840701016415285005720252426519703009925.");

        Aesi < 384 > third = "-109140342621221363471153771322382958301489183498408786342767976384318181205816199407011529603128663344781669905628.";
        Aesi < 288 > forth = "-75892564242748082637611017838729600362288292764818468621370824333290922949383.";
        third %= forth; EXPECT_EQ(third, "-70926535546088036262298176982017305013465936388853829442915816024123147606474.");
    }
    {
        Aesi < 384 > first = "-28369946867488170055671205491826488365803381329638654051798981143787271497672096395411274222232984227455180006372868.";
        Aesi < 256 > second = "50869792556243205506291616301118960727198628942580629046575048241139947195320.";
        EXPECT_EQ(first % second, "-5300589201189115965645598494258157144445538457811986299293419485426972890868.");

        Aesi < 448 > third = "2007239891649539723191422951266448511743117164767402034827412248055153401146597539085381864969696749809063896624858.";
        Aesi < 288 > forth = "-30823171253870387123979167629666532829308854120927969018392818887298177506474.";
        third %= forth; EXPECT_EQ(third, "16818192648133960168125890848363499312578421253764243840705735777959142558534.");
    }
    {
        Aesi < 480 > first = "18401784329762856598409454200368774170003833871662814584097362056242378565769588367902877940864902578881991115004039.";
        Aesi < 352 > second = "109146620508541400731363919556439697970404771684009134024160960730813195563451.";
        EXPECT_EQ(first % second, "59062318667871312439029780554739593471079465958987683296021934946388679263920.");

        Aesi < 416 > third = "-6269512436411102690352587190692204852716188403995915713384495919931337525940639612612950219904620161855090128450306.";
        Aesi < 288 > forth = "103817385037534075520242607790864340548028903204102817176165192185569351086737.";
        third %= forth; EXPECT_EQ(third, "-5370583930034390280524814130445256149197679585696656312308038931817196793102.");
    }
    {
        Aesi < 448 > first = "-15809470961307875686900914101691535179028188048740623626775500132444594667772158915089086882184961833896937757697518.";
        Aesi < 352 > second = "-55637352332499955585093674745884423921382906037056045925844302017211169047179.";
        EXPECT_EQ(first % second, "-2618350721932096066273996677127790516943984555887062896398971595895395989801.");

        Aesi < 448 > third = "-533079177926794179257261659652505711444679644908658230707697657135068057932350373496071031312272808107935093829461.";
        Aesi < 320 > forth = "-59508398298085357125846173095202557527114876375603676425011417164862105407439.";
        third %= forth; EXPECT_EQ(third, "-85404996746367892246051259587090408338841652256247697076697106516788249173.");
    }
    {
        Aesi < 384 > first = "-36583676311863234784357950896068621336718336700978459059373589557061144517325459533454086638519471788560272064006479.";
        Aesi < 256 > second = "12357806850776818332841004196660059289899452652284173242333290284856306917298.";
        EXPECT_EQ(first % second, "-7546961048141504531649148628520554208233034929255061220239509820294525628715.");

        Aesi512 third = "29272051253423958896544921736048012129706170514888662826161006110021915492709160102383017123215486139604235442614027.";
        Aesi < 288 > forth = "-62731245046001680312430732227512103478313587961638610237569835747267981007697.";
        third %= forth; EXPECT_EQ(third, "37269649534141699019542914304358347533220752508841119952650366538322887929711.");
    }
}

TEST(Modulo, Signed_Huge) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi512 o0 = "409500603940670545537551287796342541477788606852461036351697131464175962962845171905572635924624049942630184447440898508507588051822633537161798509461250.";
    Aesi512 o1 = "10261006895462786798066807577829165719.";
    EXPECT_EQ(o0 % o1, "7497189308736801177714364072477269726."); o0 %= o1; EXPECT_EQ(o0, "7497189308736801177714364072477269726.");

    Aesi512 o2 = "3886411772506028221227877545626782790074712740712553207205545966324480930589712753901017430541974158930992947273912583719658305944138848039654435813751038.";
    Aesi512 o3 = "44096160074129786340066812371492655363.";
    EXPECT_EQ(o2 % o3, "20727140241842290502169830885947155782."); o2 %= o3; EXPECT_EQ(o2, "20727140241842290502169830885947155782.");

    Aesi512 o4 = "5782112607144331445913519101823821390836357661493880199113633328285711461417518241649866161455224635233300522624220946448998880164188304554140297509893037.";
    Aesi512 o5 = "186990788979412509144995548940300654866.";
    EXPECT_EQ(o4 % o5, "12306435039372636481765829227078211559."); o4 %= o5; EXPECT_EQ(o4, "12306435039372636481765829227078211559.");

    Aesi512 o6 = "2843367156814186681542379218890611049194003988673027820663942011295378309145895243790785406022245564376536475556957271004080525636638225062568533884101072.";
    Aesi512 o7 = "228636154598164644322391966513151458997.";
    EXPECT_EQ(o6 % o7, "194355276045908173211405259771300174660."); o6 %= o7; EXPECT_EQ(o6, "194355276045908173211405259771300174660.");

    Aesi512 o8 = "5268624167917989426050251946214907727544521871316281653054093634261481666781354471840456657534844285146243659351531171776807117198814130432780172839819510.";
    Aesi512 o9 = "262920781462421681850649216248130035430.";
    EXPECT_EQ(o8 % o9, "224316096031665473488622865469212905010."); o8 %= o9; EXPECT_EQ(o8, "224316096031665473488622865469212905010.");

    Aesi512 o10 = "3255796819162371772904139044318827357967373867885413075708573651599340942456720610736045929439330946266098941387425056080719071520970968310702255644695028.";
    Aesi512 o11 = "259009686937591867471923336259624583741.";
    EXPECT_EQ(o10 % o11, "158827090753501984610230418806072355711."); o10 %= o11; EXPECT_EQ(o10, "158827090753501984610230418806072355711.");

    Aesi512 o12 = "1937377249121786152288673920996562706638298937149225729800615065636170823578002251140359176671629015890726222367167405797638750093707880654269403751319963.";
    Aesi512 o13 = "264527922677648264750359572855100885.";
    EXPECT_EQ(o12 % o13, "154942033182658075561332168883657608."); o12 %= o13; EXPECT_EQ(o12, "154942033182658075561332168883657608.");

    Aesi512 o14 = "1004619018110078573146586018714489307479676958048469894762326056956188302698266541490825190187971837893626477831382913310677130751537651231741654765057804.";
    Aesi512 o15 = "78604148911840414524739578695508301374.";
    EXPECT_EQ(o14 % o15, "2673893063474941930634501880419379360."); o14 %= o15; EXPECT_EQ(o14, "2673893063474941930634501880419379360.");

    Aesi512 o16 = "2968308201678225070789020661140137670353545701927900382857175964818890203445515853740569959700311024105611455539739145781359705351814009021402824580191694.";
    Aesi512 o17 = "102335699045058185986847295724245368593.";
    EXPECT_EQ(o16 % o17, "83018314434645209561389061119121391269."); o16 %= o17; EXPECT_EQ(o16, "83018314434645209561389061119121391269.");

    Aesi512 o18 = "6399223173495461022814389622224888155709137194509163748069011262368265881242332077276505334097101129630929712761185904867738790983108455519052748667822324.";
    Aesi512 o19 = "323647146437674715682665224889038002887.";
    EXPECT_EQ(o18 % o19, "295772928839849326058916007422140572613."); o18 %= o19; EXPECT_EQ(o18, "295772928839849326058916007422140572613.");

    Aesi512 o20 = "6575890394347046165268210624739882837600890257472722546904136362203033023854575209766241340128025210889225150005208977548836726881179415009099311192797015.";
    Aesi512 o21 = "168376457763065705046441976475060762246.";
    EXPECT_EQ(o20 % o21, "155352949924991062968278479623860679005."); o20 %= o21; EXPECT_EQ(o20, "155352949924991062968278479623860679005.");

    Aesi512 o22 = "4041789371535566240417701768508927524915551543709807836289714165726516264092034971407694450602438130381275260327868337897367753156151007095214720873052399.";
    Aesi512 o23 = "123374323747828929507911544081484071237.";
    EXPECT_EQ(o22 % o23, "13320076866500736646281629579480866720."); o22 %= o23; EXPECT_EQ(o22, "13320076866500736646281629579480866720.");

    Aesi512 o24 = "590869823402819086989170628687010150951691926308152736817633991948600085446047537496411979552821752174491137647422974798297797197624832652578926636605930.";
    Aesi512 o25 = "314553084668450361630418155239078801123.";
    EXPECT_EQ(o24 % o25, "220287220658243788130402890302396016445."); o24 %= o25; EXPECT_EQ(o24, "220287220658243788130402890302396016445.");

    Aesi512 o26 = "1255347790745621962558443760354458476033096824995022913344089590443998331465310225325616786089254163138528065491459202824356793472756199799206396239679459.";
    Aesi512 o27 = "218355050285715746012411800450761028105.";
    EXPECT_EQ(o26 % o27, "24761501791080241662542463171241123689."); o26 %= o27; EXPECT_EQ(o26, "24761501791080241662542463171241123689.");

    Aesi512 o28 = "4381496889340101805506525945483709074356222961676263171524797959684865493014817513528842471387653457839322714274888410560424445186746292816603977255073914.";
    Aesi512 o29 = "213646455571661215597944392522675211455.";
    EXPECT_EQ(o28 % o29, "11293812890720308297039503199609002539."); o28 %= o29; EXPECT_EQ(o28, "11293812890720308297039503199609002539.");

    Aesi512 o30 = "333867399274973622550994403766046829963885836018123458264166484573394379157213306412994521506872094126340500566748952725955867119701609638561645349313850.";
    Aesi512 o31 = "254770913098881001110999644381084874607.";
    EXPECT_EQ(o30 % o31, "24905172009758338194804279867673180189."); o30 %= o31; EXPECT_EQ(o30, "24905172009758338194804279867673180189.");

    Aesi512 o32 = "4153957660216755241712750834150411661035298295206729249711442788183588355380702666847500257892872704424548402482292801439719912730182796069865834062254139.";
    Aesi512 o33 = "35172699062374537619339059626016156514.";
    EXPECT_EQ(o32 % o33, "11916123947850557221344759305315357637."); o32 %= o33; EXPECT_EQ(o32, "11916123947850557221344759305315357637.");

    Aesi512 o34 = "6564694609793748905027852014916558493090912779101388542515970857992202876322747849605134933014988255333082800961843126101195762330518076934117865523730038.";
    Aesi512 o35 = "246242350222197521652120611865691858177.";
    EXPECT_EQ(o34 % o35, "103632595141445024701649882083556357692."); o34 %= o35; EXPECT_EQ(o34, "103632595141445024701649882083556357692.");

    Aesi512 o36 = "5312195220127511325547902706779167794285142602042722851112237562160335575848449548077927679780286915630540657371591170037296964978376150839519336050273863.";
    Aesi512 o37 = "178934842548515544690430019499484966948.";
    EXPECT_EQ(o36 % o37, "142334313754817206394898080858138985023."); o36 %= o37; EXPECT_EQ(o36, "142334313754817206394898080858138985023.");

    Aesi512 o38 = "1401047116413257578346839568879405361213132567894990121171733539467157022041516175484487632432100930077579116773399387005960318967953549095134207067150841.";
    Aesi512 o39 = "151246567336848655594742713804671303477.";
    EXPECT_EQ(o38 % o39, "56129243505051103512160725382779920784."); o38 %= o39; EXPECT_EQ(o38, "56129243505051103512160725382779920784.");

#ifdef NDEBUG
    Logging::addRecord("Modulo",
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}