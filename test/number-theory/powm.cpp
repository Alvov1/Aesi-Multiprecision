#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../generation.h"

TEST(Unsigned_NumberTheory, PowerByModulo) {
    constexpr auto testsAmount = 256, blocksNumber = 8;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto base = Generation::getRandomWithBits(blocksNumber * 12 - 10),
            power = Generation::getRandomWithBits(blocksNumber * 16 - 10),
            modulo = Generation::getRandomWithBits(blocksNumber * 16 - 10),
            powm = CryptoPP::ModularExponentiation(base, power, modulo);

        Aeu<blocksNumber * 32> b = base, p = power, m = modulo;
        EXPECT_EQ(Aeu<blocksNumber * 32>::powm(b, p, m), powm);
    }
}

TEST(Unsigned_NumberTheory, PowerByModuloDifferentPrecision) {
    EXPECT_EQ(true, false);
//    {
//        Aeu < 1088 > b = "57586096570152913699974892898380567793532123114264532903689671329431520511113834798455284506659101071783731920229799439845945850001882401931509362202255401183871999";
//        Aeu < 512 > p = "115792089237316195323137357245424007756186726618908221633794085042696929411079";
//        Aeu < 1088 > m = "57586096570152913699974892898380567793532123114264532903689671329431521032595044740083720782129802971518987656109067457577065805510327036016891142675844838996312064";
//        EXPECT_EQ(Aeu<1088>::powm(b, p, m), "39425654526107845777534757940252133690690500017817066812291843029593672275529926593423190550750419392824945760638655783123482275062105795307978947593311022434229247");
//    }
//    {
//        Aeu < 1216 > b = "1062275985633534197379176413098233350694217985486845501860948811928112156625225383756074256777023821120951484760158914493906870028350298551175190050077976241237449780259849816638488576";
//        Aeu < 512 > p = "115792089237316195411016781724986755903953896272547229400021678868641165213680";
//        Aeu < 1216 > m = "1062275981676247773809503868136896918344831194891962260779498837775798222235348883782564132704528202055379948047400744675847606966191432593882325306975554164754257180661652547409281024";
//        EXPECT_EQ(Aeu<1216>::powm(b, p, m), "367212410660349293556324601728065904864465591684104661804455312007376321346007876135761176000158090074118654491979754235578022146043364297442338651781033978643260258822572193433518080");
//    }
//    {
//        Aeu < 1024 > b = "3121748550315992231381597229793166305748598142664971150859156959625364501814188287858092129876928928165586101747365020360195509581413690642530303";
//        Aeu < 320 > p = "1461490486963620564874634758333721921125397561407";
//        Aeu < 1024 > m = "13407807929942597099574024998205846127479365820592393360225555645457668635093529159860903375341547980544332366780321199574129889216939184158328232821653504";
//        EXPECT_EQ(Aeu<1024>::powm(b, p, m), "7558784404007254179446202013527355286311652987153519874299029622893115829321653711359187394756810497438550966453777044499316481698080147321408978876891135");
//    }
//    {
//        Aeu < 1088 > b = "39402006196394474837777590534119765060075285035223558459942780075314432264177467154966809113375959303628109239026687";
//        Aeu < 576 > p = "497323236409786642155375666128591555276287530920617488043514648350609880299066793918463";
//        Aeu < 1088 > m = "57586096570152913699974892898380567793532123113003668714652434414310560084205278948179183670766658714471158035184897947977991938123264469667722602304733233878663167";
//        EXPECT_EQ(Aeu<1088>::powm(b, p, m), "53750455023425484251590451536225101504065350384994488259671724003959955961949211487858823711077412591697854622831147179340278924737999949165432908832771319540000734");
//    }
//    {
//        Aeu < 1280 > b = "4562440617622195218641171603857536234648335268849363357831014348506247558079881512662571649039502557186432406335067778684252157262718916566987175660155911237399442436210624017667798014472740864";
//        Aeu < 640 > p = "2135986017402921915152204728370999543107810658702551030548942207690795127816630233948373376827392";
//        Aeu < 1280 > m = "247330401444311485775426064169659743585941065205122699668893200193986361758817106118602877866534999448796070226567555134413719075795233475073408356551927274789783246120943615";
//        EXPECT_EQ(Aeu<1280>::powm(b, p, m), "71264946182442896501111445551292324403933274599945950715454908540527922469326567258770589455295819618860258960659949847068266956542322357308123242207639532834185396056696466");
//    }
//    {
//        Aeu < 1280 > b = "4562440617622195218641171605700291269974847226203682213701117539962914098952224014513569647584321690996277108355474989175092994780793822144284304711061489162472004594364639851026124360325783552";
//        Aeu < 320 > p = "1461501637325586006220545169224667791414811230207";
//        Aeu < 1280 > m = "726838724295606890549323807888004534353641360687318060281490199180639288113397923326191050713763565560762521606115062206082772981776384";
//        EXPECT_EQ(Aeu<1280>::powm(b, p, m), "573085401894547166516877089750418966107279455078782547574800402627962088464692182111276871789969819688769511207932354915166690109030400");
//    }
//    {
//        Aeu<864> b = "11090678776483259438313656736572334813745740344331483744095643836342514882089060884861484872297597135700949968707655805690179485695";
//        Aeu<256> p = "340282366920938463463374607430694469632";
//        Aeu<864> m = "139984046386112763159838289862099970323476066441406953502400293335070582272926757986317508127144017920";
//        EXPECT_EQ(Aeu<864>::powm(b, p, m), "69079484759223978771725604820876564626247001712169147692709103033711794245882473503725969639949205505");
//    }
//    {
//        Aeu<768> b = "39402006196394479212279039590884619721458231838130542792678389807758629298361274489173579184684013616334120517894144";
//        Aeu<128> p = "18167520626212866175";
//        Aeu<768> m = "601226901190101306339707032778070279008174732520529886116452717031736444463277932798047865154700400216954896384";
//        EXPECT_EQ(Aeu<768>::powm(b, p, m), "166447213589684059641991998731395753952867333867023648121519053118548941940730378205117569499309680382883135488");
//    }
//    {
//        Aeu<832> b = "601226901190101306339707032778070279008174732520529886901066488712245510429339599267430114963433206756870717439";
//        Aeu<128> p = "9223510592466305031";
//        Aeu<832> m = "169230328005378390557158926912791928617237135317019459417751424189778811413571839323317524417207459349096616339035945982496767";
//        EXPECT_EQ(Aeu<832>::powm(b, p, m), "163571374596502077158450643805430090015294226721581258074063672607038855306387274368353843169596610829477792365728343013827463");
//    }
//    {
//        Aeu<800> b = "601226901190101306339707032778069815839817783676997359291355200569599889047518065525993786130227286043797749760";
//        Aeu<256> p = "340282366920937896779396662524586754048";
//        Aeu<800> m = "2582249878086908589655919172003011874329705792829223512830659356540647622016841194629645353280137831435903171972479057920";
//        EXPECT_EQ(Aeu<800>::powm(b, p, m), "2579701408596237953229338966783854517127482461408021202063410591513655437652750417878791830806844428048643028427032494080");
//    }
//    {
//        Aeu<768> b = "9173924471937092994040168135609609028485434160723415743258425125724447672427269041667050124299132165160960";
//        Aeu<256> p = "170141183460469533963142207355997913103";
//        Aeu<768> m = "39402006196394479212279040100143613805079739270465446667948293404245721771497209282186270469969042736999567709962240";
//        EXPECT_EQ(Aeu<768>::powm(b, p, m), "5482278480730755497604397215387492614088698182310191625462706667534436415097143802580027802613073006276679523368960");
//    }
}

TEST(Unsigned_NumberTheory, PowerByModuloHuge) {
    constexpr auto testsAmount = 5, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto base = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                power = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                modulo = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                powm = CryptoPP::ModularExponentiation(base, power, modulo);

        Aeu<blocksNumber * 32> b = base, p = power, m = modulo;
        EXPECT_EQ(Aeu<blocksNumber * 32>::powm(b, p, m), powm);
    }
}