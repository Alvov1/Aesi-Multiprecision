#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../../Aesi.h"
#include "../../generation.h"

TEST(Signed_PrecisionCast, PrecisionCast) { EXPECT_TRUE(false);  }

TEST(Unsigned_PrecisionCast, PrecisionCast) {
    constexpr auto testsAmount = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto init0 = Generation::getRandomWithBits(250);
        const Aeu<256> aeu = init0;
        EXPECT_EQ(aeu.precisionCast<288>(), init0); EXPECT_EQ(aeu.precisionCast<320>(), init0); EXPECT_EQ(aeu.precisionCast<352>(), init0); EXPECT_EQ(aeu.precisionCast<384>(), init0);
        EXPECT_EQ(aeu.precisionCast<416>(), init0); EXPECT_EQ(aeu.precisionCast<448>(), init0); EXPECT_EQ(aeu.precisionCast<480>(), init0); EXPECT_EQ(aeu.precisionCast<512>(), init0);
        EXPECT_EQ(aeu.precisionCast<544>(), init0); EXPECT_EQ(aeu.precisionCast<576>(), init0); EXPECT_EQ(aeu.precisionCast<608>(), init0); EXPECT_EQ(aeu.precisionCast<640>(), init0);
        EXPECT_EQ(aeu.precisionCast<672>(), init0); EXPECT_EQ(aeu.precisionCast<704>(), init0); EXPECT_EQ(aeu.precisionCast<736>(), init0); EXPECT_EQ(aeu.precisionCast<768>(), init0);
        EXPECT_EQ(aeu.precisionCast<800>(), init0); EXPECT_EQ(aeu.precisionCast<832>(), init0); EXPECT_EQ(aeu.precisionCast<864>(), init0); EXPECT_EQ(aeu.precisionCast<896>(), init0);
    }

    Aeu < 448 > l0 = "554241380131154142962417280417998404754929828012346097179687246704910303315350311367744175378443304128036758983824324734859979768793199.";
    EXPECT_EQ(l0.precisionCast<288>(), "408360527252105534419027813905394704348416525449517229626145092686873334890812834908271.");
    Aeu < 576 > l1 = "173149006578507408541558843749803097456182801623666290320265239338181642132588175569300761116459015677654013937947922994294373883893325690805618652779686048163576445320646483.";
    EXPECT_EQ(l1.precisionCast<128>(), "210443040993062611785785420324814540627.");
    Aeu < 576 > l2 = "66531456573720529442216757952933331047332848598711407403162907502778746277356003198417676048381460115126892609965612607567352416913344652080416627158294298245466839069775512.";
    EXPECT_EQ(l2.precisionCast<160>(), "1443229828299692348357792670919834022979176257176.");
    Aeu < 384 > l3 = "1098897099174270538478178379535802020226067483295913532002005072123857444388599717512365115154922424475266850503120.";
    EXPECT_EQ(l3.precisionCast<320>(), "1913199418936122088346837076981965348887692948209008066197015782466463820230886895375591805565392.");
    Aeu < 512 > l4 = "3069476797928086052610870903467024437789501028462434594919744663711449127505497769599003817092647117494652657649787519339440892062605145266124294683385280.";
    EXPECT_EQ(l4.precisionCast<288>(), "404126981474586070624526050760237057921124714973395197115667617772232279058850682265024.");
    Aeu < 352 > l5 = "2948016904402708200913240914482790524802448562129494737510300849124666954825609112501202095235581771983067.";
    EXPECT_EQ(l5.precisionCast<192>(), "1120442763073472009948111228858560283233140945287272409307.");
    Aeu < 544 > l6 = "10812628969630480520296504856105212223514943626905130339806055726389997460671652738109321632680730144583156832174036283780045422846847056560092948355830105461452561.";
    EXPECT_EQ(l6.precisionCast<128>(), "330342928733391374787712324367724277521.");
    Aeu < 544 > l7 = "4112600136041780850943347403651201772866193144792995503451651200434798923608930918174270859878921981299563448120913143826254282006786082095303150341646975970259785.";
    EXPECT_EQ(l7.precisionCast<288>(), "257304491601243764544329915240262128843475594535859368626521507218543067528530890011465.");
    Aeu < 384 > l8 = "28048835949300858284373720359733481364253247860667645736041649560388756406925265368724245525050388699493492127215396.";
    EXPECT_EQ(l8.precisionCast<256>(), "41878091393227431195219434526290235492405672892345774495477168807404321681188.");
    Aeu < 640 > l9 = "3848583613533234298953998803964435640019755235139553979562211441430877304606643444685829881598731682613890268054287149783980908651187506403654762887423297734191981318623098337988025363561114940.";
    EXPECT_EQ(l9.precisionCast<160>(), "541300877713023860093328693961789219540436446524.");
    Aeu < 512 > l10 = "8488462826013523989070679682331534259557436274695014765129319667636222109541019846966725069632607848463488852770772093964627525568328843684570072600088832.";
    EXPECT_EQ(l10.precisionCast<160>(), "523538838524052847229876918926034085392282358016.");
    Aeu < 576 > l11 = "11236969385546154746878679488297868602806916111627187642798923353498840149284473359875405738278850824908974037092662116610981773121895588572197349961886691945064651776229050.";
    EXPECT_EQ(l11.precisionCast<128>(), "29179789164106523943541955618292268730.");
    Aeu < 480 > l12 = "924082864945322453283057577897421592990774691137290429034141806501722486590884960785337810624006801980116699350354269354398363040536296529326681.";
    EXPECT_EQ(l12.precisionCast<256>(), "15790246827509817793328866413084671789409373095293932456575888129892184748633.");
    Aeu < 512 > l13 = "9607535504179656540242447115527397529604623905521687360978734063119313561780639573012578165648934865049773849480530263572594554247909207343890208529325178.";
    EXPECT_EQ(l13.precisionCast<96>(), "43716316362024562473758891130.");
    Aeu < 352 > l14 = "504397667162110386921588331778312416154169943905827797809093816111203720055808585599403524127547394508185.";
    EXPECT_EQ(l14.precisionCast<320>(), "366270022681278488140630850294606110994384776947473041868263123438838372271196545770306825766297.");
    Aeu < 544 > l15 = "27608764507031531319380382106462863564782632077695501707054751614193288192180423340509534352068491112085645377731761780973224450169988873068318780343686563718281728.";
    EXPECT_EQ(l15.precisionCast<128>(), "299026899257808842617251361117814438400.");
    Aeu < 352 > l16 = "1978129624964056450214026876059300487877996643140932950061364608660420356925042564405231900323677600676732.";
    EXPECT_EQ(l16.precisionCast<320>(), "365332643823514689801663993143371567677009732484894048068501453581388238598858847283260927441788.");
    Aeu < 576 > l17 = "195265355728943716865923619155375969530111891393015085388776834717930166073822658430177349287072569812662023940174467801397134386843815407817766067594355521616265929622366898.";
    EXPECT_EQ(l17.precisionCast<160>(), "221431119546753700832094394260270558954283936434.");
    Aeu < 448 > l18 = "414657978233561248876686250413054606900177295904410399578041220021615424324810017863916222257119299661647319122459751028412186388057654.";
    EXPECT_EQ(l18.precisionCast<160>(), "287129154900729631574915186396319442248636812854.");
    Aeu < 416 > l19 = "56687510612260333551549640577823250078355837097233108084669493198465659199087411861462621495608437125573216962137258061197924.";
    EXPECT_EQ(l19.precisionCast<192>(), "3671611332121323818239920054049877596259686932276771811940.");
    Aeu < 640 > l20 = "1546223449542899971558886608789690118697134740086024592798235620930020323290348765670723997823619247765209481873838363731256837657816604584457400460176692587426042570895554473755247436115230189.";
    EXPECT_EQ(l20.precisionCast<320>(), "1705058499460706178048346990812902342294295809567394699482134087317553182108908996873089872575981.");
    Aeu < 544 > l21 = "174065680196633578590894771435895688468079098486287540664921962829963159709119456409138894766839898574579608494679948491219544731370758383681896937747086278361071.";
    EXPECT_EQ(l21.precisionCast<160>(), "220794710368648048438000780675654519053309411311.");
    Aeu < 448 > l22 = "136337341513559528961653942676228706381771186833849367631461720301064785651836843206251270469997623786977461200400063656318366284038665.";
    EXPECT_EQ(l22.precisionCast<224>(), "816945909177209805889026909383334983853515920833335341477807047177.");
    Aeu < 352 > l23 = "5882616107094423670384170997721509360066167036113934904312412848395219509023523677500755059444938780072409.";
    EXPECT_EQ(l23.precisionCast<192>(), "2482927072274059692659564052439247617034525043020273441241.");
    Aeu < 576 > l24 = "49857691326617957339080952718292050690932576020199908481126156219200801064319573633766216649464944745473827870799580369859357610319478396095070633417622983092014069297900742.";
    EXPECT_EQ(l24.precisionCast<96>(), "20676605872175966156536081606.");
    Aeu < 384 > l25 = "28597517267806223597843711502305823530466336309097073757475161768363138442370308787956725565043616828782360287501138.";
    EXPECT_EQ(l25.precisionCast<192>(), "2349989946988515365971908154524797542461532293278087240530.");
    Aeu < 352 > l26 = "5359709770582948688991943818180669926832250789635927200530270394005210447928718134456374737387522683948933.";
    EXPECT_EQ(l26.precisionCast<96>(), "30780870628112076616018546565.");
    Aeu < 608 > l27 = "800374186897442994345991532287383427669343624463618411247954314190792985454915883373717659254388118542800958906763592267710368929085300529376043722066941548681087911311229845689540460.";
    EXPECT_EQ(l27.precisionCast<128>(), "2854867648611582803837292073272156012.");
    Aeu < 384 > l28 = "33405800918446002208040817593884350731087014451427547055891417316377644807769010652297688512536252488106030729965818.";
    EXPECT_EQ(l28.precisionCast<224>(), "459273051701248468242064512442751256897801283813240021045830922490.");
    Aeu < 448 > l29 = "545136889659423149046640496483142589267039543174772529930583944491083582156213563276447863450923726927200116011820859957964687342869311.";
    EXPECT_EQ(l29.precisionCast<224>(), "11564578527812652389272649650310409698199102565460528063263161490239.");
}