#include <gtest/gtest.h>
#include "../../AesiMultiprecision.h"

TEST(NumberTheory, GreatestCommonDivisor) {
    Aesi512 l0 = "30922697523418831315214891280724918670019168727095386744994280347667499352114.", r0 = "4825339643774008360293590062816345530.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l0, r0), "12821438624842477420992755262.");
    Aesi512 l1 = "31961849080292819785268969543663294179771430721697647617079339485020357144725.", r1 = "3727446067991809648384987795279246850.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l1, r1), "3882505907282335979901858575.");
    Aesi512 l2 = "2307736447612429553667524445677575638234923373157313437870516348009262621928.", r2 = "67002515202141325147876176863802480.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l2, r2), "523165362710136316533116232.");
    Aesi512 l3 = "87046430649976560226711137783359050054236522238169223067744177695246626787600.", r3 = "11083446863245314338074758923226333735.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l3, r3), "4603429327046522544642727995.");
    Aesi512 l4 = "2391811794746649113187937712184739599099261148479143466809742959380047499438.", r4 = "327745934057576533326712586016092518.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l4, r4), "143611067052100504299536054.");
    Aesi512 l5 = "44102409612007505801285956112915700289576509217269628923706568721411283312488.", r5 = "15838397973720495232516710942765458764.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l5, r5), "4566197733369846131013859196.");
    Aesi512 l6 = "79337215513198539437052916170435400758362057729637120210230374439094242545016.", r6 = "5998700420433472567004541930660578444.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l6, r6), "8076610097507745564409968628.");
    Aesi512 l7 = "64688780966376239752530133685658334511281103405201777002776066235341365129225.", r7 = "4859093234270766711420050669067429550.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l7, r7), "22987909510683357171355679725.");
    Aesi512 l8 = "8606476750999578310003542728151498312333497241114339997404346437176577910370.", r8 = "2606694868714084117668961558426764400.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l8, r8), "890943595962646000321857830.");
    Aesi512 l9 = "41456516822331541542938716464591363340881610949116050481728091453839400883584.", r9 = "8356327982899908419046367330839170176.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l9, r9), "4786909156715155203589253248.");
    Aesi512 l10 = "68221902204359692596119879310883268988091867091462301382543495412706155999452.", r10 = "4692852397375603885011381554661571744.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l10, r10), "4314511915565124367185979684.");
    Aesi512 l11 = "24114999837796657166287604944540410227156337201469348187649765420349841357992.", r11 = "2855447415744833258931253501517747248.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l11, r11), "2474739521215927670528768504.");
    Aesi512 l12 = "23711513036257109407398342788642671829538419035624577994787377444276059416680.", r12 = "3347636006521072248545546680133831130.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l12, r12), "4867346944978083308657790810.");
    Aesi512 l13 = "29346419638217737414435956954609077368678593255132899552015342306906672087200.", r13 = "90954880292432935657387893760726000.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l13, r13), "112054108687802755505248697200.");
    Aesi512 l14 = "29476317395407089911261312268911929968587137172157494955214116775873087002061.", r14 = "13308080226118825510261426781243071770.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l14, r14), "3793360824007759294527111719.");
    Aesi512 l15 = "89573764779741460918333873280117836505405959835704186097479447161000984883205.", r15 = "17607104445806146111708568274291591440.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l15, r15), "21527347800177242015784985165.");
    Aesi512 l16 = "12646031466701559773116141898450730607538681185101374762380668055262753292494.", r16 = "4705616448129365293639833963324497310.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l16, r16), "2756933282368307846591807462.");
    Aesi512 l17 = "39302403754465145079412793955918053326800218301425132215177461104014798953750.", r17 = "4576894909641305828257242987925999500.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l17, r17), "3966531080560468291609604250.");
    Aesi512 l18 = "17911744455708866555684765362321776210450545505166503086932104486668942289144.", r18 = "9111903524761924727487464838632664430.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l18, r18), "3682729310567050602801600446.");
    Aesi512 l19 = "564732013155634417918479498248694836746253834745552577788943049983364444438.", r19 = "75260551213052788294853154032925150.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l19, r19), "416004510531814360285696462.");
    Aesi512 l20 = "64927503629277667338653932331739740988673920750845223696821731184985512422850.", r20 = "9399523915241866046402324975981958072.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l20, r20), "3852144060119078085009445934.");
    Aesi512 l21 = "1773793002146754844972470550543029625448337679701009458431782777220214579571.", r21 = "747264858069902583691864883556614292.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l21, r21), "443059349750663893783468953.");
    Aesi512 l22 = "2650014237119325238937527487845759285239715432602168839876989374804621544872.", r22 = "8089034070341578570464027708155470212.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l22, r22), "17725842748660063896132608196.");
    Aesi512 l23 = "12463423926789413552251689274409070802299479376851380419762400860949805301979.", r23 = "1542114177340381900245729356726294595.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l23, r23), "635095337417151919149358859.");
    Aesi512 l24 = "7603465021154055726119014911467185722952800217676364388407938500399225202028.", r24 = "1496085973645902220722915358950332394.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l24, r24), "3184568066499781944598667574.");
    Aesi512 l25 = "40538249289438196227240095810851288882609292132397142188347961526647333168933.", r25 = "3811846403954782789167633334550225073.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l25, r25), "6768743738168829128304637503.");
    Aesi512 l26 = "24922869391337264884826528013127697962200726361386398416415589076903516009260.", r26 = "8393537262160247533297660367720452730.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l26, r26), "3177608640027389003522455390.");
    Aesi512 l27 = "58169377068871965117053283148661787518908624131511029534163643711150503212693.", r27 = "12962692259456489287126067571744486404.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l27, r27), "25054649462533001363606141819.");
    Aesi512 l28 = "617594407414032680357893092447672547424111387246280734114353787786489354026.", r28 = "159330124225585639174436154648039723.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l28, r28), "38704423754503270142287093.");
    Aesi512 l29 = "16161528934947199638203784807793384766814675807946502654288727327015825643500.", r29 = "5314439976109786902519490330010952000.";
    EXPECT_EQ(AesiMP < 512 > ::gcd(l29, r29), "16960471420805084079824275500.");
}

TEST(NumberTheory, GreatestCommonDivisorDifferentPrecision) {
    {
        AesiMP < 672 > l = "4230676745963713957230888310187630372432069099345644636601407373385234444092181446632419775164492383515032489284154688242326706321937072760697455769189714765485293120009031112653110814316764869420714396.";
        AesiMP < 1056 > r = "209956445382185916560571879684192822710509265996286513955249087273201046635170590440468705699334543201017112397357524814680715615940980228897724843179093148890706469409864131728137510849203862753097509824739361739324467479868924067825818041059172501531522186466176930159962182768295794284302187614761676199305309362598.";
        EXPECT_EQ(AesiMP < 1056 > ::gcd(l, r), "64647248204074430032946973534391241670722146839208916950037622315812135371561662893244373531561692445330126366808337145392822653697465725700567202977462087954432214.");
    }
    {
        AesiMP < 928 > l = "644951599131558468448822699675101480817054053912186902041119715443011322660635640568989777809715116919537330884789519966705429083097889126874505167742076054529441346225944104364013619559102217967512847269152662680659527314311155583027074735196464695453049020550687619282706673780.";
        AesiMP < 1088 > r = "1252941560311043512523997658469534813375458357445260439433069166087750220713059429094251618826186749670237790725745107987671620766916110050637341523661806802438885757938176445765367074693538388179332139672144442296395986579205303043587462303056267734919176254191491656233887570942452329673578160793149291963102726854961476695580.";
        EXPECT_EQ(AesiMP < 1088 > ::gcd(l, r), "203293001641370607163401184222192347776467064691515055952109974505650171074441289837083156052522009397546049907256904793801021573728979083048469298910736690867259940.");
    }
    {
        AesiMP < 672 > l = "9585677621361790747542731699183518553137961470185822862239050776423435924661647685535553667410269642819739499533557302829503576772190609861145081711348261368655191548650535644687555882814079281948869199.";
        AesiMP < 640 > r = "2367441375543594502871331188239360671612910847513218362733938745389664603226319931515532518153642659183826157027536274151677433136113925683393084630993056641826869908962410949704181567477845684.";
        EXPECT_EQ(AesiMP < 672 > ::gcd(l, r), "24559920916699563850536643601549609228979323413871223776713945872990612433752548701691501009334170712436258461631937289240372019140601722675567806966645451.");
    }
    {
        AesiMP < 544 > l = "1025048602017503238379658040884423038448922686219368124848551707652216483592983433990964146351702815507561016381100414953012305321229063309483299606483418891030640.";
        AesiMP < 736 > r = "106448671726508721528388537639752408638277215306306124531666804890581168642263269874377417619676284350098150505258412414111914310935155228646579173088427225279631343476145327012828295266633490885866191085927392050419150819.";
        EXPECT_EQ(AesiMP < 736 > ::gcd(l, r), "428446229629353219094697372973316732133257685757548840764304754131555598240995410109723.");
    }
    {
        AesiMP < 864 > l = "44320326277180188605550230327284864052128535718637012693380665653124594973263618475677931894682101734289305418698016483853144404956679033883483704856378658313356101953359021132507379024088641114187409396079713991346469667148396546158054039226749486286912318560.";
        AesiMP < 736 > r = "41668490976788256893116532112044213633891359795569874372094917663858301505071893549196637596596594227914932990980233830923465691001974909138469425320780768565573086115765816099764002048746527040413983847705904743086234370.";
        EXPECT_EQ(AesiMP < 864 > ::gcd(l, r), "1708361341436101192039773551707751167254746113155336883576640520904553250449029089768025050425490.");
    }
    {
        AesiMP < 896 > l = "139443230929326819742511001780557243544488388811740733158927984804189580958726055988746111731798646180768971018453015883511382613957198801163384882078422832299227575494883780383981543181262373381387871754112543335216794666711979174645718111404389620556052627299659924852.";
        AesiMP < 896 > r = "98231837540696176192716842030040470368853612601491655676156264650167048316924367083154603468182752002665526862941805263818060742932782925036699650401442302401205876587930129172414953177353293741770889950783311103087246650679492133199578391147206960033485669612752011312.";
        EXPECT_EQ(AesiMP < 896 > ::gcd(l, r), "2251253896172415982777972236185412618641618874085326835371445315762588964357064607495729947232052045296360160019139280879776642998272477383363897626540993764114721080183425042679143079111335292.");
    }
    {
        AesiMP < 608 > l = "325769536172480976255095852600606628504126622952311254811505077256553463206419828948119183101950613306990404875120858939254435390950969141484515606308769222930743006127948420503982928.";
        AesiMP < 512 > r = "3771809266741055493549451513157005786066682818233987554309132717627080555992261098905445672420210923903478986471794947463443003537961677262127935771727004.";
        EXPECT_EQ(AesiMP < 608 > ::gcd(l, r), "1247616674975269414985803952995604494402642776008716846763186302548884361983518575690138571082732.");
    }
    {
        AesiMP < 416 > l = "20680688544267474657760903768407052783166744854758811851415026018309749322986577950145407189920193348861574921971735221802876.";
        AesiMP < 672 > r = "2956502787098119466632013235822525257867074821428011258584613412939308483532092473190441822930477576423338687218939016376452752243982602123843718199078277649634938129003787106260707247901593760272874452.";
        EXPECT_EQ(AesiMP < 672 > ::gcd(l, r), "319678443171958180783744886861788320812.");
    }
    {
        AesiMP < 864 > l = "42033894881215996563108824273565563206952280356966715576239049742633461575153250322694502197947526775303289143065490876208104443444417442712617522887284180773682812075536085210513867904387261262790036527018646882430745768502342419510371271677976466130237925712.";
        AesiMP < 864 > r = "75867476167179756925189188822046073211615450659798937685652546340627693592162333337673856937731122878141305419876052715970593556847957206562707054474122862154356783001116876909318649001507577926656457764502751649210615252720606090500519819763918607652245168558.";
        EXPECT_EQ(AesiMP < 864 > ::gcd(l, r), "27672200309473699466062460175842684257346481496243760492409880520870460338024009451890590373869371927971798965935026.");
    }
    {
        AesiMP < 960 > l = "66572371473626869504545823252619142464193730149383021256020143828360084583026747178282066297818530269672193691663477462362949663873447485909091803171757713288071500174398122128974167158184713891975707292163209464233354039844504886846255579841788803831270096992799997314710756515912384932.";
        AesiMP < 864 > r = "2717110386452237801726359356476649734114774942280259956646836330036878242770234356972517318636317673915153651731460015476270816118147748987819601991883041120536291849123792103949964526383119561363238428929684952362487430621947737505145305359619917570502724028.";
        EXPECT_EQ(AesiMP < 960 > ::gcd(l, r), "82731924146567176744607433419954882495328135619958445588203865094017394134698720720140154465802098619756362841393016720545053050602348.");
    }
    {
        AesiMP < 1120 > l = "931314542705274319646632352153246657479452521943146000934558120737682842153774932655132204160672179121064725685166170681064901674325499415057018208653032425424263459263174778437715568286760005443597997407912855518443087722119276194896495318010500498920668255593697386976200185493984053612383767195662605430705598666929947695158169041441.";
        AesiMP < 672 > r = "398242925688781908804629541692075682129940372267467448875144337157239533341579189744226800954256693369907097296571756778485088050475413320601179547067403487744721014306232014448542855962999665164043541.";
        EXPECT_EQ(AesiMP < 1120 > ::gcd(l, r), "38966954206645663651514809733667103933922380986309455307611494248772771450869010561793972862057018068536104713401481520693144745407501972923272147875716979701942577796107271.");
    }
    {
        AesiMP < 864 > l = "10992804516588380186805217471635725871174634307871676918698875265213143540071730800538791442848027248824990939372635016732120921604137649481799252610185755909065764431272038623253455612549541904818623765159251902312254920235405967708229095553864068302950582350.";
        AesiMP < 928 > r = "264835421786154535323939206198359569425765314055458651728141986282472939366203059436429914585114165822513612644403787444253291276420079233329456991861524685051179815905462425465702580586714386497847999288508080114973657727876148808072530687740756214384670550308919267040958960075.";
        EXPECT_EQ(AesiMP < 928 > ::gcd(l, r), "56951597237701561176394216804689598281166885623645391283265117535573544550707037024741985083463421121373175.");
    }
    {
        AesiMP < 672 > l = "12383346959826637872118427447702047086488691075270619429852828771735645050146119574897341135633242685360277384483601737856590574416067657407070172230606545680332151435867203211724609547921893727132274982.";
        AesiMP < 672 > r = "14348294542121772197373903409937840369255203167509944698866799165105485668890432197379691678421214540978338319810411264201636003862768473235193140639532178514767537924713971936767982999021052664568251511.";
        EXPECT_EQ(AesiMP < 672 > ::gcd(l, r), "5395601467742227452695088775148294435531930873981263517441.");
    }
    {
        AesiMP < 832 > l = "1075563162210870127379865975809649527926199247433493483352964442448416210650926451646315396118448209043587167997600329951108747528612281152439446078275152212718602491291788837072118769256000621202922534427628959737088260189079743991721963110408865892.";
        AesiMP < 1152 > r = "9574219070651515826229026814078475385260586488852288382382550113141195513014924143605339943414483810988016027646671965848107590194916779264649780566306465742916400135822764900665037457280692579555471605072859576266614248699324885765658960173216148273812405919438792725752551523804396008628809996901363194043948276316327903250387072357919162867177.";
        EXPECT_EQ(AesiMP < 1152 > ::gcd(l, r), "59674012642333753734610734990062908255218798084295958207399353741208437901643855825698694924687087177682867048554982860053126011284434405257929800112718420288354143.");
    }
    {
        AesiMP < 608 > l = "151522770749964678436148842371800661308755610344045960075519436400073891748881928936553469657464872818171536225043815173126601503063626409691786995851470215979632324780679154730655790.";
        AesiMP < 672 > r = "4514938238985691056059937449130052459716751635657861474242995973951297034721954133037737289305067035195170735945468656730231522190063989575281386186648309581728125181761236624690442222030130768178150251.";
        EXPECT_EQ(AesiMP < 672 > ::gcd(l, r), "4229133045955358202624475552855554877230999193277482980445577535964404995334906375576954301791895915830118167844572669658078638157230085205387745107022847.");
    }
    {
        AesiMP < 1024 > l = "10161055883348445617306414199629648859830426505308297391754313664187847160675266540965795990197975433659180149745200523104355845902208422575476341128527085331464552322111254979977417879009045541643340064230263018335314580877376779848213893595846951498461946789311256474748049982898555317283085505307159210152.";
        AesiMP < 800 > r = "515948972769130273350064740382292518386639731561737879067985843825058515267023925632271320701891333029923206120916362550988508113742196058545378274549901860897247568719463063376425826010741684820636164653439302208357418930310455309849769048.";
        EXPECT_EQ(AesiMP < 1024 > ::gcd(l, r), "490401245335199464371290691377917776995274646975724529767288634194266756471725789065628095645498930126084359922466265539160809022929889821467096.");
    }
    {
        AesiMP < 576 > l = "23645447925774886002998325692466057758331797656108649612849474157052278882570831703052388929013756451995500099188687523236841329466817612566959955406702858238462676245919775.";
        AesiMP < 576 > r = "60816000044152499904165146100270719114033719350347442322922861655195500069491242575612516603138520663953323040606673746723265376030821313791031108379435240630461525006170393.";
        EXPECT_EQ(AesiMP < 576 > ::gcd(l, r), "54520923532656950320107140503388701513804601249740428671147261843615703871570377448914205428830873806278088396569978623965161.");
    }
    {
        AesiMP < 896 > l = "62740249742851067499143020245042328617106625052044053623630449680318838292966703181144755319509513852788182521150171625967343453454896254576974917366461752849745977986535555222653167262834252905884504542047821006142275248747466212944147243718426751861161658531058713046.";
        AesiMP < 960 > r = "6307646547349816689884051803903026367842299810293153129468125895296675160161238101702895573682320860203534872020273771631666436912319742029537889993380537496644585713231177661458650632306520115037431356069432182492230018348182496456483937267150780657278540729068328194793314032584899821508.";
        EXPECT_EQ(AesiMP < 960 > ::gcd(l, r), "232049247473897235401214546845887461904067903584451783555238505019868948999669972398822609257046574074937853184852486345389034.");
    }
    {
        AesiMP < 672 > l = "18616024909498069904742602316774939101747083989830144251003152932678679984570171984782926648554015163415750309950069214583731001348828780685130884681562232908559551829478297203129290587119289162337871808.";
        AesiMP < 928 > r = "1462071909563042148732205222521737429230349331185649480947631195688804705564655880411994861112165458656747887795123436208097853104815460645892547697704327872720101826330401970311658217181220712291318225641639537817645927799121466748636065994794000886184985190115390134079927333401.";
        EXPECT_EQ(AesiMP < 928 > ::gcd(l, r), "472900631175425196165697542378929353717689763966490017025295390429655469782190314228943.");
    }
    {
        AesiMP < 608 > l = "85212760240756899430548729250831581127013686936228448859685109651502483027511794418800582922480125191423391639832564937254028190082219477018479477785330638252967391181184897401046076.";
        AesiMP < 416 > r = "6756442613711296058022703182509007796007227156406805847900921410721341473167524903912074030144758186183657221774913606001268.";
        EXPECT_EQ(AesiMP < 608 > ::gcd(l, r), "68017067994476894939528303364.");
    }
}