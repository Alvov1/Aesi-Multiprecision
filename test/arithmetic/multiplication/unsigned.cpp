#include <gtest/gtest.h>
#include "../../../Aeu.h"

TEST(Unsigned_Multiplication, Basic) {
    Aeu128 zero = 0u, one = 1u;
    Aeu128 m0 = 10919396u;
    EXPECT_EQ(m0 * 0u, 0u);
    EXPECT_EQ(0u * m0, 0u);
    EXPECT_EQ(m0 * 1u, 10919396u);
    EXPECT_EQ(1u * m0, 10919396u);

    EXPECT_EQ(m0 * zero, 0u);
    EXPECT_EQ(m0 * +zero, 0u);
    EXPECT_EQ(m0 * one, 10919396u);
    EXPECT_EQ(m0 * +one, 10919396u);

    EXPECT_EQ(zero * m0, 0u);
    EXPECT_EQ(zero * +m0, 0u);
    EXPECT_EQ(one * m0, 10919396u);
    EXPECT_EQ(one * +m0, 10919396u);

    EXPECT_EQ(+zero * m0, 0u);
    EXPECT_EQ(+zero * +m0, 0u);
    EXPECT_EQ(+one * m0, 10919396u);
    EXPECT_EQ(+one * +m0, 10919396u);

    m0 *= 0u; EXPECT_EQ(m0, 0u); m0 = 10919396u;
    m0 *= zero; EXPECT_EQ(m0, 0u); m0 = 10919396u;
    m0 *= +zero; EXPECT_EQ(m0, 0u); m0 = 10919396u;
    zero *= m0; EXPECT_EQ(zero, 0u);
    zero *= +m0; EXPECT_EQ(zero, 0u);

    m0 *= 1u; EXPECT_EQ(m0, 10919396u);

    m0 *= one; EXPECT_EQ(m0, 10919396u);
    m0 *= +one; EXPECT_EQ(m0, 10919396u);
    one *= m0; EXPECT_EQ(one, 10919396u); one = 1u;
    one *= +m0; EXPECT_EQ(one, 10919396u);

    Aeu<128> v = "155224914605719640152163633212502418720"; v *= 962355u; v *= 1001198u; v *= 338858154u;
    EXPECT_EQ(v, "50679771797195426729761095894027301438286084702511325315200");

    v = "258903830809715461046644388545027615099"; v *= 16241149u; v *= 258437u; v *= 891971384u;
    EXPECT_EQ(v, "969305863305525489398176833147064537330199279185615868136808");

    v = "278591894897356870393812128666404025578"; v *= 23687421u; v *= 439335u; v *= 203434095u;
    EXPECT_EQ(v, "589801401965024557668211404438098883896805984392950035416850");

    v = "336415198298474883926636522622081339907"; v *= 13175255u; v *= 71049u; v *= 348585655u;
    EXPECT_EQ(v, "109774664392404671795993903853644959445871273530255287692075");

    v = "145689721046645095586916715201599251194"; v *= 28942069u; v *= 453217u; v *= 230271426u;
    EXPECT_EQ(v, "440052738981007700916727478306357433770997459241039208052612");

    v = "17660093344641252205173861916278821040"; v *= 4410170u; v *= 585718u; v *= 146958387u;
    EXPECT_EQ(v, "6703957813813119558703243574973014098380132117622684708800");

    v = "245771054938689913784377120276911877776"; v *= 19375652u; v *= 301194u; v *= 477291525u;
    EXPECT_EQ(v, "684568794567957944956857170830279704857763039053163471619200");

    v = "88234255224939877838051287024363866263"; v *= 4991314u; v *= 651490u; v *= 750609197u;
    EXPECT_EQ(v, "215364318640647719957910096817016126032555781842774174224460");

    v = "303953042911503654234771291929715596445"; v *= 15291759u; v *= 307780u; v *= 993331635u;
    EXPECT_EQ(v, "1421014804448333548212381441567955275713402957182200997126500");

    v = "53023330969596912071077110206621364690"; v *= 25084071u; v *= 982171u; v *= 855194760u;
    EXPECT_EQ(v, "1117164601943341692274121734560858251483459850980453311640400");

    v = "274181381932259188501040549406343468760"; v *= 19912036u; v *= 397753u; v *= 293033236u;
    EXPECT_EQ(v, "636332309394768239148003131181962777961389420250000257194880");

    v = "219675333473266762908227309927878799345"; v *= 14781729u; v *= 268329u; v *= 408135618u;
    EXPECT_EQ(v, "355613827659879893743379817687926332170620248663522386146610");

    v = "308952767656426288819112607400668482679"; v *= 6820121u; v *= 614885u; v *= 691003698u;
    EXPECT_EQ(v, "895279087496890645081824659963249921169843779025809646632070");

    v = "246096730783818426199783400653200100251"; v *= 7418038u; v *= 1008122u; v *= 902997352u;
    EXPECT_EQ(v, "1661860124620658354416974119294740306299036306534685643491872");

    v = "36697532330243577359839434275074092393"; v *= 31725091u; v *= 438145u; v *= 1034359848u;
    EXPECT_EQ(v, "527629722048714140463136422086584876069661226694334895015480");

    v = "267735189431368633718120423415223639942"; v *= 25032961u; v *= 601279u; v *= 926543028u;
    EXPECT_EQ(v, "3733870979475830622239809565656192779197005821356057469132744");

    v = "8100885418509442311042458047221702783"; v *= 32338542u; v *= 370193u; v *= 553472030u;
    EXPECT_EQ(v, "53675587406825740292055279063317030893729273799301899870940");

    v = "106232138831489009224113666111756947504"; v *= 5536840u; v *= 462993u; v *= 198393759u;
    EXPECT_EQ(v, "54028179032246536801136552569461418838850511962638616136320");

    v = "208507654442371152228489530909339207749"; v *= 28536164u; v *= 987098u; v *= 222888931u;
    EXPECT_EQ(v, "1309080544219810375260718461315753064410817550864750083322968");

    v = "165131864348896085055419056119566119816"; v *= 1467211u; v *= 946902u; v *= 1060261125u;
    EXPECT_EQ(v, "243243548508349442958375479097018996311304685431548608866000");

    Aeu<384> n = "245525853859545246570580693187446556259"; n *= 712851054066607701ULL; n *= 735933771600620855ULL; n *= 999251271716239204ULL;
    EXPECT_EQ(n, "128709163784951390725303665713635814858710080377159578410726546226186371949467674936961675780");

    n = "247919860717490240182460157774639369688"; n *= 919058885780618172ULL; n *= 1097469881115754307ULL; n *= 426810819220203784ULL;
    EXPECT_EQ(n, "106729060797922676183153861932607502833249486355381514108097565176327823624259258014558183168");

    n = "107312396877864222235953882687959430497"; n *= 108787082825833204ULL; n *= 134409650425195163ULL; n *= 59351582083026515ULL;
    EXPECT_EQ(n, "93130080404501258561669947749409186930953242485610083594368706379950307590820558182604660");

    n = "112230717860677830047327074993830609457"; n *= 1007179687759170917ULL; n *= 252558210761999755ULL; n *= 567658022617889968ULL;
    EXPECT_EQ(n, "16205669274730789727902790193940871859268070016806864513498549194237222466654079484140170960");

    n = "211327708409780812364956454482899528215"; n *= 491496232476028778ULL; n *= 389364357573569567ULL; n *= 509921096370525621ULL;
    EXPECT_EQ(n, "20622238743472565517081241822835763737375583016799358342635149745806888180250334982965445890");

    n = "292075963080226873936475051187106163693"; n *= 538199452581967366ULL; n *= 528176582531212069ULL; n *= 703208337383400356ULL;
    EXPECT_EQ(n, "58385126095176882914980217118697405461128585390661369434109788083211129174578768335545295832");

    n = "64477094249545239912013199071237308687"; n *= 1005782134456712708ULL; n *= 90662086837165097ULL; n *= 875557015217300692ULL;
    EXPECT_EQ(n, "5147774539841517437784676007911170649772239548179805724226214518619982891980299537661517104");

    n = "179763940604766575441840853523272423223"; n *= 547606161950308199ULL; n *= 548490636496727950ULL; n *= 1002101185265385882ULL;
    EXPECT_EQ(n, "54106781352548836345583774207136984005267395254540938002959905063190773678993769978369816300");

    n = "314968043774443037499507958603406075163"; n *= 381152998324802337ULL; n *= 201340827009798194ULL; n *= 378173448293968606ULL;
    EXPECT_EQ(n, "9140894895282242682555152562494076534080050002907951707431346629705926592307891773250252084");

    n = "74061063557924438053592736167322251534"; n *= 24974186694849881ULL; n *= 941242558035628043ULL; n *= 594464545843459581ULL;
    EXPECT_EQ(n, "1034924842835852845969616028548706508868877543326774524516496320009596646244097369492573282");

    n = "284248969432789157594065752591818662533"; n *= 599511508733105892ULL; n *= 239328717604749208ULL; n *= 1123804775618927748ULL;
    EXPECT_EQ(n, "45833403723868939198488292969342722866499306580846783095147334992557539774881525658376778624");

    n = "61258161073692945848286163331065499650"; n *= 76532562191440292ULL; n *= 1131774466347155913ULL; n *= 541971212385618176ULL;
    EXPECT_EQ(n, "2875718154827435664620603868803244291319585015864472257135557138452746020714248247382886400");

    n = "124847071203517059256122264989868987139"; n *= 916533869098699217ULL; n *= 379430720009140075ULL; n *= 211450595185200957ULL;
    EXPECT_EQ(n, "9180541091273344320745151709544432282974962710490103430110091648563504465669594266546189325");

    n = "188517365298058684625384596192619559587"; n *= 303899900415865600ULL; n *= 949767132888648803ULL; n *= 59201718322406281ULL;
    EXPECT_EQ(n, "3221316284354950174919815717051357889853976400228636102732102425264406554644845572478329600");

    n = "178782561987518577432159842558731659191"; n *= 1088479704795633770ULL; n *= 994675261836535569ULL; n *= 578857412063376187ULL;
    EXPECT_EQ(n, "112046529125470530085259596454782388428012098489038122333281800890580434668498768360980118210");

    n = "12018068331851501865998824996543592567"; n *= 1057154686890431844ULL; n *= 1014268650566537465ULL; n *= 471408811411512359ULL;
    EXPECT_EQ(n, "6074687015991996447609239181128358377332946982191692502717116979621114166490923207735709380");

    n = "221850404636037819152043021871507306652"; n *= 572234491836537476ULL; n *= 194611504022317898ULL; n *= 1089428908584242322ULL;
    EXPECT_EQ(n, "26915450991888536853283603212158584785366252929148256635338745273220797384902923725243102912");

    n = "113274637848367592489493167198331078783"; n *= 688147182941593330ULL; n *= 208474324967709312ULL; n *= 332568574035739976ULL;
    EXPECT_EQ(n, "5404403957064417493429329621669070560961877840649792287502065127346826744470993392822343680");

    n = "88187788988037519744530008911613823354"; n *= 985009441840289609ULL; n *= 305141984509131201ULL; n *= 150327144276068249ULL;
    EXPECT_EQ(n, "3984632028144269956401874547395359283612610059409742294810095784021394695330292823998182714");

    n = "296461668338891633010545103863300784551"; n *= 117872020315701717ULL; n *= 420156174034752ULL; n *= 256453167963763529ULL;
    EXPECT_EQ(n, "3765287076020668026433116736042287489076891612432647896371226916780563404035138530099136");
}

TEST(Unsigned_Multiplication, Huge) {
    {
        Aeu512 l = "486143791826372061552403798267457643686865041183092807572875886569716133634822006720151308940414.",
                r = "281073537728109388456634287145369542469.";
        EXPECT_EQ(l * r, "136642155413195944187204934259270186109793432861737531247114872669712955919542946087335992860899183672589004539915511477937452363442166."); //446 bits
    }
    {
        Aeu512 l = "1858755136159271595128706340502926961788729607253577421827950583440452302336513678995555764935459.",
                r = "161088224621140627629201298773790076009.";
        EXPECT_EQ(l * r, "299423564889323574275901832044557273595440282459340824823979380537900633017252829892001885650097126349360609287606011148388956589303131."); //447 bits
    }
    {
        Aeu512 l = "176241566641672655159307086673688512352790973145146488652033958290771965502218712102091684460908.",
                r = "157019687428028165915217512704421183589.";
        EXPECT_EQ(l * r, "27673395705901435997233061261822572270207387011098161563470445831936936346488797280616736006772821782850888906519943591443532761638812."); //444 bits
    }
    {
        Aeu512 l = "842756005380142543407343028090178834331609985629580237638374212967042972658013745448166472685859.",
                r = "216880275372818220246511122236234846739.";
        EXPECT_EQ(l * r, "182777154518941588381267887988514383715637153442469971370042499828097780846749519154224939872949048927388326191656421167300130557563801."); //447 bits
    }
    {
        Aeu512 l = "519743465533934265373757755738335250473162657093494375969678452718533412228501400547891957960579.",
                r = "328313352423311435367256097383513464165.";
        EXPECT_EQ(l * r, "170638719569555780830437653486061624206347728451295232083449396519750993997115851879950455959692113446363376649831597185769200799151535."); //446 bits
    }
    {
        Aeu512 l = "2121572257097247963330157865016976103289157627451700871002582070164684017363233793181776845989171.",
                r = "311578167460696985278813816059166206833.";
        EXPECT_EQ(l * r, "661035596001815204054764448952759296861876076559109472423036090800480868531061484036144908000979836840145860337248215222781677864205443."); //448 bits
    }
    {
        Aeu512 l = "1493008929096422466687524913972771015799775677383582380375214299116679574700036199451180575764544.",
                r = "298954437841532674041111002246069243065.";
        EXPECT_EQ(l * r, "446341645090409693712488161242916414948344699364827828592655059154105518559889918183465914333632597120018222290324692612100225744887360."); //448 bits
    }
    {
        Aeu512 l = "466830983124302108525257009711977509516159684927132452727357158285163919373740988315327360209932.",
                r = "51235485932947929488201574667561577635.";
        EXPECT_EQ(l * r, "23918312268929432943573400426104804029581328563766050406345814278656327703600813500235531660547309497485965562225806266190130716070820."); //444 bits
    }
    {
        Aeu512 l = "712398241514271770279696489339141485849129412067120399422762549045379331398349990602900581363384.",
                r = "300697255121954010428279681682361611051.";
        EXPECT_EQ(l * r, "214216195777048387221814923299630899821556680713097460385552594202221753116406879679120466574270830421674532861810596093207212301156584."); //447 bits
    }
    {
        Aeu512 l = "1761106382609545892954396875699241315827370287474591303538144207779765774384595799464359293509542.",
                r = "316239635360962630053077835229960177682.";
        EXPECT_EQ(l * r, "556931640268306732373651202641366692803305125352524328201372717528630601484424609639696271146930019534802535625650944763312267682441644."); //448 bits
    }
    {
        Aeu512 l = "795617218485588047360180779460258241898990503634899801783045988532156283788168131776689368456210.",
                r = "98725507873944039296115796017979273817.";
        EXPECT_EQ(l * r, "78547713968244377825608966441419298450890739953309490516575830333650148366726602976381084404811673724001261483384352146387002164053570."); //445 bits
    }
    {
        Aeu512 l = "212463599689619965256789069496612668010728294127014414498895824590273018341484867060068922059948.",
                r = "67949883426067112114812453976017672700.";
        EXPECT_EQ(l * r, "14436876831192265302936824463417691872669932382545706989965870208409832467263641430514128937742653397925020331181803342992136843019600."); //443 bits
    }
    {
        Aeu512 l = "1110949195217617363051521695993408668756045892398897774049799543738735534688839282779746464258121.",
                r = "265484461544006878373084083098654089345.";
        EXPECT_EQ(l * r, "294939748895096927055400566829819271465775466459711812548198927539620902283299042088358816655233799137158489324170176229411518275820745."); //447 bits
    }
    {
        Aeu512 l = "1023709880157386472478049334808814200653923708129544559772420399955056926974757588547869123854548.",
                r = "186340845394290413515052684884674541217.";
        EXPECT_EQ(l * r, "190758964507015120239829896551975870896651693825467824961233099372093178392292002688590066522922049049077078958712743395814002538904916."); //447 bits
    }
    {
        Aeu512 l = "407630474782825340537301344167116045355594615600882077193444265177795398436850187394613827822828.",
                r = "226005168182586425298719388752622733677.";
        EXPECT_EQ(l * r, "92126594009639995836205857147295294382552771506379105845092587704677106207665204590851155840706835290777655743716538031025810584978556."); //446 bits
    }
    {
        Aeu512 l = "1238631684804221853584775400868633822243251422799584281662865415808771108311520638570596297551684.",
                r = "137756606981451924006134672754257953909.";
        EXPECT_EQ(l * r, "170629698198348827206931294068905647013693962309408508095273495308529836559056700965353201584236294993503688990593054229032120017332756."); //446 bits
    }
    {
        Aeu512 l = "1947679989241466278456139151709006579001056331171266548678908630797836873421142664008573363299734.",
                r = "291849476435603812455085281058835268310.";
        EXPECT_EQ(l * r, "568429385124224399595156495090927696781176388146134907166245726509829574466004154202567534488542358981183660696115128585312956841629540."); //448 bits
    }
    {
        Aeu512 l = "339165061331529802908627215005268597537866622706593665934381971039624890887686197458217602127252.",
                r = "333095004332121171632327900568065128569.";
        EXPECT_EQ(l * r, "112974187573530062572146422357906568400003037195597817824241388021160695379085938803857905332727764373709155646281893088796295278662388."); //446 bits
    }
    {
        Aeu512 l = "134625245445245993596601732373082460832805770340747841197323089565924170456518487526875616037407.",
                r = "88981342523928417172477041157152313153.";
        EXPECT_EQ(l * r, "11979135077331367787932986386192829486409841940854928212391950060629700084348423302632559870776893490437920845860585987877373826114271."); //443 bits
    }
    {
        Aeu512 l = "1626216784493968308287409528317163318163946897537059493262431600320799238368634236379794220512673.",
                r = "7984502656771602990403998731007064392.";
        EXPECT_EQ(l * r, "12984532236278663307461369291125856602214795077923995774355269297152818950176061858273588938328451439565514191570931588670998963039816."); //443 bits
    }
    {
        Aeu512 l = "441468262626945956467471513426614801419637315995795077300144980380785978156169104560772895300894.",
                r = "94896364359165809310305477056426620086.";
        EXPECT_EQ(l * r, "41893733103254565524486860391407413797767221559779897366032950795557162230434815998809687561511281642321449538482513258107800394156884."); //444 bits
    }
    {
        Aeu512 l = "1387207090385100990870948898061325143626975565663049936577275695095923157729289617781676158687727.",
                r = "204278725757641952281181805693495123710.";
        EXPECT_EQ(l * r, "283376896785834477589019821422024569848378538108560439363638167079722632808123172612467791426014406089845587122307632647537827123707170."); //447 bits
    }
    {
        Aeu512 l = "2008499252531891485019757700367128109835190938722353974211159557903369508828534386273062554422096.",
                r = "123461529172317899104926044377285902481.";
        EXPECT_EQ(l * r, "247972389059044815552729733794350928679784696349947143264975300182177953554099214252927741870089192017164392505923103413131666767620176."); //447 bits
    }
    {
        Aeu512 l = "1466863103284886963106490577233194533674924141458488372092017081356745766555821857203620739932623.",
                r = "7936754835927239276638748921080399679.";
        EXPECT_EQ(l * r, "11642132828639564069681854742746067097306840302240192678267515486319240251761997945228423065887432190628640477126487499078108370828017."); //443 bits
    }
    {
        Aeu512 l = "208216060023756078185884230792638920085851819203764515050156944360506875591356438184991455703310.",
                r = "37295853462010750580295917318400806319.";
        EXPECT_EQ(l * r, "7765595663083241374324737998288084102611502529877426608595008241631303970773324935038607429316913144752772872050316765329475237215890."); //442 bits
    }
    {
        Aeu512 l = "1844279682457101648649866725698090711520780866674092172482132618762821651755911339028579920579829.",
                r = "56606657898636275171001899423407329776.";
        EXPECT_EQ(l * r, "104398509054254694456065520444542823754170398419450187743774282535397292662526132681929042925108835374114536170886618017186546536688304."); //446 bits
    }
    {
        Aeu512 l = "638809432360238866876393742874365704369182474455022642542559482408598544652212843089778890097605.",
                r = "182339538269077190315530375632942930360.";
        EXPECT_EQ(l * r, "116480216938497251762437060360179959058719556480968757355170398259566372487779611054993136271928888748209374953202619514721495117787800."); //446 bits
    }
    {
        Aeu512 l = "1107207239049099085183283071050112513266235457739563228304686130878283626525235892556649139284303.",
                r = "298083500804106247421106622576800053980.";
        EXPECT_EQ(l * r, "330040209931404385267486374355779073904325920161146568062446872427311499563193856117836943364777932643372836682970037701572508966675940."); //447 bits
    }
    {
        Aeu512 l = "1153978691020077816507833939217129291578940912564870702182397808225125015923563874600906187527231.",
                r = "287834949043143600775900670048323582060.";
        EXPECT_EQ(l * r, "332155397726637652239830408863284639159604255507200041716936028883606351998837435439601030483101477939799082463052616048103895713075860."); //447 bits
    }
    {
        Aeu512 l = "2006846607715883676035261901918026553955610676569384381543726380501915009776716053860002846516413.",
                r = "137979696983051022438224553440019756887.";
        EXPECT_EQ(l * r, "276904086824101493582489914355523872920095775439964037478867300732236958688459778111948480384245290091927437590177144545669023115286331."); //447 bits
    }
    {
        Aeu512 l = "1647132495342472519760135144093699453988808456323082310045284819249231126563515359285244617855079.",
                r = "103585435592016768124689720520014705579.";
        EXPECT_EQ(l * r, "170618937007815546500705819340632204161964603409892047449459322599272081344388671446293881369912882393349439098489684920783272674785741."); //446 bits
    }
    {
        Aeu512 l = "916130358388194963348575264768138096731958034574275863672873611687621928697184193080281294060625.",
                r = "121449297808276155014774116531334990215.";
        EXPECT_EQ(l * r, "111263388727090654967942470896152460010895541486815311019097953888903135502107978257905483201310787575243575976320716871649721991784375."); //446 bits
    }
    {
        Aeu512 l = "1784998009793057576920391366529859928759354680768685887216295959879483929723383032797956653667735.",
                r = "27007820888009078408088751715077028024.";
        EXPECT_EQ(l * r, "48208906533943573923707723669129005280589847911600339376337443480851739812177551438524748523424642021726057357886106313685202979605640."); //445 bits
    }
    {
        Aeu512 l = "1745338018481351807794818401453718384718377070942779164491367252124856278995995805011235478230436.",
                r = "251642690396234803519631418697050539359.";
        EXPECT_EQ(l * r, "439201554621480750578152333987951192141255843059422946975941661394070482981593879928571716846670381980644053433280674593729716689730524."); //448 bits
    }
    {
        Aeu512 l = "1728204636584519768575879927354860500795790238405164336696197548826203655043337393407429946225329.",
                r = "156736535357341862573993979645758903811.";
        EXPECT_EQ(l * r, "270872807126751726911709975863767575283443086395136500097602021157777527798378926530051153765615008228580156286038643471334132242828819."); //447 bits
    }
    {
        Aeu512 l = "1315884449134018323506480548882170946136288245534806041383920890072107784673406840185647973777392.",
                r = "309524081477903137844959168775378973292.";
        EXPECT_EQ(l * r, "407297925449263574703037010001085009795241231861485865531633388594655642404951879516889304160364911757862263221492614111110347921414464."); //448 bits
    }
    {
        Aeu512 l = "1245307410848722571639023955214315562703880521531162194928773261599232355356012700533047782057591.",
                r = "27662781645311466819708484699139900722.";
        EXPECT_EQ(l * r, "34448666987596388565249648389931622948390020736976633583034005990948628709153230714510111301556809821907078384842462129870464626480702."); //444 bits
    }
    {
        Aeu512 l = "461051232121913132694037989949914485599472877298662229999464813670562601779374369520213891872298.",
                r = "319542434356581025928909198854071868973.";
        EXPECT_EQ(l * r, "147325433075337228513437317075113344492746928720663153602329569411363393791821996687089340580774168023280524211787657531095687104409954."); //446 bits
    }
    {
        Aeu512 l = "104941228485736575857898258251650028037415837438269705834977969308453773277938376971572003519962.",
                r = "209269969658279515912662824358467565925.";
        EXPECT_EQ(l * r, "21961047701112671258819332464414383710500982556179364645981130673371655718403328147241492227923681543052729125440948315364310288494850."); //443 bits
    }
    {
        Aeu512 l = "2005777512028850032090022970682663321279403756957666984516895734064255823141178830701372324327188.",
                r = "154337809253955217800201157978008887815.";
        EXPECT_EQ(l * r, "309567307057381523668386043986447685147223567943301221915169702677883417509384925928059273386528387880484310896462407465970604046414220."); //447 bits
    }
}

TEST(Unsigned_Multiplication, HugeAssignment) {
    {
        Aeu512 l = "1108781013101938200657737262728024978173660561102415969269244112185365001676023344273576514666251.",
                r = "53745970690819625491917796337191627867.";
        l *= r; EXPECT_EQ(l, "59592511832714061698034874327639595574096769440613301734108918602619546531031836363256697016669016857330990906559505077289374896016617."); //445 bits
    }
    {
        Aeu512 l = "1424709839301206646552659294922914193805685762974383848888744080093202502775719758847666838815681.",
                r = "320866865738518835884743148866387344316.";
        l *= r; EXPECT_EQ(l, "457142180723407019390781461525626865431395499237708951084834314230534552870148004072161440729692887132500583714107216578044688207019196."); //448 bits
    }
    {
        Aeu512 l = "1365910109571458662854026817194753790068743554751036002333992007880555842562012567562743744925945.",
                r = "66293790902230853797295419758727035228.";
        l *= r; EXPECT_EQ(l, "90551359195173514957940142477970399487505653848344716668526197354670312047588118658361766712227373300472693205202046202719118266190460."); //445 bits
    }
    {
        Aeu512 l = "1170195443371595905857952117359832130811960946374969051282051805581503910286517995271656633524920.",
                r = "278375834113271141965816141279995731005.";
        l *= r; EXPECT_EQ(l, "325754132624117156398088109110156483164167504313508000999231564849724864321031184080650797509993457611133741883984094394887365284144600."); //447 bits
    }
    {
        Aeu512 l = "1542332819803769527429840232318978312667486301006361504682733949711941110507075814836702754602498.",
                r = "12407096319612516602140361411096193344.";
        l *= r; EXPECT_EQ(l, "19135871852204943664254907505863000174927241125547371740683322329150617637606053736473755843290771139143227427186056615303903673373312."); //443 bits
    }
    {
        Aeu512 l = "1901080738028568585139686762409438427606699257340289630765368384591648707011127619722459539715458.",
                r = "135794930204698402640618742674793156370.";
        l *= r; EXPECT_EQ(l, "258157126134085999384539375967940588463340423838282281240886075698241906014942220029641002114545850611674362001935278098047275500167460."); //447 bits
    }
    {
        Aeu512 l = "1303405106226413317345810729044639497444433769664563070519837158587200785133559913348025225480041.",
                r = "146951016549910358484794065794122353781.";
        l *= r; EXPECT_EQ(l, "191536705336315332235775347198735724685271926355710261906796034582217074780743822569721779467767585351766807791408845043038414556385021."); //447 bits
    }
    {
        Aeu512 l = "1492764465050840745537313586377742416255251910016682391056291186456407864092132185345449000950756.",
                r = "321207909598179272727286251503868076607.";
        l *= r; EXPECT_EQ(l, "479487753341424896623290010876135865653426329613987326938626021964100297722115019111080797408441672837794980214413786825623140042564892."); //448 bits
    }
    {
        Aeu512 l = "1596612556574457320707962776919308167204698088521594161511375983396196305144256494040584736709513.",
                r = "39261024148004771795038069328384587958.";
        l *= r; EXPECT_EQ(l, "62684644138677403736185888220803952188330525625707035027735062314708118373066205574104735911726902176113966336710730744134343243844454."); //445 bits
    }
    {
        Aeu512 l = "207018283999243491392408828135083459788615355609752607127939949892786502335572783656928931428006.",
                r = "32401801037719697457263031176020740370.";
        l *= r; EXPECT_EQ(l, "6707765249313638799215264534772336680631675009951557285508164145193503615367866884016871615140379922329006537857648436616577472802220."); //442 bits
    }
    {
        Aeu512 l = "720379674670468781603239332550799131220156330802080768222107208710297996523457316784431185086800.",
                r = "192884014858895792821395260460777911202.";
        l *= r; EXPECT_EQ(l, "138949723873185217665833231558068432588137372098945936200951135443794324501419986802226324724379928403920012195414025346005157062333600."); //446 bits
    }
    {
        Aeu512 l = "950477776357320190499004359536659924804657447983421866761820130630206038715107069636719436514619.",
                r = "142725552581056833746374636565502513724.";
        l *= r; EXPECT_EQ(l, "135657465846612680706194803815866249697595690706470760426640010323761471505138681275829094796618119113987706977719221645629877774131156."); //446 bits
    }
    {
        Aeu512 l = "2111575792355974246049390726254111529884898778311690752940421070130823957263371126002164785517922.",
                r = "209001808261765982269104602736070462631.";
        l *= r; EXPECT_EQ(l, "441323158884169908447483975918428423304444283179377520639621094978508110463650562574250557138140495567066394101680590860717735481772782."); //448 bits
    }
    {
        Aeu512 l = "763081891680025965154934115055227843930132820408320902551723978145575519183375826651978814070004.",
                r = "196194412682029020120514253674295072210.";
        l *= r; EXPECT_EQ(l, "149712403566454381232541294859362240485361890583211635661365738694603312400331942928997379459191767605558400529416918739735511174988840."); //446 bits
    }
    {
        Aeu512 l = "58782422191478701871837786405880238275801350411865136636878104303529620850557662210786041177657.",
                r = "240664727066145377934916702298669493052.";
        l *= r; EXPECT_EQ(l, "14146855592999149044085423084824191497560494788588391886842364652355146750513477392141251987770452373312586711130003012182813259139164."); //443 bits
    }
    {
        Aeu512 l = "144475716010377850372081375744780848321102672468113312892319322018290948780469988164769416037993.",
                r = "322198085105128334508477336651692594466.";
        l *= r; EXPECT_EQ(l, "46549799042736074917391934137055265132950268540595390846160711050416553554024936815288262265819534373594771772883250061095408597546738."); //445 bits
    }
    {
        Aeu512 l = "423568870106878069056151997172178508475279285946789301491483230799782059971562503413604329865435.",
                r = "19546136274032355379829503822085418452.";
        l *= r; EXPECT_EQ(l, "8279134856546948414149791301885821242231962260701510042667119847153156209998408414999572175358696967531642039906826875790172826006620."); //442 bits
    }
    {
        Aeu512 l = "98014939903645197338256706452520221993212299959088742342447106046446207970226587346263533086829.",
                r = "40191108127588669265189816703538742814.";
        l *= r; EXPECT_EQ(l, "3939329047786509470316830744644088279130227810458528381947793549861266888173475874378676764407749250832328171714901101705567361796806."); //441 bits
    }
    {
        Aeu512 l = "150620323655497783063821208633582203360120020153124396518058663653718301038012144005168173800236.",
                r = "315792266731356999573698650313561946081.";
        l *= r; EXPECT_EQ(l, "47564733422980276255204766956917937341649741972113309910442708362980365515615380273072951501817291920241691013896513179886837497075116."); //445 bits
    }
    {
        Aeu512 l = "1046181697308049185004450108589006299311264853163430871870751844351900285358119584019973199544697.",
                r = "249718514236316114411981023019437245747.";
        l *= r; EXPECT_EQ(l, "261250939072993436394991814746285352288320010026213608782595581674902947030815249132518696929175183694635571784875037520935144099653659."); //447 bits
    }
    {
        Aeu512 l = "1701220149655291498342189997770239237507176043226416303852565236923838073641179315264446320780777.",
                r = "63385163896322091200194978349275551943.";
        l *= r; EXPECT_EQ(l, "107832118009626247565873478821823418780701000451736076057420355211487564646616098217270416962362961149535473688019888032907517379399711."); //446 bits
    }
    {
        Aeu512 l = "554465248198703083319850277200952017552571422813989168220667033114031715533487260494666871636545.",
                r = "240996011339955081936110415697160110287.";
        l *= r; EXPECT_EQ(l, "133623913242505657335600458688540130899135082481045625235371807501155759705366125137626818601887100904630758097754323521633984379638415."); //446 bits
    }
    {
        Aeu512 l = "1065117284670958730624471735789574290506863090064016694552389465940550404687349341250779297495209.",
                r = "104956987923231445583006053362490638916.";
        l *= r; EXPECT_EQ(l, "111791501984034885209904584769118674847044315901201463264913335258471531858026145295317210237411237320456230986286287632020948858953444."); //446 bits
    }
    {
        Aeu512 l = "43053340797360576634083044309276885539801082583578156123768292760671096634405678371173205653575.",
                r = "225197505705712683078132893858021788341.";
        l *= r; EXPECT_EQ(l, "9695504959863601092939373502872870503616422188615581864140424680291858683469973304560531721744834382199858579242068439617193219969075."); //442 bits
    }
    {
        Aeu512 l = "510133563928521635936103480521550362499016219989645525363762347905758150482508547812920239073144.",
                r = "319719641881416418399157910758461437585.";
        l *= r; EXPECT_EQ(l, "163099720370917585925299099698977703187865000033331691928278673278750498306085287767492920634705508081292732674566688333398686205717240."); //446 bits
    }
    {
        Aeu512 l = "476914676204139396765122998122796924111718698344320131537789031715876332342628233278224301828014.",
                r = "262925948534118938696898874758418651128.";
        l *= r; EXPECT_EQ(l, "125393243610815553121748204624627058054825723695693666529891636349262474560373039129871910128365673815402086009751694783667922523099792."); //446 bits
    }
    {
        Aeu512 l = "132704478155505029776632388706337138867622895070051354544317480550958428842749404873801951835558.",
                r = "322782050844986383419391063445539971808.";
        l *= r; EXPECT_EQ(l, "42834623615347609356790223389914932924511437714430378245794966612317314981486201979179184819645296320388090551185970236743885171948864."); //444 bits
    }
    {
        Aeu512 l = "773473448069048272521482150713277555812013149615968431964106074407088765236231600816890335943404.",
                r = "247235747462671139415325302984440573076.";
        l *= r; EXPECT_EQ(l, "191230286075880698761515222141957743081925534541364140467392145296201210092384360975548883755704571748990857320572113264378794862190704."); //447 bits
    }
    {
        Aeu512 l = "1176424813523993534282685993159648817097925058607516360525420787891208963010534611072125357098126.",
                r = "210249974830529027444904587408996648279.";
        l *= r; EXPECT_EQ(l, "247343287433429445197238426172668846133650985858083687388970458685431453205695294656339379980693438750219634811793380282021515712025154."); //447 bits
    }
    {
        Aeu512 l = "672752119795328548806761637846483266207779717153452438537703870078193561059852917287098229557071.",
                r = "112681575846721751386934936618033707148.";
        l *= r; EXPECT_EQ(l, "75806769012760151644383921300724789004738564876622036754707633305881399722308932057794917154551095203209153013267509577240096166643508."); //445 bits
    }
    {
        Aeu512 l = "1249455070118306139378978975563857886059159536297614245814374551363723752159872698283035114278907.",
                r = "323723790232736973550226662892644158441.";
        l *= r; EXPECT_EQ(l, "404478331024208203426352136731719430240030926695527371511751245278633629791075206277236149905066943611471855935534320831181201572303987."); //448 bits
    }
    {
        Aeu512 l = "1794011418568163789053211231614732852834851974224104673713206080351826802981296884717296413971269.",
                r = "332785627223388424991882658255404061977.";
        l *= r; EXPECT_EQ(l, "597021215174127213990163896276755060838538541698066396730448612063605382705779921346257210387669138662872826769920028790799836373338813."); //448 bits
    }
    {
        Aeu512 l = "411108779963054457877263843580359569664353028314149920543508128222717670090843521285407771938024.",
                r = "120056337997013668162288161147386378472.";
        l *= r; EXPECT_EQ(l, "49356214640784386268953010106510843717181118756505667189008626887419102484857882511384647051100499274309058204861618739571866191819328."); //445 bits
    }
    {
        Aeu512 l = "279854019875708304237479917460359442014112180823525962921527272188470055371452757923457276489948.",
                r = "306590681954076717724849829041350900658.";
        l *= r; EXPECT_EQ(l, "85800634801283149078644819037969401190937044363841311272749795805217600724013063079385616163187517192947975575293532226935078683585784."); //445 bits
    }
    {
        Aeu512 l = "1151621947258197819188924753194543067754416056903646809958611312264506484260976733413001594243865.",
                r = "254356217553933091834117570636073729593.";
        l *= r; EXPECT_EQ(l, "292922202556690225395305502531195435855893632574654882365408194929135436950312579265946414005519170208539268604539685255600091309196945."); //447 bits
    }
    {
        Aeu512 l = "2008000351467665616068491330186799537335542302575977075581701295823544779257683502206370137746112.",
                r = "329139237792439406037882007922031276164.";
        l *= r; EXPECT_EQ(l, "660911705169017896884813430384477287943008860273562171110878488084167409740208185121153641628844342179738997233909546802972113989274368."); //448 bits
    }
    {
        Aeu512 l = "54367226215118882452669219041051271223591661826589370804265181107434514754288890666234936389840.",
                r = "115059042640002283219745818597271501944.";
        l *= r; EXPECT_EQ(l, "6255440999304013441115235201372795219809417896723728284336895452929530334340436953622558883288546376820692925779100954245037901848960."); //442 bits
    }
    {
        Aeu512 l = "177810349968904020184422622045069074549411296919510583188199127425240015371052717128987684712337.",
                r = "238548824210583032972932508929986004408.";
        l *= r; EXPECT_EQ(l, "42416449917554333374296663804237264931453202116917658288200100462981753282098124165001109771659216680914340801590414010661151493981496."); //444 bits
    }
    {
        Aeu512 l = "1102581143828197779287501136410619739895731833327930700806959170108420767718673324882490016921850.",
                r = "333841589673229864502860811587994769521.";
        l *= r; EXPECT_EQ(l, "368087441799333643706477546360520582911498313673622589287476985035374172790312028014157129421906852783811382369295686528896580618933850."); //448 bits
    }
    {
        Aeu512 l = "1141849852483110222788766524935286062282916894046476593061933823459027944943717441418665543252455.",
                r = "53870116916580409124658761738348168210.";
        l *= r; EXPECT_EQ(l, "61511585054445240690317118492649535450101908076237971073753355288432930879849814347704323328441612121057827468637638071349674835455550."); //445 bits
    }
}