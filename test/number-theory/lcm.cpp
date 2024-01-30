#include <gtest/gtest.h>
#include "../../Aesi.h"
#include "../../Aesi-Multiprecision.h"
#include "../benchmarks/benchmarks.h"

TEST(NumberTheory, LeastCommonMultiplier) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi512 l, r;
    l = "58712898820445740578867938422005479885576561870284639903702251100561617484082.", r = "73271804942217305270802872124823125802579882393902100324004182532671979224332.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "2151000034981920403783672172121095062306443558541639433694713477795867196388687757277738437332293774445519756953714546194412172547652310751074355858541612.");
    l = "58863911179190489819692830244173244788660412955385050676183232551129080766029.", r = "105375481318482303482984022544313244487626288238837961103587526270381302447504.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "6202812972795589079742096736721522400130110975236961129038381052586061958039544291539939807045347550560275179337231203317848935613892681396546548879041616.");
    l = "51179317342583050349271006941235258436562423597891372936653459797260933392939.", r = "55266959775209160722275083290265433628244466673192193348345153793565859031460.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "2828525272895202041230880808359147207292350525571131576620878294182683480307067497603285199319937587539052724646353930204162950835601374998251034142860940.");
    l = "71075386647667831614305770432842758110719315879940702411044679625395163499679.", r = "19420096351189427599978278289427428141531872150147996722364820346761839383196.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "1380290856895751819303257815743401126063149493711871378782649470858376433336517874981015753988319170212051169855950933148737622770725943367291022103994084.");
    l = "110586714992074394146797181411236431914736024870101772209158990293971672974877.", r = "90095745879493175545179332883522377921412063459448855381268994376520063760999.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "9963392571573872777568531672672907756489864250095926879343054945375174542156084386792768301088289123312138116438854386157796187363726347176234619459422123.");
    l = "11691177235736879303818495080235291888358288326351767545149940552417902396520.", r = "57310145382359497749659187466726695618964571980150051293443066384256833778972.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "167505766767753097676984757189955493590927815543907377031261814850393904194745608903133226573760775027732650873093525151933237508622422426990921695494360.");
    l = "105615611113159431072290378480579251303824661747610597766139746203576583490323.", r = "95656957444972818291875433691253730515002678350506517662396771667838884948422.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "10102868017776289964867815851564512328703697262377968657168197342353382876541080140561804576291684420975307946425058949659454937390477647631647586591120306.");
    l = "47002868626214988913237600103633202496143573592837114969748580732987325580191.", r = "78105707027577690029866421958102439343155310020165378058779103604976696174479.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "3671192286374870984553874760312039563932953730242512832577007818461769481904638603227850388014447305660981092016354322226961640509274605764129808842145489.");
    l = "34708440348962043432973562418387415970937447482781464822946551003959587922624.", r = "25121403956361813722140882710372438545003760017928341823926147443467147296187.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "871924750701563088383308626701062260248013885339132585626872889769549893994184911179818492052448862304566413787415451463250536309340711279500501766234688.");
    l = "86520702971885834671204927155726392078832481987566394172174953831231382120843.", r = "4837952285113070922346809892097950884570150462670547805193886078424030695558.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "418583032652424340293813608103236342105412226875924722005508870628883135682811849679110428767473414209356368424214062964415388272268151228092742499315394.");
    l = "96844348140155893293362701060183275908428275320641344045899104317411151822930.", r = "33872762713107516869512287179209686203721240394414580086116183334314487307914.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "1640192812328537920748305261803131688072711813799942254067530350216405642132543615251688134614297513182213505154414623233330740103997336371647594657834010.");
    l = "112562009055647422476858980542361171789938401410081319633291657953063544835694.", r = "93930563766922787981501364474417807514895857693567522501183298330195861637964.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "480591498606110478568470151153130963130385653661993605493289274531547199816202233734902597095423699040209855401609221922843799775208231268832508186031228.");
    l = "57732686192114257568587162558625193203532844970972275322250348432317602108605.", r = "63557454376452829266092812774041468018873431581692229362343720415521236010692.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "1223114189561790048770844346700130031028896190397466878906306625517113510192857857519164993323417314207428891013013248190023116609850936254031879175068220.");
    l = "56926535338106282990865267288925080081520385997290691237155694981263201896618.", r = "58612183323799845379130043631108275902248167582900961483328012532886276304499.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "3336588525225855672558779227154762810282816544445857742652530195543518942898196225038095333959447066870191134607652527076724793124916864040566728886284382.");
    l = "94857092396393060862336380007179474899216418662468210099879820123368583017782.", r = "82839225584132591954099525884722405464481402042292739851530518892703030325440.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "3928944037639856601974180993619706073033527230306079114549826642611710668375611333691209593035601747039017177112877132243366120923453099402803468383487040.");
    l = "100216969915146904450853143373708165101190869516291594230707101649368060613141.", r = "50572698133082630703059823421389444930568905645073009402019834158807388460433.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "5068242567330948021825003915237367225118821193727994909065737408287699452533576544023292300624004618792841716566900646647475556279679528916084937998350053.");
    l = "33278054850265842511823508642403196065765942220801842524458566320673304355960.", r = "17028816936137193738681789179955375002198329416999478562807743771826719385828.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "141671476008977365957153640791871756918944637607698908644326329425840711393747979564296937888723298669341869290007835934867161516473574150438642072833720.");
    l = "66723199105003358170426434325268668390905226712220871236617608648912156434243.", r = "2992881685545027601393900167232275080613534168184605413481102433259365469642.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "199694640602338927697291990618256408748563242434084575350060370495823113783426629993907545550337164214433992488158026414841252995941974359189407785751006.");
    l = "93579083085542199025056589105247868767616095558217192257416612696150076351629.", r = "58012737282568456836142715237492929601916625360368417403675113098352964424684.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "5428778762185205194065108035814957758050288181371725368479085568779482061315392326004575800760401488678970084239091499906979755650572453337817403671210236.");
    l = "9705229121684709192196245989541349934775298411310292551338495276136656231243.", r = "68928988145144834362648812763567267868272445686115794669115064955783333239170.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "668971623074519733010867468075100566328907437279793952066886573066672938669980291498699830468671777991216341938141127051678254590435315918257343745388310.");
    l = "41460550331568012785266416757870279990256147094359536766987081581827478513858.", r = "20047284242792633322684046815462369009052142801038743320039754034390180104180.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "415585718679777155815472939653204076385910762443168697210260444266230637560808572776542697317426770313010270341907192242998914083922892582247913006863220.");
    l = "102805589161881168617678502421149630686111842598862240236480632062998679924629.", r = "40734529556131787324939805904071270047875302368461145508437087307433721719659.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "4187737310250190205201149642664115112883047954426194387780669708454616221150904615605868665785866977770809398480651464656925951833754092865260010387581511.");
    l = "21038908898221815233929890132936155269053371391691638129626208967166503982728.", r = "6798285876510020288639788851603872281952760092917977456050867110948396156053.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "143028517219962358408844520321455161750572708646998975980264902346946621551550891448357373775252886386336201873874629799110545448556494940517750304652584.");
    l = "49103160083412815477140503017078322770823305089601406224250739694178093915301.", r = "115202092683830997572183448006778988558754277174658971324858041602440206129422.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "5656786798998313786363368340365737525802412187467154860198905277658517921060498206629665815913184760397845605525225471150159523077709917062198262712086022.");
    l = "19452603255167304115449260480936666780014721303187332227323296234395920133766.", r = "49825621582733576490578645089685592371834791570728762920738583295508508583323.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "969238048591017453384030451315876037969990718832897964371659026310364557014954805583971498612860033595495752274735092183847243344141062930882401316784418.");
    l = "94617532627429483695065110009946356127684397083548387554415344960096990953973.", r = "84279937098366922918120462947567563337582230368206365080742047087508603493875.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "7974359698242436896427718624521226012412803817478631504873087736287604748181001247936996779946915564609492549240959914106578228967319717784715937112415375.");
    l = "45817717205721929485761764914978225171622776456569427346940265647809580000433.", r = "54122354456243599988778051829439083813473888984913775428509711108695507685900.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "130513827946527017599883466323542060676929443639039424936386502334427482981915486158478981010874047525915278789928307244072940219031507910511372464631300.");
    l = "75590315832253936171897904171805487553837625848728936394570977088163335928766.", r = "100394120235167471591538898304342427819858690414500030374152205751939641346081.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "7588823256277584989590889368402178627767217045338443698866115919319425000569862444694273063113095883796487986281758700653870297945804318317363074569266046.");
    l = "70176423169192876784874404824532538701163439226713214104127598596270312163813.", r = "16607648561938125670799346464151342769206367277202815726222927933472212113424.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "1165465373327807443727667582469999714757320187826716805685932499081975188863529182296509370165409220702696001010228454433529504858092329319391351224325712.");
    l = "8720764263540820948538075000211372388622786639122083293654620961763476699218.", r = "110961702458500132874127368768157413968053611045363558724462620361461962525319.";
    EXPECT_EQ(Aesi < 512 > ::lcm(l, r), "967670849421737612524447125456492735431589633387181601120103714572698658380496264988047235569079380444672810675962950119507564413129327813249961872500542.");

#ifdef NDEBUG
    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(),
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}

TEST(NumberTheory, LeastCommonMultiplierDifferentPrecision) {
    {
        Aesi < 320 > l = "1428889428460167499315584507836389062688806223786162622539505586101940356860019810643119393548628.";
        Aesi < 384 > r = "37686189677831851764349235137567848340730893948168055139726293042862756252397017656166338203011546510825102809516478.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "26924699014799309305261222989301759825749407053895948012484824872796619478759411783758695207498023836908322939173365537843992922082255100083466369611484002201518595950799733525754088165961194294483229518630146092.");
    }{
        Aesi < 320 > l = "445585476497040275428792754199092245882117256190144061470899631735583560602363446124722321311115.";
        Aesi < 512 > r = "11584600468772085186548809762410644233133379730489671847396734272870161665705497396072163081488595394932743565493805527498541312095431267177664485822616351.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "5161929719905645721091899069963718053756393957402451931250513530788134121568470308624606114017656500971498638316159565803927540341412593710598623433087145569278412856969683509403203401804862651267619303691939307357060345658422643352938095973957041365.");
    }{
        Aesi < 128 > l = "305620482598245835119228405874136670768.";
        Aesi < 352 > r = "6271999313249005520885471454193195421506548847953467911460218230725712111416876104921847811252390153684439.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "1916851456971027520335557203151863010157534059895608810532283646768238240168682987090001784100848653729109392231802571077997512814134376307779152.");
    }{
        Aesi < 320 > l = "1837770559905153169087220399359752990754417457846184412690419054182283654937923779719863640650083.";
        Aesi < 448 > r = "136504835877519612817189808871499938081871306911540530573462312893806907768275342682609910873426579197810764102551241119801118913656932.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "250864568660390259170778259559786772271347897156553294936917596581238714907038688852414726850617701869525850430523883910623991474476412116677749345328970456255964161981520915173849574561421555770565926411540389127623567499319325356.");
    }{
        Aesi < 192 > l = "736745538918083971598635012752780845779336778127334084251.";
        Aesi < 448 > r = "236137152806546033963495875795062351404943552190049467644279299951246065349676293037908806771776531069013019803244832996703701732902727.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "57990997967680234268276528836891198950453264333239805016267543533494372539178761438801952587550697481270206883874927557591177090207982234568971054316433876480553223680145603111538846201884159.");
    }{
        Aesi < 224 > l = "20987765443050621571752605242244785856580884608015774485032635454524.";
        Aesi < 416 > r = "64925246197587632193302863526859179415854504442889274249761422073469402834877306916723026313899386062776685896614658762467967.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "1362635838527283474919742878156093156449185309273351864233320993372506331879982967457091433644652739958587196176751106704330735424524796842696943284523169209953849550527005370016296455035232708.");
    }{
        Aesi < 160 > l = "1314788776006130017721621445936096241905701642733.";
        Aesi < 640 > r = "3944444654705246436004675629488289041625535259075058089861445332694212491822049164757621999207407513920003292340582793212537190066020470748745023655314270383772905995252026829745010895354716350.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "5186111559583833118008247946957140928966707676416407075856865531766477721024256755847913093746017006460925177722339533923714980117730404902248679704751492706868803810563925520688919689788339109918998445374688835745566562289990106934253784550.");
    }{
        Aesi < 256 > l = "110399586982739171513359810869059310734225326557977183334542988597723463551241.";
        Aesi < 640 > r = "701518975508802776870268459014545016104492890889996046236526008102726918559585392170531744975512156852714786407059748480023113189013212237115990723675733917646187035405253304327671756753938302.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "77447405156726142714549806316304581899954243804137127951984817110733474726946564821930445066723855627184320142717513716378689678786495666636390613692128889599151039130686313780688438830823928195648701096132741220307535555646862724046211532191052223874123624577529532782.");
    }{
        Aesi < 96 > l = "46986841881701486685010196721.";
        Aesi < 544 > r = "1850030646886377033980072771117645731129758366901627442901625282252078133633872862798157869304067930592208210526959537444141119550083489269528394057052837520995365.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "86927097481552114534223037844952887127154979249884951534535030491981190278552783077727334193700708270395167843413393742858392715097567366663074561440196160988636803687106047360674946379198165.");
    }{
        Aesi < 96 > l = "37432449885296018646891067943.";
        Aesi < 608 > r = "474785937839241632872893099626320987784561403392170270532920426519368853391253382226229336540701190676293935967837120175609367625871135730724760006971570298762958393672988822199758241.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "17772400824410683099802218529550174414107428588690970970689363415741562774751533506516827360875738182130577641557623673793005913241120221771838079841022706382788306164418704389405643713713733825487692996905168263.");
    }{
        Aesi < 224 > l = "3364715296031669908140499951214133735077241047038308382972670181872.";
        Aesi < 640 > r = "1340112860391689483714360232296503479518714923091520466970255347910445541122497710271554956230354056150688985584447589453493401429561616255045531160189603568326365865488867385402600561760079136.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "281818639985541963020159888117651583573708273570980098595544706624234811759045513488752330586363278402666521951315590058380575324336285339521043290022826940467397923979957403758746056973750509219054746435471380882128140799100188083228231390066468352639538912.");
    }{
        Aesi < 256 > l = "58459790092021701394508470089135364827491339037462067537220849857589060558016.";
        Aesi < 448 > r = "700810673265953137558889418159425740745263315907468855822986582028086842518244552890381470838890913902029495154731308458789677139055053.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "40969244853376025080897767655566297681110746202326253231627776138585555694240516429989625367186258551449025606465851761331709773774010898743518152650919109934842002332591576664955719259631406725689668947124454848.");
    }{
        Aesi < 256 > l = "102604921090451980581136985858974342028181180519082939001561541583346238258440.";
        Aesi < 384 > r = "6297843124891467970932450305672396890332225656708639491750711684374438851930714243084886927944869720408528188031971.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "646189696869534588767284494039456514464962192367149355850873646325436200409030406695870506255979871881132954546043455788486158332166073552748360199761965876711586616535504175097931490080585240.");
    }{
        Aesi < 320 > l = "164478560530762243207335500907553685451566263201717994093412784352639388362101842969818304731744.";
        Aesi < 448 > r = "5319490946670577608061369874191711536817998258556740008972445270431702956059055458359886428179258995153358658245821341884101651831898.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "437471106832399173459083490947901295311581977344438861074328125329398011377033185036378999649234140661761231729350287049808878263725094838714768869553352455368164452187042064365982882007635154071947567235430347482945289536185056.");
    }{
        Aesi < 96 > l = "70235915751471253643981998332.";
        Aesi < 416 > r = "4334467509962420771132050113552011379747797773430954899496107692253698092306633544255417130308962006634014829681913759322516.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "76108823714302493016508623529433798263640842009603506607150818581273131443051354593315289942003876829085835133381501479898381625213245069787087040510828.");
    }{
        Aesi < 288 > l = "361256790755252514861386171795771684420236715678259357620152244344774664332258164739563.";
        Aesi < 384 > r = "3141997076702265320784863668039137145169069378014210867247433364574596340960664084566870763246633646121734094642612.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "1135067780491845349381569676015298955735143592957503511871076986659206982661981970997944129256027525651105128740901677059016486211243698453754376220204749343752947753257802119081962391252531520542058556.");
    }{
        Aesi < 160 > l = "946634033776095183313917225198261895455421216618.";
        Aesi < 608 > r = "431329519518825820299821291418785609121430182478018629601938221520443028858516275878797355054829005079497379652449591746136297065139538237025402817478759681709013419579565245309412856.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "204155601474405534107310030569598789014243432482075048800181701758440183889049210812956071919368206636463280792769053991269112587998498911401603294225214443561303353569872048234218434135641133688843919973721380119122270363385020504.");
    }{
        Aesi < 192 > l = "1844114173391421390979423241481847640598628768827078369521.";
        Aesi < 576 > r = "71861083688914837449188768964192244014641761931625820086276725769455894898934907878160916911598317313444448922231861553100328099855510337630854003043773057025158939609074144.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "132520042945994940064843964988270268498265138636092631709529566563101778615922966414129207924516464561156614716649336466170005637541564367684770204959035558233477757845179209188064357817790140820120035090104826891124940155918765024.");
    }{
        Aesi < 192 > l = "5299229054500554625121192511853312859523532077992743228522.";
        Aesi < 608 > r = "770471062032868455419038686290226753515984268772178504426631523011878517911913066952323577807975081815468900902522009220457806472517268069227185299429158699022691044513423254242445060.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "2041451318788237837769631540916910086862411044828582684239761651236631689305463275660846247897174473521257659584675800550964128342448899665524213311651509628378672876514120410471068614823949612127144441856574928819732539901553081095805000660.");
    }{
        Aesi < 192 > l = "2879878003743117154578359632132509661689455731246921824735.";
        Aesi < 640 > r = "3961473525855131383612150297550357803535448068571393953913076252505955031196463560085793829025041484426974822895258708072648124970251885054025178797125506130889763696203443724383446519861816018.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "3802853489840294523603517782215791127052460161409713367238824009250547260898357013121715419835890999016759260930484127233604562998025422118523451743127789193814027922887165005727986473730659730393920057956766242030789514751976435730088063071803868410.");
    }{
        Aesi < 256 > l = "82338319530068144072624636475015227439533620323150727405791267951532216684209.";
        Aesi < 640 > r = "213691639252703915319983911070132194263429101197243967214390495271734717116033410728371393137326779737771256496203368143158732788579122164205896911554107624434484953606588781410804927459336552.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "17595010473693187214683411338234930321676797016490578027937992339897434239289232746386763130129570904309355152204868157331267139489641490263772399948359765521035724703340132550719870897454187693514929171143729675138950655282118949850819718813896924898571826384434907368.");
    }{
        Aesi < 192 > l = "3496689099380818433396281878889312361917659529211011466686.";
        Aesi < 416 > r = "39182023725984497021329813988369751998957880998246510906448523032167378677799834119862249774501538076786998243297400938900092.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "68503677627165295335053647287802481939144481656804652643193072241199190307220302893767294049029749454026842872130079727759677059083143163976478360841727050942959121892884942270167556.");
    }{
        Aesi < 192 > l = "4209759469076710452598294370916527042018268331320076890549.";
        Aesi < 448 > r = "47493678053254403877188463096562432456409900736907427690763947901229064620108246572845697171850854871803099337283593489773355798161737.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "199936960905968474526967271244104351154512369617441654200280591408366656070481646124927606978429151005273911570217203821125556217564857418975370724808160016222735817293137157675595829148723613.");
    }{
        Aesi < 160 > l = "36761795452760763959413179769039814100415903483.";
        Aesi < 416 > r = "165824316049752752761220383576514437930634226943088106567313678849229651001067095531502783247344752603588699762436264652980278.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "6095999587714964515426904336701200139875491840661852385637018922870083768118973053398398019595620952461507973405318672653578894238921738107559330640754519121004083950508274.");
    }{
        Aesi < 256 > l = "20502000005070124748943875670332768408145605290301146881830919926724552535338.";
        Aesi < 448 > r = "331206984434640621379769611312757764558751643463637667001064822711340452232638891267108849430797188216399819718318686694492529427279526.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "3395202798279131374166588494264984355264874049790251126112653765572286444648100058977948649070937210542909015436304577435140968177654148849923565081607561844862643968801286984658666088241690481535514831659444894.");
    }{
        Aesi < 96 > l = "33562356283677538908849792488.";
        Aesi < 544 > r = "50146589460452462515620400195367870547299823939120321826940238258163119614231417889357007831196540142197869437410169197775954481460907704677094654517983005797974995.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "1683037701883014550839527789801036132177267227082408976609345424567798626450952277060876730476305331173417129288307768466048537161053806908058118340538461109061560477300398450962385056362837560.");
    }{
        Aesi < 192 > l = "651747476140043414778140015459153856923590767607148580957.";
        Aesi < 448 > r = "369967480937830400094643361935006223992960983239415307725842173740425908938713894492953063443013975142860630301501791780711143348398453.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "241125371955120585565154164579882119455530541574278524918931394486973986071819783884952593272309201233370471994172423651158533247281883899447686630626076493783404755630700236180581197564059521.");
    }{
        Aesi < 192 > l = "5327691866944609449828768397433300072605439958605987230196.";
        Aesi < 448 > r = "551887659176002254442454407714321187585772295693616892225578571352447834574651440078400223292296330359321501930031019320573678860673468.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "735071848314771442873064261876397998791123167696296441525980057126747675214736069259622038235941225572459679014937936499077710761941643743515442333976339051754242477159583442038963966126409932.");
    }{
        Aesi < 192 > l = "1174681724141055414854997540388633347685533492168015644637.";
        Aesi < 384 > r = "11876159235381469984990137786310697750701997191444612254295117051723724238876261492865419592421498285751448276689294.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "4650235735597207842484404269999087008281938140907131020306521701191980335518244190189644578860015126373755028657158712451693462176313124337764807253710642999654157188805426.");
    }{
        Aesi < 320 > l = "2009968702278837526057151241692860897837685532905733600285477750051215084261388738778647252461209.";
        Aesi < 480 > r = "2038876764682567264168983741190897343271666198928438046832554117351517921573834067135703440711493675291968505649114697701286496696966769571875253.";
        EXPECT_EQ(AesiMultiprecision::lcm(l, r), "4098078484815494518988918928970371415570484750740385558683521371491536161647777095989028821989315177308012450600299021387444091565797254015441361414486134159012436672112591485508373278429950420503002623854285576158420392669943062729769560877.");
    }
}