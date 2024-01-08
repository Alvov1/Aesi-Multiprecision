#include <gtest/gtest.h>
#include "../../Aesi.h"
#include "../benchmarks/benchmarks.h"

TEST(SquareRoot, SquareRoot) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi512 m {};
    m = "6090725157803385683847790262910475439880944489553750045786895303294805776058380471608951628613013763332789951290275640362243141926540167423390151885627587.";
    EXPECT_EQ(m.squareRoot(), "78043098079224056927951137119052045281574520557910376004565433003118507400052.");
    m = "2198442897630760572542562253325024201444903498242188983498631852277766153592956295915984611974235359574640385959965624634128999962215308936301754007314365.";
    EXPECT_EQ(m.squareRoot(), "46887555893123290123289121097017676390793073607338116350101514251747344152489.");
    m = "1000283845438290419274324071760371197829217155422774112552376012708071485480283317843508475952441540071339503030773495309949929024012496228467055607230714.";
    EXPECT_EQ(m.squareRoot(), "31627264273697312931717788894381088477651234284028816841918262368868920653167.");
    m = "5835842694952909670019084191754362899884006062455030963065539151628101657144547117594407067421860388116505599752641932950422365837720731631815673944069745.";
    EXPECT_EQ(m.squareRoot(), "76392687444237158935595699183775595821743466989589796103628214600066114393611.");
    m = "3866404964809678884590261330471673378078414493316607179993947370598745698299238770431759940921769523654634371169220092627980180440637824026990197581713503.";
    EXPECT_EQ(m.squareRoot(), "62180422681175776039199472366607895973369526220983402320405142036733505240891.");
    m = "4560826394957611895281972689889828032619002380362353086899085752881078711579845744276935344290289683784503719672535533487574429157729131595128407383550127.";
    EXPECT_EQ(m.squareRoot(), "67533890713904614302701757348795319406033598443471427415486041122600729967809.");
    m = "3585126669048102094266502527182152113821800075415948154904611795480102030967683616483309824775786369945457886415807076849176955188808905018411674112744866.";
    EXPECT_EQ(m.squareRoot(), "59875927291759766902170390985122877245929448508816120373305077519115166470064.");
    m = "195432114288329197216777722466904127138981632626371213320685733903863966967954168812482349483814468962135873701539352896103173226813264512374360865490203.";
    EXPECT_EQ(m.squareRoot(), "13979703655239949227726320742062170323174505178595123308382537469230355248984.");
    m = "4885509786573540813014271044300328939442823484053161179977119277898839990932508382188884747708651211077918132443443944721575404160937901109543098897192323.";
    EXPECT_EQ(m.squareRoot(), "69896421843850782437988692032562040550970659647609725782619203272587444407377.");
    m = "5585838662982006159561231182315944605990032602869142805481013882371552431835678855360566804500930632508690837392418040312217246485178364193757989985028150.";
    EXPECT_EQ(m.squareRoot(), "74738468428126128945198936220600870524261980512835819749878936880910218598326.");
    m = "3974800749371025941688149434350317847651389369795419901011348466760296085075163008194828993428420960869579459802914702306076517313500317878238285523270583.";
    EXPECT_EQ(m.squareRoot(), "63046020884517572713684487895050254752951548754790221637178476070995379084647.");
    m = "2273897147379510952670903435983726478426446351306817064325493127253999928114620214087767336326374159343743840960901825211724186393724969928264343687801857.";
    EXPECT_EQ(m.squareRoot(), "47685397632603536387498977789929936366899731244291373598733992037978793182194.");
    m = "6058733199744491524750070639631005770149157505468511458447374326702852864729386502860578270470905464654036198444571649876873696811448222867190503890802073.";
    EXPECT_EQ(m.squareRoot(), "77837864820050732971467883500691147356859079793940498398075568852368431562477.");
    m = "1825181170929106117784121032396530262014382942214885327452078008994423651970599468513800657419183302097109018524170571049692861695737305378276276238961246.";
    EXPECT_EQ(m.squareRoot(), "42722139119303309035519517007000251493949478040923615801467678859254838920089.");
    m = "6075764831927944734820358370247697303865549504951499992047158748628379282483018278526389001427598630984046440443262928410428428718042635113579647593538861.";
    EXPECT_EQ(m.squareRoot(), "77947192585287795821166684446105483156513686295271884067163466213004011047624.");
    m = "4004830024434636953260534689014481794638697852388914654101458283982418871832780024500880693024698136721203259468218443457509515483024781849151540382579844.";
    EXPECT_EQ(m.squareRoot(), "63283726379177743180646267299797936437730074348079395258824842775083793841745.");
    m = "6285638845584197260933539177937109311654195998315458005442998568340285309085303150318584626944224585756114797146644007556131054207450258513243196878208005.";
    EXPECT_EQ(m.squareRoot(), "79282020947905946155306439097732368208323476687623652536644717329208169174849.");
    m = "1591715254732792046271554151254011707271523857378421070227426743657664509115521914378597439975038082693302574409843611930659905532648217572134988783972202.";
    EXPECT_EQ(m.squareRoot(), "39896306279313528027576127123679783731457438661317537258160347888633455098208.");
    m = "4152123138537748208119491567993802001922490996344574352904852554639792876794903343267214470467358724211881863139604808366854129862547449532439586991839781.";
    EXPECT_EQ(m.squareRoot(), "64436970277456001819589426863815839347838931575823365304411747368127005854777.");
    m = "3247953627286684745970952994405009249179276055815782700408132310818135548301596000101696199401789814574488352005606473492556320207228117525708252348438228.";
    EXPECT_EQ(m.squareRoot(), "56990820552845918345554104705511630299390699694280194924355838722665971148650.");
    m = "4879775319652791718435483278790393331977921874866324265671557515774794409920640048547316003137999878012169906596949392776285882401633843252266642302322220.";
    EXPECT_EQ(m.squareRoot(), "69855388622874267983903615010388442530873933382033428082803065074289210057526.");
    m = "525803053338669671361834125505778223635092169276267065659380173232895761190177502285921574438173914942888511016152214630374157699165671809810946680866173.";
    EXPECT_EQ(m.squareRoot(), "22930395839118645083125118415231829992844557819387390757172245849880332114817.");
    m = "4334290803334574924774563383778114592040008942935142759360740416894321011553534421329973775435676474326522198140183596669676711667493513252710889597311071.";
    EXPECT_EQ(m.squareRoot(), "65835330965482165826606708393068129703573908303472335630320393544299914361365.");
    m = "4494330897373086601741336539164891478106093875984641278376170467081308860649526546425730529688361230199738125538355893634422638784156646016107466222780002.";
    EXPECT_EQ(m.squareRoot(), "67039771012236360425452587112083879727352832026664715817865354531219320762856.");
    m = "4909510064291890007986701314959525545085343439669677637524745730528601969117283762582438181258785487159761082768699492288861101615356412408593052549486905.";
    EXPECT_EQ(m.squareRoot(), "70067896102936400412185908830070038094171381020525736544627701873950301457751.");
    m = "4309049860317095930345932817895929869158824714777420937049291489804408232553991632400000046593886407203130406067551835015280609170269107469747222587657575.";
    EXPECT_EQ(m.squareRoot(), "65643353512119533778839084743587241350254146984599233979995995828880625948728.");
    m = "2815183315959571534328333003798938203992295790601303864349480258693322749533160295492522791249654065813587711866285995434966581656634932226191180498298887.";
    EXPECT_EQ(m.squareRoot(), "53058301103216370645516088373866164223784123497457847471308939067872641642125.");
    m = "4184860263603606187505821657132333624811545808661814236148222750233758380867808991661648584399968418659227868360294455960518442603673931161268026915012004.";
    EXPECT_EQ(m.squareRoot(), "64690495929491885350008967512291554906930240598671190466110126164525773366383.");
    m = "3626179759665635710024894729908975749929310670913138437843697802717405269318753497101352884994201707580987493633460658245393661537089901470476706585576636.";
    EXPECT_EQ(m.squareRoot(), "60217769467704760793090420382288205284909086897093775532775991407068572475902.");
    m = "1210412217259070228850414662144211880122237220938898411592655648252451736373500486335107930995793797685095207365686273369950811799215724156756984295918621.";
    EXPECT_EQ(m.squareRoot(), "34790978963792758282821680720425169316397664924467261128155497854070715310641.");
    m = "5304183900049677899434155872940835545001908862281144467394746404718288247305566418344505514638218320360739860741811447449909599229045321155974023091798730.";
    EXPECT_EQ(m.squareRoot(), "72829828367569821559420558783391366244804688716707570395943155683543681885445.");
    m = "3640585299333987128710539656247326601027384236906409606719943101559261518050830948964339951529015129486869923842545631907069889461277512779785277993208500.";
    EXPECT_EQ(m.squareRoot(), "60337262942016081549376915367861634870082502555913772148331064200251220185180.");
    m = "3322370697186985694789513400287270387301112233930212545202925835983912017769594900335591947550245455892079786111000094962594255136285982280844343511192695.";
    EXPECT_EQ(m.squareRoot(), "57640009517582365635109531157872118327321192156029885084868676476655818355537.");
    m = "363784768752142937563532516802248023200783991735884641019640603853335279495874238171768858622548171987239433433940329270027707838693346134632383823341935.";
    EXPECT_EQ(m.squareRoot(), "19073142602941523059626261036079470804203755080099072645277917290890123925763.");
    m = "6573219637626330602027647709989350575004966449566640867406801913893816685114013865954636777879975709920815941414116164213346809608268977636961264397247984.";
    EXPECT_EQ(m.squareRoot(), "81075394773175977437905239306818533094686679307548108118687932991147134727933.");
    m = "2651697164253891110974861916707961800399415237006671427054906096036127767551025689184240443309042522935877367416817011482578545456921736574292175068578268.";
    EXPECT_EQ(m.squareRoot(), "51494632382937652261828149358433557173471303538112140869567269588752896291463.");
    m = "747574725367714742179421620118183706234552472659906649927272598397868687230961900721483170976277920481053460135612564397505407018863990352790291461172926.";
    EXPECT_EQ(m.squareRoot(), "27341812766671392415294159584452864674807851989798208193274217983842152820397.");
    m = "1386941192648168135929395522078905161709959773925029007669200584596632075854980700020366355221969540218767669166559150892077370741807280531484664981086514.";
    EXPECT_EQ(m.squareRoot(), "37241659370229035699777185636279191613762937271489308117700674762048976617383.");
    m = "5621106178103901219344046536326204325148496347375026526085835464563757533577400230588398787642778757593635030148118674346481873791016817115783785604037128.";
    EXPECT_EQ(m.squareRoot(), "74974036693404078836937829975059038773972455949610022517963046206798417449826.");
    m = "5955813785675588109518975395472748958137987074475981158876199212009943507928308334573693535545462979243470395215093466596560400578633978166204064517715396.";
    EXPECT_EQ(m.squareRoot(), "77173919076820169671684270220139975015863766388260649610600962754358359900654.");

#ifdef NDEBUG
    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(),
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}