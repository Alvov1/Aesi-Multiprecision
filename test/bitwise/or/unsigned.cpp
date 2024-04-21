#include <gtest/gtest.h>
#include "../../../Aeu.h"

TEST(Unsigned_Bitwise, OR) {
    {
        Aeu256 l = "0xe364c563d48f2ff4e91cb1b56c2ecbdfd4df17046bb46389338a9e4b515f58", r = "0x50a9e23dd74febe618a53ead68c6ace941bc2ef046680805423298909e581a"; EXPECT_EQ(l | r, "0xf3ede77fd7cfeff6f9bdbfbd6ceeefffd5ff3ff46ffc6b8d73ba9edbdf5f5a");
        l = "0x30c702b01287d1503dd4297f4647db62fbe5a5c1b1d3647aeed98282956ed7b", r = "0x25d4bfa039588c0b8229e4c472413e3a2ffc0b1a2e04efd83ab330451dea41b"; l |= r; EXPECT_EQ(l, "0x35d7bfb03bdfdd5bbffdedff7647ff7afffdafdbbfd7effafefbb2c79deed7b");
    }{
        Aeu256 l = "0x1eb9027640b1dbd280f92ad366d34e12787ab31ce273064aef70281dadb2ac8", r = "0x1ce570e9c16ed851370f3e787e03a9a8af27ac8586ad11ba70912e94338be7d"; EXPECT_EQ(l | r, "0x1efd72ffc1ffdbd3b7ff3efb7ed3efbaff7fbf9de6ff17fafff12e9dbfbbefd");
        l = "0x39ff241b3103c1a8e917ff3f73e05935d407eccbe63a8e94ee40a1f16f75ac9", r = "0x3e209fdf0a742f4950d81a68c59f5bb0913be6f3075763f625dc0543fdb6886"; l |= r; EXPECT_EQ(l, "0x3fffbfdf3b77efe9f9dfff7ff7ff5bb5d53feefbe77feff6efdca5f3fff7acf");
    }{
        Aeu256 l = "0x2565a5741ef15baf0e7a2568b4b4490e0bc021d37cf3a8753eafaca39cd2972", r = "0x2d92e35302c41c970a633ad12a2ce1ac25e59e0a7004540b476400a7da17f04"; EXPECT_EQ(l | r, "0x2df7e7771ef55fbf0e7b3ff9bebce9ae2fe5bfdb7cf7fc7f7fefaca7ded7f76");
        l = "0x22a116424b2478eed6d9f1ba5fe38bbec6d6daaa9acb5ed28760b98ec34bf23", r = "0x493288f60d85bbeddccddbc8f842c1bfbbf8957f2dac1161ffeb27196fc1ae"; l |= r; EXPECT_EQ(l, "0x26b33ecf6bfc7bfedfddfdbedfe7afbfffffdbfffadbdfd69ffebbffd7fffaf");
    }{
        Aeu256 l = "0x1d3311066ea8de28173deada772472ce6c17158faebae7aabaaf5a9147b334b", r = "0x1b442b9a15ad1aa9c4c361cdd3069a7f7cd99949928ffef89f6ec0186612a4a"; EXPECT_EQ(l | r, "0x1f773b9e7faddea9d7ffebdff726faff7cdf9dcfbebffffabfefda9967b3b4b");
        l = "0x162b1fc98d6736c4043e74f3f766a2117b5a18480e79b28ccd58400d582c545", r = "0x30f7ff5150942499b34af96a62d98f7c8a926c74e93160e20577d52eeb3787d"; l |= r; EXPECT_EQ(l, "0x36ffffd9ddf736ddb77efdfbf7ffaf7dfbda7c7cef79f2eecd7fd52ffb3fd7d");
    }{
        Aeu256 l = "0xa2160365d90af2d850e63a8ddad86723ddb1b45d2f3cf47fa836601c075941", r = "0x11477dba5545cc702f5555634dfd723cbb1dca95c66670dc3a8f8201edb5ae9"; EXPECT_EQ(l | r, "0x1b677dbe5dd5ef7daf5f77ebddfdf67ebfdfdbd5d6f7ffdffa8fe601edf5be9");
        l = "0x200bf6a43e18194862d4f86d2ecf45fa32ac50fa3f6b4faba60777e0bc043c4", r = "0x123c1face4fe872dcb74fe0b6e307afa22b8b9e89aef3adcf5225772ec59e57"; l |= r; EXPECT_EQ(l, "0x323fffacfefe9f6debf4fe6f6eff7ffa32bcf9fabfef7ffff72777f2fc5dfd7");
    }{
        Aeu256 l = "0x23a3b3dd40f8e8ff9362e299d0860abbeab36e1fb87d808e048f66183a486e2", r = "0x326aa32a63e79febf191fb2666f94029fd435719ef32a02ce902afd1bea81e7"; EXPECT_EQ(l | r, "0x33ebb3ff63fffffff3f3fbbff6ff4abbfff37f1fff7fa0aeed8fefd9bee87e7");
        l = "0xb68af8d8b4b9d54a05b7097036896addf91232a2ca6831d0139ddf4b3b08c1", r = "0x365b3dd276a850a6f19b1744ba468fa7ba12e5f90457fd7dfbfb7d941b0e499"; l |= r; EXPECT_EQ(l, "0x3f7bbfdfffebddf6f1db77d7bb6e9fafff93e7fb2cf7ff7dfbfbfdf4bbbecd9");
    }{
        Aeu256 l = "0x2c9094fa01cab6cb9c0419031b0ac9cb320bb079d32ff0b602208ee95db6b79", r = "0x2159253b95542091350905f49688ac444854d0f935f40013d08bd5ed919cc06"; EXPECT_EQ(l | r, "0x2dd9b5fb95deb6dbbd0d1df79f8aedcf7a5ff0f9f7fff0b7d2abdfedddbef7f");
        l = "0x57c516adc9e561a64702f03cc58468a1b4b82f3ec4b311cf135544fec2e044", r = "0x3411909fb94dbfcc9568f58129f2c6bf3e55ce01748c586184e6376367142"; l |= r; EXPECT_EQ(l, "0x57c51fadfbf5fbfecf56ff7cd79f6cebf7fd7ffed7fbd5cf1b5f67fef6f146");
    }{
        Aeu256 l = "0xc5670ca5aebae1dd995ddb3ae800ad70915fa3818313fdb582080824384c65", r = "0x27138e062b720b4dd1b5b6395801ab4d59244d03d2410aeafe0589f8df63f0b"; EXPECT_EQ(l | r, "0x2f57fece7bfbaf5dd9b5ffbbfe81abdf5935ff3bda713ffbfe2589fadfe7f6f");
        l = "0x2f284dbb69d4f6135884be7ddf0b1ff4c750ebe7b22f17c99c08cbcf1a5db36", r = "0x963e8d2a4d3ffdd0017986ccbb2b22a0b78d63efe10a14a2695aba4f7bd06f"; l |= r; EXPECT_EQ(l, "0x2f6bedfbedd7ffdf5897be7ddfbbbffecf78fffffe3fb7cbbe9debeffffdb7f");
    }{
        Aeu256 l = "0x22d50dcac43970d98c296c1088bc6e284e9a6423e77a476150142943c9acd3e", r = "0x1ef113e441b13436b9e7ec2ac42754e316fa77b3de0c930cf3e7bf3aaf1ced1"; EXPECT_EQ(l | r, "0x3ef51feec5b974ffbdefec3accbf7eeb5efa77b3ff7ed76df3f7bf7befbcfff");
        l = "0x2a221353fa69cb77ff298f60e0798911eb3c67118c748cfd99fbfb9a9ceaa1d", r = "0x1fe8c04af5004856f45bc8bc9dcfe7ff16a5b6617ca7c9a5c0ddaceb62fb62d"; l |= r; EXPECT_EQ(l, "0x3fead35bff69cb77ff7bcffcfdffefffffbdf771fcf7cdfdd9fffffbfefbe3d");
    }{
        Aeu256 l = "0xf3f9ff6e2601c3bdf43417f081f7a6e3b6478061d6344488a2377cac3a6726", r = "0x25cbd3ab2ae2ec3aa2ae141a071cb2c177644b104d1e93763e52292f6d0c806"; EXPECT_EQ(l | r, "0x2fffdfffeae2fc3bffef557f0f1ffaef7f647b165d7fd77ebe737fefefaef26");
        l = "0x31eb34517f2a33958a9f2f518c0429c544e9e0c682bac268cac9725e6fbdc24", r = "0x1d21ca68a149e4b9db61d399e03aa9ff4919480213681f7febaf9a5b0849d78"; l |= r; EXPECT_EQ(l, "0x3debfe79ff6bf7bddbffffd9ec3ea9ff4df9e8c693fadf7febeffa5f6ffdd7c");
    }{
        Aeu256 l = "0x3b80ec3ef24ec639f603bd107c836497dabdfec278d671416d2368a8643c4f3", r = "0x3101e619f25fc296c69bd379728280c0f28374582a3f632e80863b56167830a"; EXPECT_EQ(l | r, "0x3b81ee3ff25fc6bff69bff797e83e4d7fabffeda7aff736feda77bfe767c7fb");
        l = "0x752f9bb839f4a033a3ab6b3663b2145b251de0c546255094e06421e9a66ac7", r = "0x1b67330bbed4d736fd2f530ea1d989cea6c149c9ad2b2f3fd937b11d7c1d185"; l |= r; EXPECT_EQ(l, "0x1f77fbbbbfdfdf37ff3ff7bfe7fba9cfb6d1dfcdfd6b7f3fdf37f31ffe7fbc7");
    }{
        Aeu256 l = "0x297ef816b634947cbb2e4eaab082bb7858aa8aebfabc417da3bcdfaedeeab83", r = "0x223c9f2269cc1748af31e43a58ca4f3fc00523d7c626a7eb8f16588856e33a3"; EXPECT_EQ(l | r, "0x2b7eff36fffc977cbf3feebaf8caff7fd8afabfffebee7ffafbedfaedeebba3");
        l = "0x31f41aa0e7a3877cc4d8bfa57cf96e726afe4da1256185f14501ef80fb1fef4", r = "0xdcfc06bfc0953831e499248ab85a0a034947b297340bf99285e56ead7a473e"; l |= r; EXPECT_EQ(l, "0x3dffdaebffabd7ffded9bfedfffdeef27efe7fa97761bff96d5fffeaffbfffe");
    }{
        Aeu256 l = "0x340c1629ab5cbd62f08c2afb2ba7a3946cd1f116836f63448d278ac584b8c43", r = "0x39e2073a5ac097fcf2bc0ad502fcf9ce712436919da33f78e7052d9c0b24281"; EXPECT_EQ(l | r, "0x3dee173bfbdcbffef2bc2aff2bfffbde7df5f7979fef7f7cef27afdd8fbcec3");
        l = "0x10b09d2091ed41c2c51b987aec3d6f818b62fd05ba06e8fbe5bf47942cdf230", r = "0x3ed54cb232814f73aa77c803303337f4141d70811c9b28250d7704c0a113c99"; l |= r; EXPECT_EQ(l, "0x3ef5ddb2b3ed4ff3ef7fd87bfc3f7ff59f7ffd85be9fe8ffedff47d4addfeb9");
    }{
        Aeu256 l = "0x19100d6e850134e05c5ee78464e3d4b080c32a5d808c99b1836dd035b27436f", r = "0xca1fca05fa01bb27bd7024cfa4361bafa67fd859c0e355887a46c3de064af6"; EXPECT_EQ(l | r, "0x1db1fdeedfa13ff27fdfe7ccfee3f5bafae7ffdd9c8ebdf987edfc3df274bff");
        l = "0x169af752a257d7ae0891ed7373711470f260e78eb26a936eec1bc4a92adf810", r = "0x3dadfef1a97a77a996ee35b4313c2e876263bdad8428ed01cfe5ef97993b2b2"; l |= r; EXPECT_EQ(l, "0x3fbffff3ab7ff7af9efffdf7737d3ef7f263ffafb66aff6fefffefbfbbffab2");
    }{
        Aeu256 l = "0x33387a99995e8eb78235db3f4680b0fef68b9ec242e225771fd04dab51a5c2a", r = "0x19b9e6934f3ef83617261889d9064c67f4f3e6064aa4e06ed868e6451812913"; EXPECT_EQ(l | r, "0x3bb9fe9bdf7efeb79737dbbfdf86fcfff6fbfec64ae6e57fdff8efef59b7d3b");
        l = "0xea901e1b9540afbe3a067413e294bd7c7bdf1b0384db35d39d67c1edaf9c66", r = "0xc46411c01765c590e7475dc0c9e868330dd55ac8bda9e73880ce17ec6045cf"; l |= r; EXPECT_EQ(l, "0xeef41fdb9765efbeff477dd3ebfcfd7f7fdf5bcbbdfbf7fb9defd7edefddef");
    }{
        Aeu256 l = "0x322a5cbd5895d4e52487d9fe1d25dde22de72aa710a809ff9c6846f5b9cfc9a", r = "0x3b81b39a690137ddae0be5a2dc8ee6822eb95520d01665182faac0981e73235"; EXPECT_EQ(l | r, "0x3babffbf7995f7fdae8ffdfeddafffe22fff7fa7d0be6dffbfeac6fdbfffebf");
        l = "0x2f323ebe61312b220e85896cce4d33bbb2305870a5ce90655b3bcce7e346ccc", r = "0x302ac62839e50bc69147703b789b8f796e2bb43cd2563ad2c14a4e664e54222"; l |= r; EXPECT_EQ(l, "0x3f3afebe79f52be69fc7f97ffedfbffbfe3bfc7cf7debaf7db7bcee7ef56eee");
    }{
        Aeu256 l = "0x601d2b73dd90e3d31fa2b3fa6ebe47afff1f0a3e1d688a9580e6400bc9930f", r = "0x9d16622ef6a3e7098625cc160bf6f3ca93f6614a43d185ed62de02fa704a40"; EXPECT_EQ(l | r, "0xfd1f6b7fffb3e7db9fa7fffe6ffef7efffff6b7e5ff98ffde2fe42fbf9db4f");
        l = "0x204537bbe0b893d6e7fdb1c8d88a917e0d60f3ec567ef68198b6158c6d99719", r = "0x2ce3b72f0ac2d2a0d98d4e04d3b0767ff8d77dd5cd7f1b8121c7dcfe315688d"; l |= r; EXPECT_EQ(l, "0x2ce7b7bfeafad3f6fffdffccdbbaf77ffdf7fffddf7fff81b9f7ddfe7ddff9d");
    }{
        Aeu256 l = "0x1d66f0c56af9da23d5fdd90718ff25786c29c7f1882627c1518928ffbbc7723", r = "0x3dc56c5c6b03aaa633eeea5ea101a1635a11dc83165d7dadcfe162bd0f1b059"; EXPECT_EQ(l | r, "0x3de7fcdd6bfbfaa7f7fffb5fb9ffa57b7e39dff39e7f7feddfe96affbfdf77b");
        l = "0x1ea5c581760aa234728fff9c251c4e32ca48a5eee447eff4d51f03249a1e94c", r = "0x2c02c83efc7424590d6283759237fb5fd290179341f7d2627c91defdf6f4ed8"; l |= r; EXPECT_EQ(l, "0x3ea7cdbffe7ea67d7feffffdb73fff7fdad8b7ffe5f7fff6fd9fdffdfefefdc");
    }{
        Aeu256 l = "0x154d40d56e14884c29fa0e173c0a5669a50c21408de073c5529ce2b3a9a0334", r = "0x364fb7da6bcff6ab9c905256a722b86881bbd3e387adc3f3d905a93b44a02be"; EXPECT_EQ(l | r, "0x374ff7df6fdffeefbdfa5e57bf2afe69a5bff3e38fedf3f7db9debbbeda03be");
        l = "0xa4d0d87cc574e48685895b529e601fa121ca0eee0d07cc93fcffde939499ab", r = "0x6e1f4f00eeb75a5b112b891bd287c5554cc1a702da97c6010c68a06c591012"; l |= r; EXPECT_EQ(l, "0xeedfdf7ceff7fedf95abdb5bdee7dff56dcbafeedf97ce93fcfffeffdd99bb");
    }{
        Aeu256 l = "0x21230fe16c0fc90bf6462abe4cdb8babd61fc37acd4061d8e17f804bec6ec76", r = "0x1f1dce6f1c9b4b9f830fa40110d98efeb12b25e85eaf74d33041523a6dd8626"; EXPECT_EQ(l | r, "0x3f3fcfef7c9fcb9ff74faebf5cdb8ffff73fe7fadfef75dbf17fd27bedfee76");
        l = "0x4a0e347354457d84156a6822a961428a7a1c374fc7a87e93663cf9a50d441", r = "0x1fd0989fb55d6237d586648afef7c28a09ede60f5c42afe5d161ec2587fbd55"; l |= r; EXPECT_EQ(l, "0x1fda9ebff75d677fd5976eeafeffe3ca8bfffe3f5fc7afffd367fcfda7ffd55");
    }{
        Aeu256 l = "0x76eb19f48a28eabe549b512275433a394411c40120ee4aec159c91edd3c1fd", r = "0x3204e57407eddfbb7093377d362a1bb267604a6dfaf62a492ba078f9d687d08"; EXPECT_EQ(l | r, "0x376ef5ff4fefdfbbf5dbb77f377e3bb3f7615e6dfafeeeefebf9f9ffdfbfdfd");
        l = "0x391378e0e5d57d6d6552d15fb1dfbd0cf209833670cefaf9769644d0716dc8b", r = "0x22ccfe3e6cd8ad89b089654375f5ec896836e9928286ce09d1cd320ef613966"; l |= r; EXPECT_EQ(l, "0x3bdffefeedddfdedf5dbf55ff5fffd8dfa3febb6f2cefef9f7df76def77fdef");
    }{
        Aeu256 l = "0xb123646dbb621c974a79fd182bea835792a5658a58f06c31e766cc0979f808", r = "0x2ffcf2e44bfc50885ce22259132e85aa18f20f78d8e155c53623f804d44da3e"; EXPECT_EQ(l | r, "0x2ffef6e6dbfe71c97ce7bfd993beadbf79fa5f78fdef57c73e77fcc4d7dfa3e");
        l = "0x170e8d993bd467934c607195ff3a7dafd28a1c39144cafd26639b527746efd7", r = "0x3913eb652fc4b7d5438e44fe49d056a5e2c58a2ebbf2d6aca01bdcbaae5ea60"; l |= r; EXPECT_EQ(l, "0x3f1feffd3fd4f7d74fee75fffffa7faff2cf9e3fbffefffee63bfdbffe7eff7");
    }{
        Aeu256 l = "0x185d5aea34a0d4b068a7404eaba72dc6fc0901cc4211f5a42bff2facef901fa", r = "0x36f2cf9fe97d2b0be544f401322adf2c0aa6e8150ebecaf764ea125397d2ea1"; EXPECT_EQ(l | r, "0x3effdffffdfdffbbede7f44fbbafffeefeafe9dd4ebffff76fff3fffffd2ffb");
        l = "0x2164bf297f467dd4000d5cd0d491dfaa466f8e5990397890beac8e6375e5685", r = "0x38fda246f518ff61078915ca0b2034525c28724f39b39ef17b04b4608a2a34b"; l |= r; EXPECT_EQ(l, "0x39fdbf6fff5efff5078d5ddadfb1fffa5e6ffe5fb9bbfef1ffacbe63ffef7cf");
    }{
        Aeu256 l = "0x2adfcce325544ad4f1efa409003eb2a8b9794ede0dc98ff1607bc5c6462e91b", r = "0x1b52f75de030d578c6d77f6b5f7cc2bb619e60cb77dd31e7d2cb1e20cb37e74"; EXPECT_EQ(l | r, "0x3bdfffffe574dffcf7ffff6b5f7ef2bbf9ff6edf7fddbff7f2fbdfe6cf3ff7f");
        l = "0x3d92722778b1d17a05de238574a9d22e68a44bae35e4c2a83a7b9b7166e5b99", r = "0x24460c7f52cee8c2a48debddea989782360e48b9ba2ab90856b13a83efcfe79"; l |= r; EXPECT_EQ(l, "0x3dd67e7f7afff9faa5dfebddfeb9d7ae7eae4bbfbfeefba87efbbbf3efefff9");
    }{
        Aeu256 l = "0x3766feacd7dcdb5d7711cccd233feb7a76938384061c3407b61060aa4d95340", r = "0x305f5cb723e8b6bf1876bef170bdf5bd61ceba8f9911aadc3504fcbd0d2a622"; EXPECT_EQ(l | r, "0x377ffebff7fcffff7f77fefd73bfffff77dfbb8f9f1dbedfb714fcbf4dbf762");
        l = "0x1d43cd25cfd3d4bc872859906425c8ee841c8b0cd38bec1f09bd15bdb648520", r = "0x2b563071798b52f51d4602e8796281ca5549b6e341d2b9a54f5c79b4601c101"; l |= r; EXPECT_EQ(l, "0x3f57fd75ffdbd6fd9f6e5bf87d67c9eed55dbfefd3dbfdbf4ffd7dbdf65c521");
    }{
        Aeu256 l = "0x21fdf2488cc290071904bebd74611f5a846fc6e67e360087525d096545e0693", r = "0x282c711afca18c2e63eb8d0d3fc1a9bb69e34b90b5d94affb3190896317b10d"; EXPECT_EQ(l | r, "0x29fdf35afce39c2f7befbfbd7fe1bffbedefcff6ffff4afff35d09f775fb79f");
        l = "0x1e1e60c7525cf6fe9dc9c4752d8133ac89225bb28d0d232299671f5c572e7c0", r = "0x2ec0c5e95a7ed876fadee6e882376b13e6af56565cd65893ce5ba45de10339f"; l |= r; EXPECT_EQ(l, "0x3edee5ef5a7efefeffdfe6fdafb77bbfefaf5ff6dddf7bb3df7fbf5df72f7df");
    }{
        Aeu256 l = "0x380b85a04b8b4ef93aa67b72fbe1699cc6f83f6eccd9f3b40b0a39210fbcb80", r = "0x39b6bab45be9ea0d13a8c9cac8589dee3064b4a3a074af17971e9597e3130b0"; EXPECT_EQ(l | r, "0x39bfbfb45bebeefd3baefbfafbf9fdfef6fcbfefecfdffb79f1ebdb7efbfbb0");
        l = "0x147993a267f06084f2e5ffcacea64adfe18a42d348f69208d00f391fedd3cc0", r = "0x2bcde9538ab91eb37e7b8d9c2d8dbb1b5cf1ca5bf22c543fa71dd3dfd337aca"; l |= r; EXPECT_EQ(l, "0x3ffdfbf3eff97eb7feffffdeefaffbdffdfbcadbfafed63ff71ffbdffff7eca");
    }{
        Aeu256 l = "0x21b64055deed9b2ea3bce568f4856e541462eb6206a2c358c50047f738ae25b", r = "0x3a69b9fffa163a253ae06331d0ceba0a279bb34fa23d4d7c08e5f5b0b447784"; EXPECT_EQ(l | r, "0x3bfff9fffeffbb2fbbfce779f4cffe5e37fbfb6fa6bfcf7ccde5f7f7bcef7df");
        l = "0x1e470e91280bcb9cc6e73d117bb09f6fe0eff59ef00c69fd45e85dc748e0c7f", r = "0x303d52ac47c83591df39636bc5a138cdb7d7c467a87c169d4e84a45a9f35c24"; l |= r; EXPECT_EQ(l, "0x3e7f5ebd6fcbff9ddfff7f7bffb1bfeff7fff5fff87c7ffd4fecfddfdff5c7f");
    }{
        Aeu256 l = "0x58241c347d78861e07a51e4a7069de7c1052b25be5a4ee6069409b6605db72", r = "0x7c68037b071880953357c3b677ba0eee8554f367a9c28f47d01562798504ff"; EXPECT_EQ(l | r, "0x7c6c1f7f7f78869f37f7dffe77fbdefe9556f37fede6ef67f955fb7f85dfff");
        l = "0x364ba28462edd6e7be4bc97b3dc7e56faa2f93bce0e95571baaa64a9a451440", r = "0x145ef86e02dd77e08d01710c3f16ccaa86251ffdade79e787eff1e26a96809"; l |= r; EXPECT_EQ(l, "0x374fef86e2edd7ffbedbdf7bfff7edefaa6fd3fffaff7df7bfeff5ebeed7c49");
    }{
        Aeu256 l = "0x395800b4f2a4c19b1e7f22ce0452541f2d017c457244c5c82ad801d3f910d65", r = "0x279e8739e32e823a2719ba071d96fa7a6ba73daab946b8c99cbdb99123545d1"; EXPECT_EQ(l | r, "0x3fde87bdf3aec3bb3f7fbacf1dd6fe7f6fa77deffb46fdc9befdb9d3fb54df5");
        l = "0x1340c8f4d2f74d212050bb537db7b6ffb87e20051194910d1474309b8301269", r = "0x3bafd618d649db333508dd320dba11712548cee096c949b452f5fe1296faa0"; l |= r; EXPECT_EQ(l, "0x13fafdf5dff7ddb33350bfd37dffb7ffba7eacef19fc959f557f7ffbab6fae9");
    }
}