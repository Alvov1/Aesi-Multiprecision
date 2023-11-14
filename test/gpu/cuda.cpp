#include <gtest/gtest.h>
#include "../../AesiMultiprecision.h"

TEST(Cuda, ComplexTesting) {
#ifdef __CUDACC__
    constexpr auto closure = [] __global__ () {
        const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid != 0) return;

        Aesi o10 = 1562144106091796071UL;
        printf("Were in thread zero and number is %lu\n", o10.integralCast<unsigned long>());
    };

    closure<<<32, 32>>>();
    const auto code = cudaDeviceSynchronize();
    if(code != cudaSuccess)
        FAIL() << "Execution failed: " << cudaGetErrorString(code) << '.';
#else
//    Sign: 1
//    3850364740 713592733 2672340566 959760 0 0 0 0 0 0 0 0 0 0 0 0
//    Aesi<512> test = {};
//
//    test.sign = Aesi<512>::Sign::Positive;
//    test.blocks = { 3850364740, 713592733, 2672340566, 959760, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
//
//    char buffer [100] {};
//    unsigned l = test.getString<10>(buffer, 100);

    SUCCEED() << "Everything's fine, just different compiler.";
#endif
}