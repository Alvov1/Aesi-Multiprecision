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
    SUCCEED() << "Everything's fine, just different compiler.";
#endif
}