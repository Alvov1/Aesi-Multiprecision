#include <gtest/gtest.h>
#include "../../AesiMultiprecision.h"

TEST(Bitwise, Device) {
#ifdef __CUDACC__
    const auto kernel = [] __global__ (const std::pair<Aesi, Aesi>& data, Aesi& result) {
        const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid != 0) return;

        const auto& [first, second] = data;
        result = (~(((first & second) - (first | second) + (first ^ second)) >> 96)) << 48;
    };

    const std::pair values = {
            Aesi("123426017006182806728593424683999798008235734137469123231828679"),
            Aesi("8683317618811886495518194401279999999")
    };
    Aesi result {};
    kernel<<<32, 32>>>(values, result);

    const auto code = cudaDeviceSynchronize();
    if (code != cudaSuccess)
        FAIL() << cudaGetErrorString(code);

    EXPECT_EQ(result, "584507039455445099497252123895536881383910670336");
#endif
}