#include <gtest/gtest.h>
#include "../../Aesi.h"
#ifdef __CUDACC__
#include "thrust/device_vector.h"
#endif

TEST(Cuda, ComplexTesting) {
#ifdef __CUDACC__
    constexpr auto closure = [] __global__ (Aesi512* const number) {
        const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid != 0) return;

        //(2^502) + (3 * 2^498) - (5 * 2^496) + (7 * 2^494) - (11 * 2^490) + 987654321
        *number = Aesi512::power2(502) + 3 * Aesi512::power2(498) - 5 * Aesi512::power2(496) - 11 * Aesi512::power2(490) + 1234567890;
    };

    thrust::device_ptr<Aesi512> number = thrust::device_malloc(1);

    closure<<<32, 32>>>(thrust::raw_pointer_cast(number.get()));
    const auto code = cudaDeviceSynchronize();
    if(code != cudaSuccess)
        FAIL() << "Execution failed: " << cudaGetErrorString(code) << '.';

    EXPECT_EQ(*number, Aesi512("14848534544607010728721939591566599669967092093623081979638562895318888168261438776789833573099438290982293601340348173669740387359021143293664083994801"));
#else
    SUCCEED() << "Everything's fine, just different compiler.";
#endif
}