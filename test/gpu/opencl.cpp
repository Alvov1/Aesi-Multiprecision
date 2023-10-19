#include <gtest/gtest.h>
#include "../../AesiMultiprecision.h"

TEST(OpenCL, ComplexTesting) {
#ifdef __OPENCL_VERSION__

#else
    SUCCEED() << "Everything's fine, just different compiler.";
#endif
}