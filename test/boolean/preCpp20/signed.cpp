#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../../Aesi-Multiprecision.h"
#include "../../benchmarks/benchmarks.h"

#if __cplusplus < 202002L
    TEST(Signed_Boolean_preCpp20, ThreeWayComparasion) { }

    TEST(Signed_Boolean_preCpp20, ThreeWayEquallComparasion) { }

    TEST(Signed_Boolean_preCpp20, DifferentPrecisions) { }
#endif