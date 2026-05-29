#include <AesiMultiprecision/Aeu.h>
#include <AesiMultiprecision/Aesi.h>

template <std::size_t Bits>
__global__ void unsignedKernel(const Aeu<Bits>* a, const Aeu<Bits>* b, Aeu<Bits>* out) {
    *out = *a + *b;
    *out *= *b;
}

template <std::size_t Bits>
__global__ void signedKernel(const Aesi<Bits>* a, const Aesi<Bits>* b, Aesi<Bits>* out) {
    *out = *a + *b;
    *out *= *b;
}

int main() {
    using Unsigned = Aeu<128>;
    using Signed = Aesi<128>;

    /* Unsigned */
    Unsigned h_ua = 100u, h_ub = 200u, h_uc = 0u;

    Unsigned* d_ua = nullptr, *d_ub = nullptr, *d_uc = nullptr;
    cudaMalloc(&d_ua, sizeof(Unsigned));
    cudaMalloc(&d_ub, sizeof(Unsigned));
    cudaMalloc(&d_uc, sizeof(Unsigned));
    cudaMemcpy(d_ua, &h_ua, sizeof(Unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ub, &h_ub, sizeof(Unsigned), cudaMemcpyHostToDevice);

    unsignedKernel<128><<<1, 1>>>(d_ua, d_ub, d_uc);

    cudaMemcpy(&h_uc, d_uc, sizeof(Unsigned), cudaMemcpyDeviceToHost);
    cudaFree(d_ua); cudaFree(d_ub); cudaFree(d_uc);

    /* Signed */
    Signed h_sa = -50, h_sb = 75, h_sc = 0;

    Signed* d_sa = nullptr, *d_sb = nullptr, *d_sc = nullptr;
    cudaMalloc(&d_sa, sizeof(Signed));
    cudaMalloc(&d_sb, sizeof(Signed));
    cudaMalloc(&d_sc, sizeof(Signed));
    cudaMemcpy(d_sa, &h_sa, sizeof(Signed), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sb, &h_sb, sizeof(Signed), cudaMemcpyHostToDevice);

    signedKernel<128><<<1, 1>>>(d_sa, d_sb, d_sc);

    cudaMemcpy(&h_sc, d_sc, sizeof(Signed), cudaMemcpyDeviceToHost);
    cudaFree(d_sa); cudaFree(d_sb); cudaFree(d_sc);

    return 0;
}