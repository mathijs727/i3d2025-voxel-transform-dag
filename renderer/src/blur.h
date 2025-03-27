#pragma once
#include "array.h"
#include "array2d.h"

class BlurKernel {
public:
    static BlurKernel allocate(uint32_t halfKernelSize);
    void free();

    bool is_valid() const;
    size_t size_in_bytes() const;

    void apply(StaticArray2D<float> values, StaticArray2D<float> scratch) const;
    void applyEdgeAware(StaticArray2D<float> values, StaticArray2D<float> scratch, StaticArray2D<uint3> paths) const;

private:
    uint32_t halfKernelSize;
    StaticArray<float> kernelWeights;
};
