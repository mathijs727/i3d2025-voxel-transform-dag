#include "tracer_impl.h"
#include "transform_dag.h"
#include "typedefs.h"

__global__ void getChildMask_kernel(uint16_t const* pNode, uint32_t* outMask)
{
    *outMask = TransformDAG16::convert_child_mask(*pNode);
}
uint32_t getChildMask_gpu(uint16_t const* pNode)
{
    uint32_t *pOut, out;
    cudaMallocAsync(&pOut, sizeof(out), nullptr);
    CUDA_CHECK_ERROR();
    getChildMask_kernel<<<1, 1>>>(pNode, pOut);
    CUDA_CHECK_ERROR();
    cudaMemcpyAsync(&out, pOut, sizeof(out), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    cudaFreeAsync(pOut, nullptr);
    CUDA_CHECK_ERROR();
    return out;
}

__global__ void getChildPointer_kernel(TransformDAG16 dag, uint16_t const* pNode, uint32_t childLevel, uint32_t childIdx, TransformDAG16::TransformPointer* outPointer)
{
    *outPointer = dag.get_child_index(childLevel, pNode, *pNode, childIdx);
}
typename TransformDAG16::TransformPointer getChildPointer_gpu(TransformDAG16 dag, uint16_t const* pNode, uint32_t childLevel, uint32_t childIdx)
{
    typename TransformDAG16::TransformPointer *pOut, out;
    cudaMallocAsync(&pOut, sizeof(out), nullptr);
    CUDA_CHECK_ERROR();
    getChildPointer_kernel<<<1, 1>>>(dag, pNode, childLevel, childIdx, pOut);
    CUDA_CHECK_ERROR();
    cudaMemcpyAsync(&out, pOut, sizeof(out), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    cudaFreeAsync(pOut, nullptr);
    CUDA_CHECK_ERROR();
    return out;
}

__global__ void intersect_ray_impl_kernel(const TransformDAG16 dag, Tracer::Ray ray)
{
    Tracer::intersect_ray_impl<false, TransformDAG16::levels>(Tracer::s_transformTraversalConstants, dag, Path(0, 0, 0), 0, 0, ray);
}
void testRayTraversal(const TransformDAG16& dag)
{
    const auto traversalConstants = dag.createTraversalConstants();
    cudaMemcpyToSymbolAsync(Tracer::s_transformTraversalConstants, &traversalConstants, sizeof(traversalConstants), 0, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    for (int x = -10000; x < +10000; ++x) {
        for (int y = -10000; y < +10000; ++y) {
            Tracer::Ray ray { .origin = make_float3(x, y, -10000), .direction = make_float3(0, 0, 1), .tmin = 0.0f, .tmax = 100000.0f };
            //Tracer::intersect_ray_impl<false, TransformDAG16::levels>(traversalConstants, dag, Path(0, 0, 0), 0, 0, ray);
            intersect_ray_impl_kernel<<<1, 64>>>(dag, ray);
        }
    }
}