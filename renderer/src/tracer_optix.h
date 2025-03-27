

#include "dags/transform_dag/transform_dag.h"
#include "tracer.h"
#include <optix.h>

namespace Tracer {

struct PathTracingOptixParams {
    TracePathTracingParams traceParams;
    OptixTraversableHandle accelerationStructure;

    TransformDAG16 dag;
    TransformDAG16OptixParams const* pSubTrees;
    TransformDAG16::TraversalConstants dagTraversalConstants;
};

static constexpr uint32_t OptixLevel = 5;

}