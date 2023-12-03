#pragma once

#include "EngineInterface/CellFunctionConstants.h"

#include "Base.cuh"

struct ShapeGeneratorResult
{
    float angle;
    int numRequiredAdditionalConnections;
};

class CudaShapeGenerator
{
public:
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionData(ConstructionShape shape);
    __inline__ __device__ ConstructorAngleAlignment getConstructorAngleAlignment(ConstructionShape shape);

private:
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionDataForSegment();
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionDataForTriangle();
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionDataForRectangle();
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionDataForHexagon();
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionDataForLoop();
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionDataForTube();
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionDataForLolli();
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionDataForSmallLolli();
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionDataForZigzag();

    int _edgePos = 0;
    int _processedEdges = 0;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionData(ConstructionShape shape)
{
    switch (shape) {
    case ConstructionShape_Segment:
        return generateNextConstructionDataForSegment();
    case ConstructionShape_Triangle:
        return generateNextConstructionDataForTriangle();
    case ConstructionShape_Rectangle:
        return generateNextConstructionDataForRectangle();
    case ConstructionShape_Hexagon:
        return generateNextConstructionDataForHexagon();
    case ConstructionShape_Loop:
        return generateNextConstructionDataForLoop();
    case ConstructionShape_Tube:
        return generateNextConstructionDataForTube();
    case ConstructionShape_Lolli:
        return generateNextConstructionDataForLolli();
    case ConstructionShape_SmallLolli:
        return generateNextConstructionDataForSmallLolli();
    case ConstructionShape_Zigzag:
        return generateNextConstructionDataForZigzag();
    default:
        return ShapeGeneratorResult();
    }
}

__inline__ __device__ ConstructorAngleAlignment CudaShapeGenerator::getConstructorAngleAlignment(ConstructionShape shape)
{
    switch (shape) {
    case ConstructionShape_Custom:
        return ConstructorAngleAlignment_60;
    case ConstructionShape_Segment:
        return ConstructorAngleAlignment_60;
    case ConstructionShape_Triangle:
        return ConstructorAngleAlignment_60;
    case ConstructionShape_Rectangle:
        return ConstructorAngleAlignment_90;
    case ConstructionShape_Hexagon:
        return ConstructorAngleAlignment_60;
    case ConstructionShape_Loop:
        return ConstructorAngleAlignment_60;
    case ConstructionShape_Tube:
        return ConstructorAngleAlignment_60;
    case ConstructionShape_Lolli:
        return ConstructorAngleAlignment_60;
    case ConstructionShape_SmallLolli:
        return ConstructorAngleAlignment_60;
    default:
        return ConstructorAngleAlignment_60;
    }
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForSegment()
{
    ShapeGeneratorResult result;
    result.angle = 0;
    result.numRequiredAdditionalConnections = 0;
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForTriangle()
{
    ShapeGeneratorResult result;
    auto edgeLength = max(2, _processedEdges + 1);
    result.angle = _edgePos < edgeLength - 1 ? 0 : 120.0f;
    if (_processedEdges == 0) {
        result.numRequiredAdditionalConnections = 0;
    } else if (_processedEdges == 1) {
        result.numRequiredAdditionalConnections = _edgePos == 0 ? 1 : 0;
    } else {
        if (_edgePos == edgeLength - 1) {
            result.numRequiredAdditionalConnections = 0;
        } else if (_edgePos == edgeLength - 2) {
            result.numRequiredAdditionalConnections = 1;
        } else {
            result.numRequiredAdditionalConnections = 2;
        }
    }
    if (++_edgePos == edgeLength) {
        _edgePos = 0;
        ++_processedEdges;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForRectangle()
{
    ShapeGeneratorResult result;
    if (_processedEdges == 0) {
        result.angle = 0.0f;
        result.numRequiredAdditionalConnections = 0;
    } else if (_processedEdges == 1) {
        result.angle = 90.0f;
        result.numRequiredAdditionalConnections = 0;
    } else {
        result.angle = _edgePos == 0 ? 90.0f : 0.0f;
        result.numRequiredAdditionalConnections = _edgePos == 0 ? 0 : 1;
    }

    auto edgeLength = _processedEdges / 2;
    if (++_edgePos > edgeLength) {
        _edgePos = 0;
        ++_processedEdges;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForHexagon()
{
    ShapeGeneratorResult result;

    auto edgeLength = _processedEdges / 6 + 1;
    if (_processedEdges % 6 == 1) {
        --edgeLength;
    }

    if (_processedEdges < 2) {
        result.angle = 120.0f;
        result.numRequiredAdditionalConnections = 0;
    } else if (_processedEdges < 6) {
        result.angle = 60.0f;
        result.numRequiredAdditionalConnections = 1;
    } else {
        result.angle = _edgePos < edgeLength - 1 ? 0.0f : 60.0f;
        result.numRequiredAdditionalConnections = _edgePos < edgeLength - 1 ? 2 : 1;
    }

    if (++_edgePos >= edgeLength) {
        _edgePos = 0;
        ++_processedEdges;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForLoop()
{
    ShapeGeneratorResult result;

    auto edgeLength = (_processedEdges + 1) / 6 + 1;
    if (_processedEdges % 6 == 0) {
        --edgeLength;
    }

    if (_processedEdges < 5) {
        result.angle = 60.0f;
        result.numRequiredAdditionalConnections = 0;
    } else if (_processedEdges == 5) {
        result.angle = _edgePos == 0 ? 0.0f : 60.0f;
        result.numRequiredAdditionalConnections = 1;
    } else {
        result.angle = _edgePos < edgeLength - 1 ? 0.0f : 60.0f;
        result.numRequiredAdditionalConnections = _edgePos < edgeLength - 1 ? 2 : 1;
    }

    if (++_edgePos >= edgeLength) {
        _edgePos = 0;
        ++_processedEdges;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForTube()
{
    ShapeGeneratorResult result;
    if (_edgePos % 6 == 0) {
        result.angle = 0;
        result.numRequiredAdditionalConnections = 2;
    }
    if (_edgePos % 6 == 1) {
        result.angle = 60.0f;
        result.numRequiredAdditionalConnections = _edgePos == 1 ? 0 : 1;
    }
    if (_edgePos % 6 == 2) {
        result.angle = 120.0f;
        result.numRequiredAdditionalConnections = 0;
    }
    if (_edgePos % 6 == 3) {
        result.angle = 0;
        result.numRequiredAdditionalConnections = 2;
    }
    if (_edgePos % 6 == 4) {
        result.angle = -120.0f;
        result.numRequiredAdditionalConnections = _edgePos == 4 ? 1 : 2;
    }
    if (_edgePos % 6 == 5) {
        result.angle = -60.0f;
        result.numRequiredAdditionalConnections = 1;
    }
    ++_edgePos;

    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForLolli()
{
    ShapeGeneratorResult result;

    if (_processedEdges < 12 || _edgePos == 0) {
        auto edgeLength = _processedEdges / 6 + 1;
        if (_processedEdges % 6 == 1) {
            --edgeLength;
        }

        if (_processedEdges < 2) {
            result.angle = 120.0f;
            result.numRequiredAdditionalConnections = 0;
        } else if (_processedEdges < 6) {
            result.angle = 60.0f;
            result.numRequiredAdditionalConnections = 1;
        } else {
            result.angle = _edgePos < edgeLength - 1 ? 0.0f : 60.0f;
            result.numRequiredAdditionalConnections = _edgePos < edgeLength - 1 ? 2 : 1;
        }

        if (++_edgePos >= edgeLength) {
            _edgePos = 0;
            ++_processedEdges;
        }
    } else {
        result.angle = _edgePos == 1 ? -60.0f : 0.0f;
        result.numRequiredAdditionalConnections = _edgePos == 1 ? 2 : 0;
        ++_edgePos;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForSmallLolli()
{
    ShapeGeneratorResult result;

    if (_processedEdges < 6) {
        auto edgeLength = _processedEdges / 6 + 1;
        if (_processedEdges % 6 == 1) {
            --edgeLength;
        }

        if (_processedEdges < 2) {
            result.angle = 120.0f;
            result.numRequiredAdditionalConnections = 0;
        } else {
            result.angle = 60.0f;
            result.numRequiredAdditionalConnections = 1;
        }

        if (++_edgePos >= edgeLength) {
            _edgePos = 0;
            ++_processedEdges;
        }
    } else {
        result.angle = _edgePos == 0 ? -60.0f : 0.0f;
        result.numRequiredAdditionalConnections = _edgePos == 0 ? 2 : 0;
        ++_edgePos;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForZigzag()
{
    ShapeGeneratorResult result;
    if (_edgePos % 4 == 0) {
        result.angle = 120.0f;
        result.numRequiredAdditionalConnections = 0;
    }
    if (_edgePos % 4 == 1) {
        result.angle = 0;
        result.numRequiredAdditionalConnections = _edgePos == 1 ? 0 : 1;
    }
    if (_edgePos % 4 == 2) {
        result.angle = -120.0f;
        result.numRequiredAdditionalConnections = 0;
    }
    if (_edgePos % 4 == 3) {
        result.angle = 0;
        result.numRequiredAdditionalConnections = 1;
    }
    ++_edgePos;
    return result;
}
