#pragma once

#include "EngineInterface/CellFunctionConstants.h"

#include "Base.cuh"

struct ShapeGeneratorResult
{
    float angle;
    int numRequiredAdditionalConnections;
    int requiredNodeId1;  // -1 = none
    int requiredNodeId2;  // -1 = none
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

    int _nodePos = 0;
    int _edgePos = 0;
    int _connectedNodePos1 = 0;
    int _connectedNodePos2 = 0;
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
    result.requiredNodeId1 = -1;
    result.requiredNodeId2 = -1;
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForTriangle()
{
    ShapeGeneratorResult result;
    auto edgeLength = max(2, _edgePos + 1);
    result.angle = _nodePos < edgeLength - 1 ? 0 : 120.0f;
    if (_edgePos == 0) {
        result.numRequiredAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    } else if (_edgePos == 1) {
        result.numRequiredAdditionalConnections = _nodePos == 0 ? 1 : 0;
        result.requiredNodeId1 = _nodePos == 0 ? 0 : -1;
        result.requiredNodeId2 = -1;
    } else {
        if (_nodePos == edgeLength - 1) {
            result.numRequiredAdditionalConnections = 0;
        } else if (_nodePos == edgeLength - 2) {
            result.numRequiredAdditionalConnections = 1;
        } else {
            result.numRequiredAdditionalConnections = 2;
        }

        if (_nodePos == 0) {
            result.requiredNodeId1 = _connectedNodePos2;
            result.requiredNodeId2 = _connectedNodePos1;
        } else if (_nodePos == edgeLength - 2) {
            result.requiredNodeId1 = _connectedNodePos2;
            result.requiredNodeId2 = -1;
        } else if (_nodePos == edgeLength - 1) {
            result.requiredNodeId1 = -1;
            result.requiredNodeId2 = -1;
        } else {
            result.requiredNodeId1 = _connectedNodePos2;
            result.requiredNodeId2 = _connectedNodePos2 + 1;
            ++_connectedNodePos2;
        }
    }

    if (_edgePos > 0) {
        ++_connectedNodePos1;
    }
    if (++_nodePos == edgeLength) {
        _nodePos = 0;
        ++_edgePos;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForRectangle()
{
    auto edgeLength = _edgePos / 2;

    ShapeGeneratorResult result;
    if (_edgePos == 0) {
        result.angle = 0.0f;
        result.numRequiredAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    } else if (_edgePos == 1) {
        result.angle = 90.0f;
        result.numRequiredAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    } else {
        result.angle = _nodePos == 0 ? 90.0f : 0.0f;
        result.numRequiredAdditionalConnections = _nodePos == 0 ? 0 : 1;
        result.requiredNodeId1 = _connectedNodePos1;
        result.requiredNodeId2 = -1;
    }

    if (_edgePos >= 4 && _nodePos >= 1 && _nodePos < edgeLength) {
        ++_connectedNodePos1;
    }
    if (++_nodePos > edgeLength) {
        _nodePos = 0;
        ++_edgePos;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForHexagon()
{
    ShapeGeneratorResult result;

    auto edgeLength = _edgePos / 6 + 1;
    if (_edgePos % 6 == 1) {
        --edgeLength;
    }

    if (_edgePos < 2) {
        result.angle = 120.0f;
        result.numRequiredAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    } else if (_edgePos < 6) {
        result.angle = 60.0f;
        result.numRequiredAdditionalConnections = 1;
        result.requiredNodeId1 = 0;
        result.requiredNodeId2 = -1;
    } else {
        result.angle = _nodePos < edgeLength - 1 ? 0.0f : 60.0f;

        if (_nodePos < edgeLength - 1) {
            result.numRequiredAdditionalConnections = 2;
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = _connectedNodePos1 + 1;
            ++_connectedNodePos1;
        } else {
            result.numRequiredAdditionalConnections = 1;
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = -1;
        }
    }

    if (++_nodePos >= edgeLength) {
        _nodePos = 0;
        ++_edgePos;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForLoop()
{
    ShapeGeneratorResult result;

    auto edgeLength = (_edgePos + 1) / 6 + 1;
    if (_edgePos % 6 == 0) {
        --edgeLength;
    }

    if (_edgePos < 5) {
        result.angle = 60.0f;
        result.numRequiredAdditionalConnections = 0;
    } else if (_edgePos == 5) {
        result.angle = _nodePos == 0 ? 0.0f : 60.0f;
        result.numRequiredAdditionalConnections = 1;
    } else {
        result.angle = _nodePos < edgeLength - 1 ? 0.0f : 60.0f;
        result.numRequiredAdditionalConnections = _nodePos < edgeLength - 1 ? 2 : 1;
    }

    if (++_nodePos >= edgeLength) {
        _nodePos = 0;
        ++_edgePos;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForTube()
{
    ShapeGeneratorResult result;
    if (_nodePos % 6 == 0) {
        result.angle = 0;
        result.numRequiredAdditionalConnections = 2;
    }
    if (_nodePos % 6 == 1) {
        result.angle = 60.0f;
        result.numRequiredAdditionalConnections = _nodePos == 1 ? 0 : 1;
    }
    if (_nodePos % 6 == 2) {
        result.angle = 120.0f;
        result.numRequiredAdditionalConnections = 0;
    }
    if (_nodePos % 6 == 3) {
        result.angle = 0;
        result.numRequiredAdditionalConnections = 2;
    }
    if (_nodePos % 6 == 4) {
        result.angle = -120.0f;
        result.numRequiredAdditionalConnections = _nodePos == 4 ? 1 : 2;
    }
    if (_nodePos % 6 == 5) {
        result.angle = -60.0f;
        result.numRequiredAdditionalConnections = 1;
    }
    ++_nodePos;

    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForLolli()
{
    ShapeGeneratorResult result;

    if (_edgePos < 12 || _nodePos == 0) {
        auto edgeLength = _edgePos / 6 + 1;
        if (_edgePos % 6 == 1) {
            --edgeLength;
        }

        if (_edgePos < 2) {
            result.angle = 120.0f;
            result.numRequiredAdditionalConnections = 0;
        } else if (_edgePos < 6) {
            result.angle = 60.0f;
            result.numRequiredAdditionalConnections = 1;
        } else {
            result.angle = _nodePos < edgeLength - 1 ? 0.0f : 60.0f;
            result.numRequiredAdditionalConnections = _nodePos < edgeLength - 1 ? 2 : 1;
        }

        if (++_nodePos >= edgeLength) {
            _nodePos = 0;
            ++_edgePos;
        }
    } else {
        result.angle = _nodePos == 1 ? -60.0f : 0.0f;
        result.numRequiredAdditionalConnections = _nodePos == 1 ? 2 : 0;
        ++_nodePos;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForSmallLolli()
{
    ShapeGeneratorResult result;

    if (_edgePos < 6) {
        auto edgeLength = _edgePos / 6 + 1;
        if (_edgePos % 6 == 1) {
            --edgeLength;
        }

        if (_edgePos < 2) {
            result.angle = 120.0f;
            result.numRequiredAdditionalConnections = 0;
        } else {
            result.angle = 60.0f;
            result.numRequiredAdditionalConnections = 1;
        }

        if (++_nodePos >= edgeLength) {
            _nodePos = 0;
            ++_edgePos;
        }
    } else {
        result.angle = _nodePos == 0 ? -60.0f : 0.0f;
        result.numRequiredAdditionalConnections = _nodePos == 0 ? 2 : 0;
        ++_nodePos;
    }
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForZigzag()
{
    ShapeGeneratorResult result;
    if (_nodePos % 4 == 0) {
        result.angle = 120.0f;
        result.numRequiredAdditionalConnections = 0;
    }
    if (_nodePos % 4 == 1) {
        result.angle = 0;
        result.numRequiredAdditionalConnections = _nodePos == 1 ? 0 : 1;
    }
    if (_nodePos % 4 == 2) {
        result.angle = -120.0f;
        result.numRequiredAdditionalConnections = 0;
    }
    if (_nodePos % 4 == 3) {
        result.angle = 0;
        result.numRequiredAdditionalConnections = 1;
    }
    ++_nodePos;
    return result;
}
