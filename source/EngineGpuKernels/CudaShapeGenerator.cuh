#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "Base.cuh"

struct ShapeGeneratorResult
{
    float angle;
    int numAdditionalConnections;
    int requiredNodeId1;  // -1 = none
    int requiredNodeId2;  // -1 = none
};

class CudaShapeGenerator
{
public:
    __inline__ __device__ ShapeGeneratorResult generateNextConstructionData(ConstructorShape shape);
    __inline__ __device__ ConstructorAngleAlignment getConstructorAngleAlignment(ConstructorShape shape);

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
    int _connectedNodePos2 = 0;
    int _connectedNodePos1 = 0;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionData(ConstructorShape shape)
{
    switch (shape) {
    case ConstructorShape_Segment:
        return generateNextConstructionDataForSegment();
    case ConstructorShape_Triangle:
        return generateNextConstructionDataForTriangle();
    case ConstructorShape_Rectangle:
        return generateNextConstructionDataForRectangle();
    case ConstructorShape_Hexagon:
        return generateNextConstructionDataForHexagon();
    case ConstructorShape_Loop:
        return generateNextConstructionDataForLoop();
    case ConstructorShape_Tube:
        return generateNextConstructionDataForTube();
    case ConstructorShape_Lolli:
        return generateNextConstructionDataForLolli();
    case ConstructorShape_SmallLolli:
        return generateNextConstructionDataForSmallLolli();
    case ConstructorShape_Zigzag:
        return generateNextConstructionDataForZigzag();
    default:
        return ShapeGeneratorResult();
    }
}

__inline__ __device__ ConstructorAngleAlignment CudaShapeGenerator::getConstructorAngleAlignment(ConstructorShape shape)
{
    switch (shape) {
    case ConstructorShape_Custom:
        return ConstructorAngleAlignment_60;
    case ConstructorShape_Segment:
        return ConstructorAngleAlignment_60;
    case ConstructorShape_Triangle:
        return ConstructorAngleAlignment_60;
    case ConstructorShape_Rectangle:
        return ConstructorAngleAlignment_90;
    case ConstructorShape_Hexagon:
        return ConstructorAngleAlignment_60;
    case ConstructorShape_Loop:
        return ConstructorAngleAlignment_60;
    case ConstructorShape_Tube:
        return ConstructorAngleAlignment_60;
    case ConstructorShape_Lolli:
        return ConstructorAngleAlignment_60;
    case ConstructorShape_SmallLolli:
        return ConstructorAngleAlignment_60;
    default:
        return ConstructorAngleAlignment_60;
    }
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForSegment()
{
    ShapeGeneratorResult result;
    result.angle = 0;
    result.numAdditionalConnections = 0;
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
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    } else if (_edgePos == 1) {
        result.numAdditionalConnections = _nodePos == 0 ? 1 : 0;
        result.requiredNodeId1 = _nodePos == 0 ? 0 : -1;
        result.requiredNodeId2 = -1;
    } else {
        if (_nodePos == 0) {
            result.numAdditionalConnections = 2;
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = _connectedNodePos2;
        } else if (_nodePos == edgeLength - 2) {
            result.numAdditionalConnections = 1;
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = -1;
        } else if (_nodePos == edgeLength - 1) {
            result.numAdditionalConnections = 0;
            result.requiredNodeId1 = -1;
            result.requiredNodeId2 = -1;
        } else {
            result.numAdditionalConnections = 2;
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = _connectedNodePos1 + 1;
        }
    }

    if (_edgePos > 0) {
        ++_connectedNodePos2;
    }
    if (_edgePos > 1 && _nodePos > 0 && _nodePos < edgeLength - 2) {
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
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    } else if (_edgePos == 1) {
        result.angle = 90.0f;
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    } else {
        result.angle = _nodePos == 0 ? 90.0f : 0.0f;
        result.numAdditionalConnections = _nodePos == 0 ? 0 : 1;
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
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    } else if (_edgePos < 6) {
        result.angle = 60.0f;
        result.numAdditionalConnections = 1;
        result.requiredNodeId1 = 0;
        result.requiredNodeId2 = -1;
    } else {
        result.angle = _nodePos < edgeLength - 1 ? 0.0f : 60.0f;

        if (_nodePos < edgeLength - 1) {
            result.numAdditionalConnections = 2;
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = _connectedNodePos1 + 1;
        } else {
            result.numAdditionalConnections = 1;
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = -1;
        }
    }

    if (_edgePos >= 6 && _nodePos < edgeLength - 1) {
        ++_connectedNodePos1;
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
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    } else if (_edgePos == 5) {
        result.angle = _nodePos == 0 ? 0.0f : 60.0f;
        result.numAdditionalConnections = 1;
        result.requiredNodeId1 = 0;
        result.requiredNodeId2 = -1;
    } else {
        result.angle = _nodePos < edgeLength - 1 ? 0.0f : 60.0f;
        result.numAdditionalConnections = _nodePos < edgeLength - 1 ? 2 : 1;
        if (_nodePos < edgeLength - 1) {
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = _connectedNodePos1 + 1;
        } else {
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = -1;
        }
    }

    if (_edgePos >= 6 && _nodePos < edgeLength - 1) {
        ++_connectedNodePos1;
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
        if (_nodePos == 0) {
            result.numAdditionalConnections = 0;
            result.requiredNodeId1 = -1;
            result.requiredNodeId2 = -1;
        } else {
            result.numAdditionalConnections = 2;
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = _connectedNodePos1 + 1;
        }
    }
    if (_nodePos % 6 == 1) {
        result.angle = 60.0f;
        if (_nodePos == 1) {
            result.numAdditionalConnections = 0;
            result.requiredNodeId1 = -1;
            result.requiredNodeId2 = -1;
        } else {
            result.numAdditionalConnections = 1;
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = -1;
        }
    }
    if (_nodePos % 6 == 2) {
        result.angle = 120.0f;
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    }
    if (_nodePos % 6 == 3) {
        result.angle = 0;
        result.numAdditionalConnections = 2;
        result.requiredNodeId1 = _connectedNodePos1;
        result.requiredNodeId2 = _connectedNodePos1 + 1;
    }
    if (_nodePos % 6 == 4) {
        result.angle = -120.0f;
        result.numAdditionalConnections = _nodePos == 4 ? 1 : 2;
        if (_nodePos == 4) {
            result.requiredNodeId1 = _connectedNodePos1;
            result.requiredNodeId2 = -1;
        } else {
            result.requiredNodeId1 = _connectedNodePos1 - 1;
            result.requiredNodeId2 = _connectedNodePos1;
        }
    }
    if (_nodePos % 6 == 5) {
        result.angle = -60.0f;
        result.numAdditionalConnections = 1;
        result.requiredNodeId1 = _connectedNodePos1 + 3;
        result.requiredNodeId2 = -1;
    }

    if (_nodePos % 6 == 1 && _nodePos > 1) {
        _connectedNodePos1 += 4;
    }
    if (_nodePos % 6 == 5) {
        _connectedNodePos1 += 2;
    }
    ++_nodePos;

    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForLolli()
{
    ShapeGeneratorResult result;
    if (_edgePos < 12 || (_edgePos == 12 && _nodePos == 0)) {
        return generateNextConstructionDataForHexagon();
    }

    if (_nodePos == 1) {
        result.angle = -60.0f;
        result.numAdditionalConnections = 2;
        result.requiredNodeId1 = 6;
        result.requiredNodeId2 = 7;
    } else {
        result.angle = 0.0f;
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    }

    _nodePos = 2;
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForSmallLolli()
{
    ShapeGeneratorResult result;
    if (_edgePos < 6) {
        return generateNextConstructionDataForHexagon();
    }

    if (_nodePos == 0) {
        result.angle = -60.0f;
        result.numAdditionalConnections = 2;
        result.requiredNodeId1 = 0;
        result.requiredNodeId2 = 1;
    } else {
        result.angle = 0.0f;
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    }

    _nodePos = 1;
    return result;
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForZigzag()
{
    ShapeGeneratorResult result;
    if (_nodePos % 4 == 0) {
        result.angle = 120.0f;
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    }
    if (_nodePos % 4 == 1) {
        result.angle = 0;
        result.numAdditionalConnections = _nodePos == 1 ? 0 : 1;
        result.requiredNodeId1 = _connectedNodePos1;
        result.requiredNodeId2 = -1;
    }
    if (_nodePos % 4 == 2) {
        result.angle = -120.0f;
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    }
    if (_nodePos % 4 == 3) {
        result.angle = 0;
        result.numAdditionalConnections = 1;
        result.requiredNodeId1 = _connectedNodePos1;
        result.requiredNodeId2 = -1;
    }
    if (_nodePos > 1) {
        ++_connectedNodePos1;
    }
    ++_nodePos;
    return result;
}
