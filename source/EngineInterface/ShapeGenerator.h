#pragma once

#include "CellTypeConstants.h"
#include "Definitions.h"

struct ShapeGeneratorResult
{
    float angle = 0.0f;
    int numAdditionalConnections = 0;
    int requiredNodeId1 = -1;
    int requiredNodeId2 = -1;
};

class ShapeGenerator
{
public:
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionData(ConstructorShape shape);
    HOST_DEVICE inline ConstructorAngleAlignment getConstructorAngleAlignment(ConstructorShape shape);
    HOST_DEVICE inline float getPreferredFrontAngle(ConstructorShape shape);

private:
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionDataForSegment();
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionDataForTriangle();
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionDataForRectangle();
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionDataForHexagon();
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionDataForLoop();
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionDataForTube();
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionDataForLolli();
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionDataForSmallLolli();
    HOST_DEVICE inline ShapeGeneratorResult generateNextConstructionDataForZigzag();

    int _nodePos = 0;
    int _edgePos = 0;
    int _connectedNodePos2 = 0;
    int _connectedNodePos1 = 0;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionData(ConstructorShape shape)
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

HOST_DEVICE inline ConstructorAngleAlignment ShapeGenerator::getConstructorAngleAlignment(ConstructorShape shape)
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

HOST_DEVICE inline float ShapeGenerator::getPreferredFrontAngle(ConstructorShape shape)
{
    switch (shape) {
    case ConstructorShape_Segment:
        return 0.0f;
    case ConstructorShape_Triangle:
        return -30.0f;
    case ConstructorShape_Rectangle:
        return 0.0f;
    case ConstructorShape_Hexagon:
        return 0.0f;
    case ConstructorShape_Loop:
        return 0.0f;
    case ConstructorShape_Tube:
        return 240.0f;
    case ConstructorShape_Lolli:
        return 120.0f;
    case ConstructorShape_SmallLolli:
        return 120.0f;
    case ConstructorShape_Zigzag:
        return 120.0f;
    default:
        return 0.0f;
    }
}

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForSegment()
{
    ShapeGeneratorResult result;
    result.angle = 0;
    result.numAdditionalConnections = 0;
    result.requiredNodeId1 = -1;
    result.requiredNodeId2 = -1;
    return result;
}

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForTriangle()
{
    ShapeGeneratorResult result;
    auto edgeLength = _edgePos + 1 > 2 ? _edgePos + 1 : 2;
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

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForRectangle()
{
    // Builds a growing square (quadrat): each ring k adds an L-shaped border
    // extending the (k-1)x(k-1) square to k×k.
    // Even rings (Type B): right 1, up k-1, left k-1.
    // Odd rings  (Type A): up 1, right k-1, down k-1.
    // _edgePos = k-1 (ring counter), _nodePos = position within ring.
    // _connectedNodePos1 = absolute index of the next cross-connection target.
    //   Reset to _edgePos^2 - 2 at the start of each ring k >= 3.
    //   Decremented after each cross-connection except at the "pivot" (p == k-2).
    ShapeGeneratorResult result;
    result.requiredNodeId2 = -1;
    auto k = _edgePos + 1;
    auto p = _nodePos;

    if (_edgePos == 0) {
        result.angle = 0.0f;
        result.numAdditionalConnections = 0;
        result.requiredNodeId1 = -1;
    } else {
        auto isTypeB = (k % 2 == 0);
        if (p == 0 || p == k - 1) {
            result.angle = isTypeB ? 90.0f : -90.0f;
        } else if (p == 2 * k - 2) {
            result.angle = isTypeB ? -90.0f : 90.0f;
        } else {
            result.angle = 0.0f;
        }

        if (p == 0 || p == k - 1) {
            result.numAdditionalConnections = 0;
            result.requiredNodeId1 = -1;
        } else {
            result.numAdditionalConnections = 1;
            result.requiredNodeId1 = _connectedNodePos1;
            if (p != k - 2) {
                --_connectedNodePos1;
            }
        }
    }

    if (++_nodePos > 2 * k - 2) {
        _nodePos = 0;
        ++_edgePos;
        if (_edgePos >= 2) {
            // First cross-connection target for ring k = S_k - 2 = (k-1)^2 - 2 = _edgePos^2 - 2
            _connectedNodePos1 = _edgePos * _edgePos - 2;
        }
    }
    return result;
}

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForHexagon()
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

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForLoop()
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

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForTube()
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

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForLolli()
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

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForSmallLolli()
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

HOST_DEVICE inline ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForZigzag()
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
