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
    __inline__ __device__ int getHexagonRingMoveDir(int ringSize, int ringStep) const;
    __inline__ __device__ int getHexagonOutgoingDir() const;
    __inline__ __device__ int getHexagonRingStartIndex(int ringSize) const;
    __inline__ __device__ bool isHexagonNeighbor(int q1, int r1, int q2, int r2) const;
    __inline__ __device__ void getHexagonAdditionalConnections(int& numAdditionalConnections, int& requiredNodeId1, int& requiredNodeId2) const;
    __inline__ __device__ void advanceHexagonState(int outgoingDir);

    int _nodePos = 0;
    int _edgePos = 0;
    int _connectedNodePos2 = 0;
    int _connectedNodePos1 = 0;
    int _hexNodeIndex = 0;
    int _hexRingSize = 1;
    int _hexRingPos = 0;
    int _hexQ = 0;
    int _hexR = 0;
    int _hexIncomingDir = 0;
    int _hexCurrentRingStartQ = 0;
    int _hexCurrentRingStartR = 0;
    int _hexPrevRingStartQ = 0;
    int _hexPrevRingStartR = 0;
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

__inline__ __device__ int CudaShapeGenerator::getHexagonRingMoveDir(int ringSize, int ringStep) const
{
    if (ringSize == 2) {
        int constexpr RING2[6] = {0, 2, 0, 2, 3, 4};  // E, SW, E, SW, W, NW
        return RING2[ringStep];
    }

    if (ringSize % 2 == 1) {
        if (ringStep == 0) {
            return 2;  // SW
        }
        ringStep -= 1;
        auto const segment2Length = ringSize - 1;
        if (ringStep < segment2Length) {
            return 1;  // SE
        }
        ringStep -= segment2Length;
        auto const segment3Length = 2 * ringSize - 1;
        if (ringStep < segment3Length) {
            return ringStep % 2 == 0 ? 5 : 1;  // NE, SE, ...
        }
        ringStep -= segment3Length;
        auto const segment4Length = 2 * (ringSize - 2);
        if (ringStep < segment4Length) {
            return ringStep % 2 == 0 ? 4 : 0;  // NW, E, ...
        }
        return 4;  // NW
    }

    if (ringStep == 0) {
        return 0;  // E
    }
    ringStep -= 1;
    auto const segment2Length = ringSize - 1;
    if (ringStep < segment2Length) {
        return 1;  // SE
    }
    ringStep -= segment2Length;
    auto const segment3Length = 2 * ringSize - 3;
    if (ringStep < segment3Length) {
        return ringStep % 2 == 0 ? 3 : 1;  // W, SE, ...
    }
    ringStep -= segment3Length;
    if (ringStep < 2) {
        return ringStep == 0 ? 1 : 3;  // SE, W
    }
    ringStep -= 2;
    auto const segment5Length = 2 * (ringSize - 2);
    if (ringStep < segment5Length) {
        return ringStep % 2 == 0 ? 4 : 2;  // NW, SW, ...
    }
    return 4;  // NW
}

__inline__ __device__ int CudaShapeGenerator::getHexagonOutgoingDir() const
{
    if (_hexRingSize == 1) {
        return getHexagonRingMoveDir(2, 0);
    }
    auto const nodesInCurrentRing = 6 * (_hexRingSize - 1);
    if (_hexRingPos + 1 < nodesInCurrentRing) {
        return getHexagonRingMoveDir(_hexRingSize, _hexRingPos + 1);
    }
    return getHexagonRingMoveDir(_hexRingSize + 1, 0);
}

__inline__ __device__ int CudaShapeGenerator::getHexagonRingStartIndex(int ringSize) const
{
    if (ringSize <= 1) {
        return 0;
    }
    return 1 + 3 * (ringSize - 2) * (ringSize - 1);
}

__inline__ __device__ bool CudaShapeGenerator::isHexagonNeighbor(int q1, int r1, int q2, int r2) const
{
    auto const dq = q2 - q1;
    auto const dr = r2 - r1;
    auto const ds = -dq - dr;
    return max(max(abs(dq), abs(dr)), abs(ds)) == 1;
}

__inline__ __device__ void CudaShapeGenerator::getHexagonAdditionalConnections(int& numAdditionalConnections, int& requiredNodeId1, int& requiredNodeId2) const
{
    numAdditionalConnections = 0;
    requiredNodeId1 = -1;
    requiredNodeId2 = -1;

    if (_hexNodeIndex == 0 || _hexRingSize == 1) {
        return;
    }

    auto const previousNodeId = _hexNodeIndex - 1;
    auto const previousRingSize = _hexRingSize - 1;
    auto const previousRingStartIndex = getHexagonRingStartIndex(previousRingSize);
    auto const previousRingNodeCount = previousRingSize == 1 ? 1 : 6 * (previousRingSize - 1);

    auto previousQ = _hexPrevRingStartQ;
    auto previousR = _hexPrevRingStartR;
    for (int pos = 0; pos < previousRingNodeCount; ++pos) {
        auto const nodeId = previousRingStartIndex + pos;
        if (nodeId != previousNodeId && isHexagonNeighbor(_hexQ, _hexR, previousQ, previousR)) {
            if (numAdditionalConnections == 0) {
                requiredNodeId1 = nodeId;
            } else if (numAdditionalConnections == 1) {
                requiredNodeId2 = nodeId;
            }
            ++numAdditionalConnections;
        }

        if (pos + 1 < previousRingNodeCount) {
            auto const dir = getHexagonRingMoveDir(previousRingSize, pos + 1);
            int constexpr DIR_Q[6] = {1, 0, -1, -1, 0, 1};
            int constexpr DIR_R[6] = {0, 1, 1, 0, -1, -1};
            previousQ += DIR_Q[dir];
            previousR += DIR_R[dir];
        }
    }

    if (numAdditionalConnections > 2) {
        numAdditionalConnections = 2;
    }
}

__inline__ __device__ void CudaShapeGenerator::advanceHexagonState(int outgoingDir)
{
    int constexpr DIR_Q[6] = {1, 0, -1, -1, 0, 1};
    int constexpr DIR_R[6] = {0, 1, 1, 0, -1, -1};

    _hexQ += DIR_Q[outgoingDir];
    _hexR += DIR_R[outgoingDir];
    _hexIncomingDir = outgoingDir;
    ++_hexNodeIndex;

    if (_hexRingSize == 1) {
        _hexRingSize = 2;
        _hexRingPos = 0;
        _hexPrevRingStartQ = 0;
        _hexPrevRingStartR = 0;
        _hexCurrentRingStartQ = _hexQ;
        _hexCurrentRingStartR = _hexR;
        return;
    }

    auto const nodesInCurrentRing = 6 * (_hexRingSize - 1);
    if (++_hexRingPos >= nodesInCurrentRing) {
        _hexRingPos = 0;
        ++_hexRingSize;
        _hexPrevRingStartQ = _hexCurrentRingStartQ;
        _hexPrevRingStartR = _hexCurrentRingStartR;
        _hexCurrentRingStartQ = _hexQ;
        _hexCurrentRingStartR = _hexR;
    }
}

__inline__ __device__ ShapeGeneratorResult CudaShapeGenerator::generateNextConstructionDataForHexagon()
{
    ShapeGeneratorResult result;
    auto const outgoingDir = getHexagonOutgoingDir();
    if (_hexNodeIndex == 0) {
        result.angle = 120.0f;
    } else {
        auto delta = outgoingDir - _hexIncomingDir;
        while (delta <= -3) {
            delta += 6;
        }
        while (delta > 3) {
            delta -= 6;
        }
        result.angle = static_cast<float>(delta * 60);
    }
    getHexagonAdditionalConnections(result.numAdditionalConnections, result.requiredNodeId1, result.requiredNodeId2);

    advanceHexagonState(outgoingDir);
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
        auto edgeLength = _edgePos / 6 + 1;
        if (_edgePos % 6 == 1) {
            --edgeLength;
        }

        if (_edgePos < 2) {
            result.angle = 120.0f;
            result.numAdditionalConnections = 0;
            result.requiredNodeId1 = -1;
            result.requiredNodeId2 = -1;
        } else {
            result.angle = 60.0f;
            result.numAdditionalConnections = 1;
            result.requiredNodeId1 = _edgePos < 6 ? 0 : _connectedNodePos1;
            result.requiredNodeId2 = -1;
        }

        if (++_nodePos >= edgeLength) {
            _nodePos = 0;
            ++_edgePos;
        }
        return result;
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
