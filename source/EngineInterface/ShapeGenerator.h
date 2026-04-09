#pragma once

#include "CellTypeConstants.h"
#include "Definitions.h"

struct ShapeGeneratorResult
{
    float angle = 0.0f;
    int numAdditionalConnections = 0;
    int requiredNodeId1 = -1;
    int requiredNodeId2 = -1;
    int requiredNodeId3 = -1;
};

class ShapeGenerator
{
public:
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionData(ConstructorShape shape);
    HOST_DEVICE ConstructorAngleAlignment getConstructorAngleAlignment(ConstructorShape shape);
    HOST_DEVICE float getPreferredFrontAngle(ConstructorShape shape);

private:
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForSegment();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForTriangle();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForRectangle();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForHexagon();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForLoop();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForTube();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForLolli();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForSmallLolli();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForZigzag();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForSpiralHexagon();

    HOST_DEVICE float getHexagonAngle(int n);
    HOST_DEVICE void getHexagonRequiredNodeIds(int n, int& r1, int& r2, int& r3);

    int _nodePos = 0;
    int _edgePos = 0;
    int _connectedNodePos2 = 0;
    int _connectedNodePos1 = 0;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionData(ConstructorShape shape)
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

HOST_DEVICE ConstructorAngleAlignment ShapeGenerator::getConstructorAngleAlignment(ConstructorShape shape)
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

HOST_DEVICE float ShapeGenerator::getPreferredFrontAngle(ConstructorShape shape)
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

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForSegment()
{
    ShapeGeneratorResult result;
    result.angle = 0;
    result.numAdditionalConnections = 0;
    result.requiredNodeId1 = -1;
    result.requiredNodeId2 = -1;
    return result;
}

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForTriangle()
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

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForRectangle()
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

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForHexagon()
{
    ShapeGeneratorResult result;
    int n = _nodePos;

    result.angle = getHexagonAngle(n);
    getHexagonRequiredNodeIds(n, result.requiredNodeId1, result.requiredNodeId2, result.requiredNodeId3);

    result.numAdditionalConnections = 0;
    if (result.requiredNodeId1 != -1) {
        ++result.numAdditionalConnections;
    }
    if (result.requiredNodeId2 != -1) {
        ++result.numAdditionalConnections;
    }
    if (result.requiredNodeId3 != -1) {
        ++result.numAdditionalConnections;
    }

    ++_nodePos;
    return result;
}

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForLoop()
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

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForTube()
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

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForLolli()
{
    ShapeGeneratorResult result;
    if (_edgePos < 12 || (_edgePos == 12 && _nodePos == 0)) {
        return generateNextConstructionDataForSpiralHexagon();
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

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForSmallLolli()
{
    ShapeGeneratorResult result;
    if (_edgePos < 6) {
        return generateNextConstructionDataForSpiralHexagon();
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

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForZigzag()
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

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForSpiralHexagon()
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

HOST_DEVICE float ShapeGenerator::getHexagonAngle(int n)
{
    if (n == 0) {
        return 0.0f;
    }

    int k = 1;
    while (1 + 3 * k * (k + 1) <= n) {
        ++k;
    }

    int firstIndex = 1 + 3 * (k - 1) * k;
    int local = n - firstIndex;
    int side = local / k;
    int pos = local % k;
    bool odd = (k % 2 == 1);

    if (k == 1) {
        float ring1[6] = {60.0f, 60.0f, 120.0f, -120.0f, 120.0f, -120.0f};
        return ring1[local];
    }

    switch (side) {
    case 0:
        if (pos == 0) {
            return odd ? 60.0f : -60.0f;
        }
        return 0.0f;
    case 1:
        return (pos % 2 == 0) ? (odd ? 120.0f : -120.0f) : (odd ? -120.0f : 120.0f);
    case 2:
        if (!odd) {
            return (pos % 2 == 0) ? -120.0f : 120.0f;
        }
        if (pos == k - 2) {
            return 60.0f;
        }
        if (pos == k - 1) {
            return 120.0f;
        }
        return (pos % 2 == 0) ? -120.0f : 120.0f;
    case 3:
        if (odd) {
            return (pos % 2 == 0) ? -120.0f : 120.0f;
        }
        if (pos == 0) {
            return -120.0f;
        }
        if (pos == 1) {
            return -60.0f;
        }
        return (pos % 2 == 0) ? 120.0f : -120.0f;
    case 4:
        return (pos % 2 == 0) ? 120.0f : -120.0f;
    case 5:
        if (pos < k - 1) {
            return 0.0f;
        }
        return odd ? -120.0f : 120.0f;
    default:
        return 0.0f;
    }
}

HOST_DEVICE void ShapeGenerator::getHexagonRequiredNodeIds(int n, int& r1, int& r2, int& r3)
{
    r1 = -1;
    r2 = -1;
    r3 = -1;

    if (n < 0) {
        return;
    }

    if (n <= 21) {
        int baseTable[22][3] = {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {2, 1, 0},    {3, -1, -1},  {4, 0, -1},   {5, -1, -1},
                                {5, -1, -1},  {-1, -1, -1}, {8, 5, 3},    {9, -1, -1},  {10, 3, -1},  {11, -1, -1}, {12, -1, -1}, {12, 3, 2},
                                {14, -1, -1}, {15, 2, -1},  {2, 1, -1},   {17, -1, -1}, {17, 16, -1}, {16, -1, -1}};
        r1 = baseTable[n][0];
        r2 = baseTable[n][1];
        r3 = baseTable[n][2];
        return;
    }

    auto shellStart = [](int s) { return 3 * s * s - 2 * s + 1; };

    int s = 3;
    while (shellStart(s + 1) <= n) {
        ++s;
    }

    int D = shellStart(s);
    int Dp = shellStart(s - 1);
    int t = n - D;

    if (s % 2 == 1) {
        // ODD SHELL
        if (t == 0) {
            return;
        }

        // Segment A: length 2*s - 2, range [1 .. 2*s-2]
        if (1 <= t && t <= 2 * s - 2) {
            int u = t - 1;
            if (u % 2 == 0) {
                int k = u / 2;
                int y = Dp + (4 * s - 5) - 2 * k;
                if (k < s - 2) {
                    r1 = n - 2;
                    r2 = y;
                    r3 = y - 2;
                } else {
                    r1 = n - 2;
                    r2 = y;
                    r3 = y - 1;
                }
            } else {
                r1 = n - 2;
            }
            return;
        }

        // Separator after segment A
        if (t == 2 * s - 1) {
            return;
        }

        // Segment B: length 2*s + 1, range [2*s .. 4*s]
        if (2 * s <= t && t <= 4 * s) {
            int u = t - 2 * s;
            int z0 = D - (4 * s - 3);
            if (u == 0) {
                r1 = n - 2;
                r2 = n - 3;
                r3 = z0;
                return;
            }
            if (u % 2 == 1) {
                r1 = n - 2;
                return;
            }
            int j = (u - 1) / 2;
            int z = z0 - 2 * j;
            if (j < s - 1) {
                r1 = n - 2;
                r2 = z;
                r3 = z - 2;
            } else {
                r1 = n - 2;
                r2 = z;
            }
            return;
        }

        // Segment C1: length s - 1, range [4*s+1 .. 5*s-1]
        if (4 * s + 1 <= t && t <= 5 * s - 1) {
            int j = t - (4 * s + 1);
            r1 = Dp - j;
            r2 = Dp - j - 1;
            return;
        }

        // Segment C2: length s + 1, range [5*s .. 6*s]
        if (5 * s <= t && t <= 6 * s) {
            int u = t - 5 * s;
            int E = D + 5 * s - 2;
            if (u == 0) {
                r1 = n - 2;
                return;
            }
            if (u == s) {
                r1 = E - (s - 1);
                return;
            }
            r1 = E - (u - 1);
            r2 = E - u;
            return;
        }
    } else {
        // EVEN SHELL
        if (t == 0) {
            return;
        }

        // Segment A1: length 2*s, range [1 .. 2*s]
        if (1 <= t && t <= 2 * s) {
            int u = t - 1;
            if (u % 2 == 0) {
                int k = u / 2;
                int y = Dp + (4 * s - 5) - 2 * k;
                if (k < s - 1) {
                    r1 = n - 2;
                    r2 = y;
                    r3 = y - 2;
                } else {
                    r1 = n - 2;
                    r2 = y;
                }
            } else {
                r1 = n - 2;
            }
            return;
        }

        // Segment A2: length 2*s, range [2*s+1 .. 4*s]
        if (2 * s + 1 <= t && t <= 4 * s) {
            int u = t - (2 * s + 1);
            int z0 = Dp + (2 * s - 3);
            if (u == 0) {
                r1 = n - 2;
                return;
            }
            if (u == 1) {
                r1 = n - 3;
                r2 = z0;
                r3 = z0 - 1;
                return;
            }
            if (u % 2 == 0) {
                r1 = n - 2;
                return;
            }
            int j = (u - 1) / 2;
            int z = z0 - 2 * (j - 1) - 1;
            if (j < s - 1) {
                r1 = n - 2;
                r2 = z;
                r3 = z - 2;
            } else {
                r1 = n - 2;
                r2 = z;
            }
            return;
        }

        // Segment C1: length s - 1, range [4*s+1 .. 5*s-1]
        if (4 * s + 1 <= t && t <= 5 * s - 1) {
            int j = t - (4 * s + 1);
            r1 = Dp - j;
            r2 = Dp - j - 1;
            return;
        }

        // Segment C2: length s + 1, range [5*s .. 6*s]
        if (5 * s <= t && t <= 6 * s) {
            int u = t - 5 * s;
            int E = D + 5 * s - 2;
            if (u == 0) {
                r1 = n - 2;
                return;
            }
            if (u == s) {
                r1 = E - (s - 1);
                return;
            }
            r1 = E - (u - 1);
            r2 = E - u;
            return;
        }
    }
}
