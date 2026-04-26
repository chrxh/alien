#pragma once

#include "CellTypeConstants.h"
#include "Definitions.h"

struct ShapeGeneratorResult
{
    float angle = 0.0f;
    int numAdditionalConnections = 0;

    int requiredNodeId[3] = {-1, -1, -1};
    float requiredNodeAngle1[3] = {0, 0, 0};
    float requiredNodeAngle2[3] = {0, 0, 0};
};

class ShapeGenerator
{
public:
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionData(ConstructorShape shape);
    HOST_DEVICE float getPreferredFrontAngle(ConstructorShape shape);

private:
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForSegment();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForTriangle();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForRectangle();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForHexagon();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForTube();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForLargeLolli();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForSmallLolli();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForZigzag();
    HOST_DEVICE ShapeGeneratorResult generateNextConstructionDataForSpiralHexagon();

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
    case ConstructorShape_Tube:
        return generateNextConstructionDataForTube();
    case ConstructorShape_LargeLolli:
        return generateNextConstructionDataForLargeLolli();
    case ConstructorShape_SmallLolli:
        return generateNextConstructionDataForSmallLolli();
    case ConstructorShape_Zigzag:
        return generateNextConstructionDataForZigzag();
    default:
        return ShapeGeneratorResult();
    }
}

HOST_DEVICE float ShapeGenerator::getPreferredFrontAngle(ConstructorShape shape)
{
    switch (shape) {
    case ConstructorShape_Segment:
        return -180.0f;
    case ConstructorShape_Triangle:
        return -150.0f;
    case ConstructorShape_Rectangle:
        return -90.0f;
    case ConstructorShape_Hexagon:
        return -120.0f;
    case ConstructorShape_Tube:
        return -120.0f;
    case ConstructorShape_LargeLolli:
        return -120.0f;
    case ConstructorShape_SmallLolli:
        return 120.0f;
    case ConstructorShape_Zigzag:
        return -120.0f;
    default:
        return 0.0f;
    }
}

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForSegment()
{
    ShapeGeneratorResult result;
    result.angle = 0;
    result.numAdditionalConnections = 0;
    result.requiredNodeId[0] = -1;
    result.requiredNodeId[1] = -1;
    return result;
}

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForTriangle()
{
    // Builds a growing equilateral triangle: each edge k adds a row of L = max(k+1, 2) nodes.
    // Total nodes after edges 0..k-1 = 1 + k*(k+1)/2 for k >= 1.
    // Even edges end with angle 120°, odd edges end with angle -120°.
    // _edgePos = k (edge counter), _nodePos = position within edge.
    // _connectedNodePos1 = absolute index of the next cross-connection target.
    //   Reset to k*(k+1)/2 - 1 at the start of each edge (harmless for k=0 since edge 0 has no connections).
    //   Decremented after each double cross-connection.
    ShapeGeneratorResult result;
    result.requiredNodeId[1] = -1;
    auto k = _edgePos;
    auto L = k + 1 > 2 ? k + 1 : 2;
    auto p = _nodePos;

    if (k == 0) {
        result.angle = (p == L - 1) ? 120.0f : 0.0f;
        result.numAdditionalConnections = 0;
        result.requiredNodeId[0] = -1;
        result.requiredNodeAngle1[0] = 0.0f;
    } else {
        auto isEven = (k % 2 == 0);

        if (p < L - 2) {
            result.angle = 0.0f;
        } else if (p == L - 2) {
            result.angle = isEven ? 60.0f : -60.0f;
        } else {
            result.angle = isEven ? 120.0f : -120.0f;
        }

        if (p == L - 1) {
            result.numAdditionalConnections = 0;
            result.requiredNodeId[0] = -1;
            result.requiredNodeAngle1[0] = 0.0f;
        } else if (p == L - 2) {
            result.numAdditionalConnections = 1;
            result.requiredNodeId[0] = _connectedNodePos1;
            result.requiredNodeAngle1[0] = isEven ? -120.0f : 120.0f;
            result.requiredNodeAngle2[0] = isEven ? 240.0f : (k == 1 ? 60.0f : 120.0f);
        } else {
            result.numAdditionalConnections = 2;
            result.requiredNodeId[0] = _connectedNodePos1;
            result.requiredNodeId[1] = _connectedNodePos1 - 1;
            result.requiredNodeAngle1[0] = isEven ? -120.0f : 120.0f;
            result.requiredNodeAngle1[1] = isEven ? -60.0f : 60.0f;
            result.requiredNodeAngle2[0] = isEven ? (p == 0 ? 300.0f : 240.0f) : (p == 0 ? 60.0f : 120.0f);
            result.requiredNodeAngle2[1] = isEven ? 300.0f : 60.0f;
            --_connectedNodePos1;
        }
    }

    if (++_nodePos == L) {
        _nodePos = 0;
        ++_edgePos;
        _connectedNodePos1 = _edgePos * (_edgePos + 1) / 2 - 1;
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
    result.requiredNodeId[1] = -1;
    auto k = _edgePos + 1;
    auto p = _nodePos;

    if (_edgePos == 0) {
        result.angle = 0.0f;
        result.numAdditionalConnections = 0;
        result.requiredNodeId[0] = -1;
        result.requiredNodeAngle1[0] = 0.0f;
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
            result.requiredNodeId[0] = -1;
            result.requiredNodeAngle1[0] = 0.0f;
        } else {
            result.numAdditionalConnections = 1;
            result.requiredNodeId[0] = _connectedNodePos1;
            result.requiredNodeAngle1[0] = isTypeB ? 90.0f : -90.0f;
            result.requiredNodeAngle2[0] = (p == k && p != 2 * k - 2) ? 180.0f : (isTypeB ? 90.0f : 270.0f);
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

    if (n <= 0) {
        ++_nodePos;
        return result;
    }

    float angleSign = 0.0f;  // Set in every branch that also sets requiredNodeId

    if (n <= 21) {
        struct BaseEntry
        {
            float angle;
            int r1, r2, r3;
            float sign;
            float a21, a22, a23;
        };
        BaseEntry baseTable[21] = {
            {60.0f, -1, -1, -1, 0.0f, 0.0f, 0.0f, 0.0f},        // 1
            {60.0f, -1, -1, -1, 0.0f, 0.0f, 0.0f, 0.0f},        // 2
            {120.0f, -1, -1, -1, 0.0f, 0.0f, 0.0f, 0.0f},       // 3
            {-120.0f, 2, 1, 0, 1.0f, 60.0f, 60.0f, 60.0f},      // 4
            {120.0f, 3, -1, -1, -1.0f, 300.0f, 0.0f, 0.0f},     // 5
            {-120.0f, 4, 0, -1, 1.0f, 60.0f, 120.0f, 0.0f},     // 6
            {-60.0f, 5, -1, -1, -1.0f, 300.0f, 0.0f, 0.0f},     // 7
            {0.0f, 5, -1, -1, -1.0f, 240.0f, 0.0f, 0.0f},       // 8
            {-120.0f, -1, -1, -1, 0.0f, 0.0f, 0.0f, 0.0f},      // 9
            {120.0f, 8, 5, 3, -1.0f, 300.0f, 180.0f, 240.0f},   // 10
            {-120.0f, 9, -1, -1, 1.0f, 60.0f, 0.0f, 0.0f},      // 11
            {120.0f, 10, 3, -1, -1.0f, 300.0f, 180.0f, 0.0f},   // 12
            {-120.0f, 11, -1, -1, 1.0f, 60.0f, 0.0f, 0.0f},     // 13
            {-60.0f, 12, -1, -1, -1.0f, 300.0f, 0.0f, 0.0f},    // 14
            {120.0f, 12, 3, 2, -1.0f, 240.0f, 120.0f, 300.0f},  // 15
            {-120.0f, 14, -1, -1, 1.0f, 60.0f, 0.0f, 0.0f},     // 16
            {0.0f, 15, 2, -1, -1.0f, 300.0f, 240.0f, 0.0f},     // 17
            {120.0f, 2, 1, -1, -1.0f, 180.0f, 300.0f, 0.0f},    // 18
            {60.0f, 17, -1, -1, 1.0f, 60.0f, 0.0f, 0.0f},       // 19
            {0.0f, 17, 16, -1, 1.0f, 120.0f, 60.0f, 0.0f},      // 20
            {0.0f, 16, -1, -1, 1.0f, 120.0f, 0.0f, 0.0f},       // 21
        };
        auto const& e = baseTable[n - 1];
        result.angle = e.angle;
        result.requiredNodeId[0] = e.r1;
        result.requiredNodeId[1] = e.r2;
        result.requiredNodeId[2] = e.r3;
        result.requiredNodeAngle2[0] = e.a21;
        result.requiredNodeAngle2[1] = e.a22;
        result.requiredNodeAngle2[2] = e.a23;
        angleSign = e.sign;
    } else {
        // Shell decomposition for angle
        int k = 1;
        while (1 + 3 * k * (k + 1) <= n) {
            ++k;
        }
        int firstIndex = 1 + 3 * (k - 1) * k;
        int local = n - firstIndex;
        int side = local / k;
        int pos = local % k;
        bool odd = (k % 2 == 1);

        switch (side) {
        case 0:
            result.angle = (pos == 0) ? (odd ? 60.0f : -60.0f) : 0.0f;
            break;
        case 1:
            result.angle = (pos % 2 == 0) ? (odd ? 120.0f : -120.0f) : (odd ? -120.0f : 120.0f);
            break;
        case 2:
            if (!odd) {
                result.angle = (pos % 2 == 0) ? -120.0f : 120.0f;
            } else if (pos == k - 2) {
                result.angle = 60.0f;
            } else if (pos == k - 1) {
                result.angle = 120.0f;
            } else {
                result.angle = (pos % 2 == 0) ? -120.0f : 120.0f;
            }
            break;
        case 3:
            if (odd) {
                result.angle = (pos % 2 == 0) ? -120.0f : 120.0f;
            } else if (pos == 0) {
                result.angle = -120.0f;
            } else if (pos == 1) {
                result.angle = -60.0f;
            } else {
                result.angle = (pos % 2 == 0) ? 120.0f : -120.0f;
            }
            break;
        case 4:
            result.angle = (pos % 2 == 0) ? 120.0f : -120.0f;
            break;
        case 5:
            result.angle = (pos < k - 1) ? 0.0f : (odd ? -120.0f : 120.0f);
            break;
        default:
            break;
        }

        // Shell decomposition for required node IDs
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
            if (1 <= t && t <= 2 * s - 2) {
                // Segment A: length 2*s - 2
                int u = t - 1;
                angleSign = (u % 2 == 0) ? 1.0f : -1.0f;
                if (u % 2 == 0) {
                    int j = u / 2;
                    int y = Dp + (4 * s - 5) - 2 * j;
                    result.requiredNodeAngle2[0] = 60.0f;
                    result.requiredNodeAngle2[1] = 180.0f;
                    if (j < s - 2) {
                        result.requiredNodeId[0] = n - 2;
                        result.requiredNodeId[1] = y;
                        result.requiredNodeId[2] = y - 2;
                        result.requiredNodeAngle2[2] = 120.0f;
                    } else {
                        result.requiredNodeId[0] = n - 2;
                        result.requiredNodeId[1] = y;
                        result.requiredNodeId[2] = y - 1;
                        result.requiredNodeAngle2[2] = 60.0f;
                    }
                } else {
                    result.requiredNodeId[0] = n - 2;
                    result.requiredNodeAngle2[0] = 300.0f;
                }
            } else if (2 * s <= t && t <= 4 * s) {
                // Segment B: length 2*s + 1
                int u = t - 2 * s;
                angleSign = (u % 2 == 0) ? 1.0f : -1.0f;
                int z0 = D - (4 * s - 3);
                if (u == 0) {
                    result.requiredNodeId[0] = n - 2;
                    result.requiredNodeId[1] = n - 3;
                    result.requiredNodeId[2] = z0;
                    result.requiredNodeAngle2[0] = 60.0f;
                    result.requiredNodeAngle2[1] = 60.0f;
                    result.requiredNodeAngle2[2] = 120.0f;
                } else if (u % 2 == 1) {
                    result.requiredNodeId[0] = n - 2;
                    result.requiredNodeAngle2[0] = 300.0f;
                } else {
                    int j = (u - 1) / 2;
                    int z = z0 - 2 * j;
                    result.requiredNodeAngle2[0] = 60.0f;
                    result.requiredNodeAngle2[1] = 180.0f;
                    if (j < s - 1) {
                        result.requiredNodeId[0] = n - 2;
                        result.requiredNodeId[1] = z;
                        result.requiredNodeId[2] = z - 2;
                        result.requiredNodeAngle2[2] = 120.0f;
                    } else {
                        result.requiredNodeId[0] = n - 2;
                        result.requiredNodeId[1] = z;
                    }
                }
            } else if (4 * s + 1 <= t && t <= 5 * s - 1) {
                // Segment C1: length s - 1
                angleSign = 1.0f;
                int j = t - (4 * s + 1);
                result.requiredNodeId[0] = Dp - j;
                result.requiredNodeId[1] = Dp - j - 1;
                result.requiredNodeAngle2[0] = (j == 0) ? 240.0f : 120.0f;
                result.requiredNodeAngle2[1] = 60.0f;
            } else if (5 * s <= t && t <= 6 * s) {
                // Segment C2: length s + 1
                angleSign = -1.0f;
                int u = t - 5 * s;
                int E = D + 5 * s - 2;
                if (u == 0) {
                    result.requiredNodeId[0] = n - 2;
                    result.requiredNodeAngle2[0] = 300.0f;
                } else if (u == s) {
                    result.requiredNodeId[0] = E - (s - 1);
                    result.requiredNodeAngle2[0] = 240.0f;
                } else {
                    result.requiredNodeId[0] = E - (u - 1);
                    result.requiredNodeId[1] = E - u;
                    result.requiredNodeAngle2[0] = 240.0f;
                    result.requiredNodeAngle2[1] = 300.0f;
                }
            }
        } else {
            // EVEN SHELL
            if (1 <= t && t <= 2 * s) {
                // Segment A1: length 2*s
                int u = t - 1;
                angleSign = (u % 2 == 0) ? -1.0f : 1.0f;
                if (u % 2 == 0) {
                    int j = u / 2;
                    int y = Dp + (4 * s - 5) - 2 * j;
                    result.requiredNodeAngle2[0] = 300.0f;
                    result.requiredNodeAngle2[1] = 180.0f;
                    if (j < s - 1) {
                        result.requiredNodeId[0] = n - 2;
                        result.requiredNodeId[1] = y;
                        result.requiredNodeId[2] = y - 2;
                        result.requiredNodeAngle2[2] = 240.0f;
                    } else {
                        result.requiredNodeId[0] = n - 2;
                        result.requiredNodeId[1] = y;
                    }
                } else {
                    result.requiredNodeId[0] = n - 2;
                    result.requiredNodeAngle2[0] = 60.0f;
                }
            } else if (2 * s + 1 <= t && t <= 4 * s) {
                // Segment A2: length 2*s
                int u = t - (2 * s + 1);
                angleSign = (u >= 2 && u % 2 == 0) ? 1.0f : -1.0f;
                int z0 = Dp + (2 * s - 3);
                if (u == 0) {
                    result.requiredNodeId[0] = n - 2;
                    result.requiredNodeAngle2[0] = 300.0f;
                } else if (u == 1) {
                    result.requiredNodeId[0] = n - 3;
                    result.requiredNodeId[1] = z0;
                    result.requiredNodeId[2] = z0 - 1;
                    result.requiredNodeAngle2[0] = 240.0f;
                    result.requiredNodeAngle2[1] = 120.0f;
                    result.requiredNodeAngle2[2] = 300.0f;
                } else if (u % 2 == 0) {
                    result.requiredNodeId[0] = n - 2;
                    result.requiredNodeAngle2[0] = 60.0f;
                } else {
                    int j = (u - 1) / 2;
                    int z = z0 - 2 * (j - 1) - 1;
                    result.requiredNodeAngle2[0] = 300.0f;
                    result.requiredNodeAngle2[1] = (u == 3) ? 240.0f : 180.0f;
                    if (j < s - 1) {
                        result.requiredNodeId[0] = n - 2;
                        result.requiredNodeId[1] = z;
                        result.requiredNodeId[2] = z - 2;
                        result.requiredNodeAngle2[2] = 240.0f;
                    } else {
                        result.requiredNodeId[0] = n - 2;
                        result.requiredNodeId[1] = z;
                    }
                }
            } else if (4 * s + 1 <= t && t <= 5 * s - 1) {
                // Segment C1: length s - 1
                angleSign = -1.0f;
                int j = t - (4 * s + 1);
                result.requiredNodeId[0] = Dp - j;
                result.requiredNodeId[1] = Dp - j - 1;
                result.requiredNodeAngle2[0] = (j == 0) ? 120.0f : 240.0f;
                result.requiredNodeAngle2[1] = 300.0f;
            } else if (5 * s <= t && t <= 6 * s) {
                // Segment C2: length s + 1
                angleSign = 1.0f;
                int u = t - 5 * s;
                int E = D + 5 * s - 2;
                if (u == 0) {
                    result.requiredNodeId[0] = n - 2;
                    result.requiredNodeAngle2[0] = 60.0f;
                } else if (u == s) {
                    result.requiredNodeId[0] = E - (s - 1);
                    result.requiredNodeAngle2[0] = 120.0f;
                } else {
                    result.requiredNodeId[0] = E - (u - 1);
                    result.requiredNodeId[1] = E - u;
                    result.requiredNodeAngle2[0] = 120.0f;
                    result.requiredNodeAngle2[1] = 60.0f;
                }
            }
        }
    }

    if (result.requiredNodeId[0] != -1) {
        result.requiredNodeAngle1[0] = angleSign * 120.0f;
    }
    if (result.requiredNodeId[1] != -1) {
        result.requiredNodeAngle1[1] = angleSign * 60.0f;
    }

    result.numAdditionalConnections = 0;
    if (result.requiredNodeId[0] != -1) {
        ++result.numAdditionalConnections;
    }
    if (result.requiredNodeId[1] != -1) {
        ++result.numAdditionalConnections;
    }
    if (result.requiredNodeId[2] != -1) {
        ++result.numAdditionalConnections;
    }

    ++_nodePos;
    return result;
}

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForTube()
{
    ShapeGeneratorResult result;

    if (_nodePos == 0) {
        result.angle = 0;
    } else if (_nodePos == 1) {
        result.angle = 120.0f;
    } else {
        auto posInGroup = (_nodePos - 2) % 3;
        auto sign = ((_nodePos - 2) / 3 % 2 == 0) ? 1.0f : -1.0f;

        if (posInGroup == 0) {
            result.angle = sign * 60.0f;
            result.numAdditionalConnections = 1;
            result.requiredNodeId[0] = _connectedNodePos1;
            result.requiredNodeAngle1[0] = sign * 120.0f;
            result.requiredNodeAngle2[0] = sign > 0 ? (_nodePos == 2 ? 60.0f : 120.0f) : 240.0f;
        } else if (posInGroup == 1) {
            result.angle = -sign * 120.0f;
            result.numAdditionalConnections = 1;
            result.requiredNodeId[0] = _connectedNodePos1;
            result.requiredNodeAngle1[0] = sign * 120.0f;
            result.requiredNodeAngle2[0] = _nodePos == 3 ? 120.0f : 180.0f;
            if (_connectedNodePos1 > 0) {
                result.numAdditionalConnections = 2;
                result.requiredNodeId[1] = _connectedNodePos1 - 1;
                result.requiredNodeAngle1[1] = sign * 60.0f;
                result.requiredNodeAngle2[1] = sign > 0 ? 60.0f : 300.0f;
            }
        } else {
            result.angle = -sign * 60.0f;
            result.numAdditionalConnections = 1;
            result.requiredNodeId[0] = _nodePos - 2;
            result.requiredNodeAngle1[0] = -sign * 120.0f;
            result.requiredNodeAngle2[0] = sign > 0 ? 300.0f : 60.0f;
            _connectedNodePos1 = _nodePos - 2;
        }
    }

    ++_nodePos;
    return result;
}

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForLargeLolli()
{
    // Builds a diamond-shaped hexagonal head (19 nodes) followed by a straight tail.
    // Head layout:
    //     00  01  08
    //   03  02  07  09
    // 04  05  06  11  10
    //   15  14  13  12
    //     16  17  18
    // Tail: 19, 20, 21, ... extending diagonally from node 18.
    // Entry fields: {angle, requiredNodeId0, requiredNodeAngle1_0, requiredNodeAngle2_0, ...}
    ShapeGeneratorResult result;

    if (_nodePos < 19) {
        struct Entry
        {
            float angle;
            int r0;
            float a10;
            float a20;
            int r1;
            float a11;
            float a21;
            int r2;
            float a12;
            float a22;
        };
        Entry table[19] = {
            {0.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f},        // 00
            {120.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f},      // 01
            {60.0f, 0, 120.0f, 60.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f},     // 02
            {-60.0f, 0, 120.0f, 120.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f},   // 03
            {-120.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f},     // 04
            {0.0f, 3, -120.0f, 300.0f, 2, -60.0f, 300.0f, -1, 0.0f, 0.0f}, // 05
            {-60.0f, 2, -120.0f, 240.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f},  // 06
            {0.0f, 2, -120.0f, 180.0f, 1, -60.0f, 300.0f, -1, 0.0f, 0.0f}, // 07
            {120.0f, 1, -120.0f, 240.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f},  // 08
            {0.0f, 7, 120.0f, 60.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f},      // 09
            {120.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f},      // 10
            {-120.0f, 9, 120.0f, 60.0f, 7, 60.0f, 120.0f, 6, 0.0f, 60.0f}, // 11
            {120.0f, 10, -120.0f, 300.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f}, // 12
            {0.0f, 11, 120.0f, 60.0f, 6, 60.0f, 120.0f, -1, 0.0f, 0.0f},   // 13
            {0.0f, 6, 120.0f, 180.0f, 5, 60.0f, 60.0f, -1, 0.0f, 0.0f},    // 14
            {-120.0f, 5, 120.0f, 120.0f, 4, 60.0f, 60.0f, -1, 0.0f, 0.0f}, // 15
            {-60.0f, 14, -120.0f, 300.0f, -1, 0.0f, 0.0f, -1, 0.0f, 0.0f}, // 16
            {0.0f, 14, -120.0f, 240.0f, 13, -60.0f, 300.0f, -1, 0.0f, 0.0f}, // 17
            {60.0f, 13, -120.0f, 240.0f, 12, -60.0f, 300.0f, -1, 0.0f, 0.0f}, // 18
        };

        auto const& e = table[_nodePos];
        result.angle = e.angle;
        result.requiredNodeId[0] = e.r0;
        result.requiredNodeAngle1[0] = e.a10;
        result.requiredNodeAngle2[0] = e.a20;
        result.requiredNodeId[1] = e.r1;
        result.requiredNodeAngle1[1] = e.a11;
        result.requiredNodeAngle2[1] = e.a21;
        result.requiredNodeId[2] = e.r2;
        result.requiredNodeAngle1[2] = e.a12;
        result.requiredNodeAngle2[2] = e.a22;

        result.numAdditionalConnections = 0;
        if (e.r0 != -1) {
            ++result.numAdditionalConnections;
        }
        if (e.r1 != -1) {
            ++result.numAdditionalConnections;
        }
        if (e.r2 != -1) {
            ++result.numAdditionalConnections;
        }
    } else {
        result.angle = 0.0f;
        result.numAdditionalConnections = 0;
    }

    ++_nodePos;
    return result;
}

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForSmallLolli()
{
    ShapeGeneratorResult result;

    switch (_nodePos) {
    case 0:
        result.angle = 0.0f;
        break;
    case 1:
        result.angle = 60.0f;
        break;
    case 2:
        result.angle = 120.0f;
        break;
    case 3:
        result.angle = 0.0f;
        result.numAdditionalConnections = 2;
        result.requiredNodeId[0] = 1;
        result.requiredNodeAngle1[0] = 120.0f;
        result.requiredNodeAngle2[0] = 60.0f;
        result.requiredNodeId[1] = 0;
        result.requiredNodeAngle1[1] = 60.0f;
        result.requiredNodeAngle2[1] = 60.0f;
        break;
    case 4:
        result.angle = -120.0f;
        result.numAdditionalConnections = 1;
        result.requiredNodeId[0] = 0;
        result.requiredNodeAngle1[0] = 120.0f;
        result.requiredNodeAngle2[0] = 120.0f;
        break;
    case 5:
        result.angle = -60.0f;
        result.numAdditionalConnections = 1;
        result.requiredNodeId[0] = 3;
        result.requiredNodeAngle1[0] = -120.0f;
        result.requiredNodeAngle2[0] = 300.0f;
        break;
    case 6:
        result.angle = 60.0f;
        result.numAdditionalConnections = 2;
        result.requiredNodeId[0] = 3;
        result.requiredNodeAngle1[0] = -120.0f;
        result.requiredNodeAngle2[0] = 240.0f;
        result.requiredNodeId[1] = 2;
        result.requiredNodeAngle1[1] = -60.0f;
        result.requiredNodeAngle2[1] = 300.0f;
        break;
    default:
        result.angle = 0.0f;
        break;
    }

    ++_nodePos;
    return result;
}

HOST_DEVICE ShapeGeneratorResult ShapeGenerator::generateNextConstructionDataForZigzag()
{
    ShapeGeneratorResult result;
    auto mod8 = _nodePos % 8;

    if (mod8 == 2) {
        result.angle = 120.0f;
    } else if (mod8 == 3) {
        result.angle = 0.0f;
        result.numAdditionalConnections = 1;
        result.requiredNodeId[0] = _nodePos - 2;
        result.requiredNodeAngle1[0] = 120.0f;
        result.requiredNodeAngle2[0] = 60.0f;
    } else if (mod8 == 6) {
        result.angle = -120.0f;
    } else if (mod8 == 7) {
        result.angle = 0.0f;
        result.numAdditionalConnections = 1;
        result.requiredNodeId[0] = _nodePos - 2;
        result.requiredNodeAngle1[0] = -120.0f;
        result.requiredNodeAngle2[0] = 300.0f;
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
        result.requiredNodeId[0] = -1;
        result.requiredNodeId[1] = -1;
    } else if (_edgePos < 6) {
        result.angle = 60.0f;
        result.numAdditionalConnections = 1;
        result.requiredNodeId[0] = 0;
        result.requiredNodeId[1] = -1;
    } else {
        result.angle = _nodePos < edgeLength - 1 ? 0.0f : 60.0f;

        if (_nodePos < edgeLength - 1) {
            result.numAdditionalConnections = 2;
            result.requiredNodeId[0] = _connectedNodePos1;
            result.requiredNodeId[1] = _connectedNodePos1 + 1;
        } else {
            result.numAdditionalConnections = 1;
            result.requiredNodeId[0] = _connectedNodePos1;
            result.requiredNodeId[1] = -1;
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
