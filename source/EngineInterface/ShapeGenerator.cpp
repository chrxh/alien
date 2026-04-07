#include "ShapeGenerator.h"

#include <algorithm>
#include <array>
#include <cmath>

class _SegmentGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;
        result.angle = 0;
        result.numAdditionalConnections = 0;
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

    float getPreferredFrontAngle() override { return 0.0f; }
};

class _TriangleGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;
        auto edgeLength = std::max(2, _edgePos + 1);
        result.angle = _nodePos < edgeLength - 1 ? 0 : 120.0f;
        if (_edgePos == 0) {
            result.numAdditionalConnections = 0;
        } else if (_edgePos == 1) {
            result.numAdditionalConnections = _nodePos == 0 ? 1 : 0;
        } else {
            if (_nodePos == edgeLength - 1) {
                result.numAdditionalConnections = 0;
            } else if (_nodePos == edgeLength - 2) {
                result.numAdditionalConnections = 1;
            } else {
                result.numAdditionalConnections = 2;
            }
        }
        if (++_nodePos == edgeLength) {
            _nodePos = 0;
            ++_edgePos;
        }
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

    float getPreferredFrontAngle() override { return -30.0f; }

private:
    int _nodePos = 0;
    int _edgePos = 0;
};

class _RectangleGenerator : public _ShapeGenerator
{
public:
    // Builds a growing square (quadrat): each ring k adds an L-shaped border
    // extending the (k-1)x(k-1) square to k×k.
    // Even rings (Type B): go right 1 step, up k-1 steps, left k-1 steps.
    // Odd rings  (Type A): go up 1 step, right k-1 steps, down k-1 steps.
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;
        auto k = _edgePos + 1;
        auto p = _nodePos;

        if (_edgePos == 0) {
            result.angle = 0.0f;
            result.numAdditionalConnections = 0;
        } else {
            auto isTypeB = (k % 2 == 0);
            if (p == 0 || p == k - 1) {
                result.angle = isTypeB ? 90.0f : -90.0f;
            } else if (p == 2 * k - 2) {
                result.angle = isTypeB ? -90.0f : 90.0f;
            } else {
                result.angle = 0.0f;
            }
            result.numAdditionalConnections = (p == 0 || p == k - 1) ? 0 : 1;
        }

        if (++_nodePos > 2 * k - 2) {
            _nodePos = 0;
            ++_edgePos;
        }
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_90; }

    float getPreferredFrontAngle() override { return 0.0f; }

private:
    int _nodePos = 0;
    int _edgePos = 0;
};

class _HexagonGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;
        auto const outgoingDir = getOutgoingDir();
        result.angle = _nodeIndex == 0 ? 120.0f : static_cast<float>(normalizeDirDelta(outgoingDir - _incomingDir) * 60);
        result.numAdditionalConnections = countAdditionalConnections();
        advance(outgoingDir);
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

    float getPreferredFrontAngle() override { return 0.0f; }

private:
    static constexpr std::array<int, 6> DIR_Q = {1, 0, -1, -1, 0, 1};
    static constexpr std::array<int, 6> DIR_R = {0, 1, 1, 0, -1, -1};

    static int normalizeDirDelta(int delta)
    {
        auto result = delta;
        while (result <= -3) {
            result += 6;
        }
        while (result > 3) {
            result -= 6;
        }
        return result;
    }

    static int getRingMoveDir(int ringSize, int ringStep)
    {
        if (ringSize == 2) {
            static constexpr std::array<int, 6> RING2 = {0, 2, 0, 2, 3, 4};  // E, SW, E, SW, W, NW
            return RING2.at(ringStep);
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

    int getOutgoingDir() const
    {
        if (_ringSize == 1) {
            return getRingMoveDir(2, 0);
        }
        auto const nodesInCurrentRing = 6 * (_ringSize - 1);
        if (_ringPos + 1 < nodesInCurrentRing) {
            return getRingMoveDir(_ringSize, _ringPos + 1);
        }
        return getRingMoveDir(_ringSize + 1, 0);
    }

    bool isNeighbor(int q1, int r1, int q2, int r2) const
    {
        auto const dq = q2 - q1;
        auto const dr = r2 - r1;
        auto const ds = -dq - dr;
        return std::max({std::abs(dq), std::abs(dr), std::abs(ds)}) == 1;
    }

    int getRingStartIndex(int ringSize) const
    {
        if (ringSize <= 1) {
            return 0;
        }
        return 1 + 3 * (ringSize - 2) * (ringSize - 1);
    }

    int countAdditionalConnections() const
    {
        if (_nodeIndex == 0 || _ringSize == 1) {
            return 0;
        }

        auto const previousNodeId = _nodeIndex - 1;
        auto const previousRingSize = _ringSize - 1;
        auto const previousRingStartIndex = getRingStartIndex(previousRingSize);
        auto const previousRingNodeCount = previousRingSize == 1 ? 1 : 6 * (previousRingSize - 1);

        auto previousQ = _prevRingStartQ;
        auto previousR = _prevRingStartR;
        auto numAdditionalConnections = 0;
        for (int pos = 0; pos < previousRingNodeCount; ++pos) {
            auto const nodeId = previousRingStartIndex + pos;
            if (nodeId != previousNodeId && isNeighbor(_q, _r, previousQ, previousR)) {
                ++numAdditionalConnections;
            }

            if (pos + 1 < previousRingNodeCount) {
                auto const dir = getRingMoveDir(previousRingSize, pos + 1);
                previousQ += DIR_Q.at(dir);
                previousR += DIR_R.at(dir);
            }
        }

        // Search current ring (positions 0 to _ringPos-2, skipping immediate predecessor)
        if (_ringPos >= 2) {
            auto const currentRingStartIndex = getRingStartIndex(_ringSize);
            auto currentQ = _currentRingStartQ;
            auto currentR = _currentRingStartR;
            for (int pos = 0; pos <= _ringPos - 2; ++pos) {
                auto const nodeId = currentRingStartIndex + pos;
                if (nodeId != previousNodeId && isNeighbor(_q, _r, currentQ, currentR)) {
                    ++numAdditionalConnections;
                }

                if (pos + 1 <= _ringPos - 2) {
                    auto const dir = getRingMoveDir(_ringSize, pos + 1);
                    currentQ += DIR_Q.at(dir);
                    currentR += DIR_R.at(dir);
                }
            }
        }

        return numAdditionalConnections;
    }

    void advance(int outgoingDir)
    {
        _q += DIR_Q.at(outgoingDir);
        _r += DIR_R.at(outgoingDir);
        _incomingDir = outgoingDir;
        ++_nodeIndex;

        if (_ringSize == 1) {
            _ringSize = 2;
            _ringPos = 0;
            _prevRingStartQ = 0;
            _prevRingStartR = 0;
            _currentRingStartQ = _q;
            _currentRingStartR = _r;
            return;
        }

        auto const nodesInCurrentRing = 6 * (_ringSize - 1);
        if (++_ringPos >= nodesInCurrentRing) {
            _ringPos = 0;
            ++_ringSize;
            _prevRingStartQ = _currentRingStartQ;
            _prevRingStartR = _currentRingStartR;
            _currentRingStartQ = _q;
            _currentRingStartR = _r;
        }
    }

    int _nodeIndex = 0;
    int _ringSize = 1;
    int _ringPos = 0;
    int _q = 0;
    int _r = 0;
    int _incomingDir = 0;  // Direction from previous node to current node.
    int _currentRingStartQ = 0;
    int _currentRingStartR = 0;
    int _prevRingStartQ = 0;
    int _prevRingStartR = 0;
};

class _LoopGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;

        auto edgeLength = (_edgePos + 1) / 6 + 1;
        if (_edgePos % 6 == 0) {
            --edgeLength;
        }

        if (_edgePos < 5) {
            result.angle = 60.0f;
            result.numAdditionalConnections = 0;
        } else if (_edgePos == 5) {
            result.angle = _nodePos == 0 ? 0.0f : 60.0f;
            result.numAdditionalConnections = 1;
        } else {
            result.angle = _nodePos < edgeLength - 1 ? 0.0f : 60.0f;
            result.numAdditionalConnections = _nodePos < edgeLength - 1 ? 2 : 1;
        }

        if (++_nodePos >= edgeLength) {
            _nodePos = 0;
            ++_edgePos;
        }
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

    float getPreferredFrontAngle() override { return 0.0f; }

private:
    int _nodePos = 0;
    int _edgePos = 0;
};

class _TubeGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;
        if (_pos % 6 == 0) {
            result.angle = 0;
            result.numAdditionalConnections = 2;
        }
        if (_pos % 6 == 1) {
            result.angle = 60.0f;
            result.numAdditionalConnections = _pos == 1 ? 0 : 1;
        }
        if (_pos % 6 == 2) {
            result.angle = 120.0f;
            result.numAdditionalConnections = 0;
        }
        if (_pos % 6 == 3) {
            result.angle = 0;
            result.numAdditionalConnections = 2;
        }
        if (_pos % 6 == 4) {
            result.angle = -120.0f;
            result.numAdditionalConnections = _pos == 4 ? 1 : 2;
        }
        if (_pos % 6 == 5) {
            result.angle = -60.0f;
            result.numAdditionalConnections = 1;
        }
        ++_pos;

        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

    float getPreferredFrontAngle() override { return 240.0f; }

private:
    int _pos = 0;
};

class _LolliGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;

        if (_edgePos < 12 || _nodePos == 0) {
            auto edgeLength = _edgePos / 6 + 1;
            if (_edgePos % 6 == 1) {
                --edgeLength;
            }

            if (_edgePos < 2) {
                result.angle = 120.0f;
                result.numAdditionalConnections = 0;
            } else if (_edgePos < 6) {
                result.angle = 60.0f;
                result.numAdditionalConnections = 1;
            } else {
                result.angle = _nodePos < edgeLength - 1 ? 0.0f : 60.0f;
                result.numAdditionalConnections = _nodePos < edgeLength - 1 ? 2 : 1;
            }

            if (++_nodePos >= edgeLength) {
                _nodePos = 0;
                ++_edgePos;
            }
        } else {
            result.angle = _nodePos == 1 ? -60.0f : 0.0f;
            result.numAdditionalConnections = _nodePos == 1 ? 2 : 0;
            ++_nodePos;
        }
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

    float getPreferredFrontAngle() override { return 120.0f; }

private:
    int _nodePos = 0;
    int _edgePos = 0;
};

class _SmallLolliGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
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
            } else {
                result.angle = 60.0f;
                result.numAdditionalConnections = 1;
            }

            if (++_nodePos >= edgeLength) {
                _nodePos = 0;
                ++_edgePos;
            }
        } else {
            result.angle = _nodePos == 0 ? -60.0f : 0.0f;
            result.numAdditionalConnections = _nodePos == 0 ? 2 : 0;
            ++_nodePos;
        }
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

    float getPreferredFrontAngle() override { return 120.0f; }

private:
    int _nodePos = 0;
    int _edgePos = 0;
};

class _ZigzagGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;
        if (_nodePos % 4 == 0) {
            result.angle = 120.0f;
            result.numAdditionalConnections = 0;
        }
        if (_nodePos % 4 == 1) {
            result.angle = 0;
            result.numAdditionalConnections = _nodePos == 1 ? 0 : 1;
        }
        if (_nodePos % 4 == 2) {
            result.angle = -120.0f;
            result.numAdditionalConnections = 0;
        }
        if (_nodePos % 4 == 3) {
            result.angle = 0;
            result.numAdditionalConnections = 1;
        }
        ++_nodePos;
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

    float getPreferredFrontAngle() override { return 120.0f; }

private:
    int _nodePos = 0;
};

ShapeGenerator ShapeGeneratorFactory::create(ConstructorShape shape)
{
    switch (shape) {
    case ConstructorShape_Segment:
        return std::make_shared<_SegmentGenerator>();
    case ConstructorShape_Triangle:
        return std::make_shared<_TriangleGenerator>();
    case ConstructorShape_Rectangle:
        return std::make_shared<_RectangleGenerator>();
    case ConstructorShape_Hexagon:
        return std::make_shared<_HexagonGenerator>();
    case ConstructorShape_Loop:
        return std::make_shared<_LoopGenerator>();
    case ConstructorShape_Tube:
        return std::make_shared<_TubeGenerator>();
    case ConstructorShape_Lolli:
        return std::make_shared<_LolliGenerator>();
    case ConstructorShape_SmallLolli:
        return std::make_shared<_SmallLolliGenerator>();
    case ConstructorShape_Zigzag:
        return std::make_shared<_ZigzagGenerator>();
    }
    return nullptr;
}
