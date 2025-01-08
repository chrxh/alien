#include "ShapeGenerator.h"

class _SegmentGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;
        result.angle = 0;
        result.numRequiredAdditionalConnections = 0;
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }
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
            result.numRequiredAdditionalConnections = 0;
        } else if (_edgePos == 1) {
            result.numRequiredAdditionalConnections = _nodePos == 0 ? 1 : 0;
        } else {
            if (_nodePos == edgeLength - 1) {
                result.numRequiredAdditionalConnections = 0;
            } else if (_nodePos == edgeLength - 2) {
                result.numRequiredAdditionalConnections = 1;
            } else {
                result.numRequiredAdditionalConnections = 2;
            }
        }
        if (++_nodePos == edgeLength) {
            _nodePos = 0;
            ++_edgePos;
        }
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

private:
    int _nodePos = 0;
    int _edgePos = 0;
};

class _RectangleGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;
        if (_edgePos == 0) {
            result.angle = 0.0f;
            result.numRequiredAdditionalConnections = 0;
        } else if (_edgePos == 1) {
            result.angle = 90.0f;
            result.numRequiredAdditionalConnections = 0;
        } else {
            result.angle = _nodePos == 0 ? 90.0f : 0.0f;
            result.numRequiredAdditionalConnections = _nodePos == 0 ? 0 : 1;
        }

        auto edgeLength = _edgePos / 2;
        if (++_nodePos > edgeLength) {
            _nodePos = 0;
            ++_edgePos;
        }
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_90; }

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
        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

private:
    int _nodePos = 0;
    int _edgePos = 0;
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

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

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
            result.numRequiredAdditionalConnections = 2;
        }
        if (_pos % 6 == 1) {
            result.angle = 60.0f;
            result.numRequiredAdditionalConnections = _pos == 1 ? 0 : 1;
        }
        if (_pos % 6 == 2) {
            result.angle = 120.0f;
            result.numRequiredAdditionalConnections = 0;
        }
        if (_pos % 6 == 3) {
            result.angle = 0;
            result.numRequiredAdditionalConnections = 2;
        }
        if (_pos % 6 == 4) {
            result.angle = -120.0f;
            result.numRequiredAdditionalConnections = _pos == 4 ? 1 : 2;
        }
        if (_pos % 6 == 5) {
            result.angle = -60.0f;
            result.numRequiredAdditionalConnections = 1;
        }
        ++_pos;

        return result;
    }

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

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

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

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

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

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

    ConstructorAngleAlignment getConstructorAngleAlignment() override { return ConstructorAngleAlignment_60; }

private:
    int _nodePos = 0;
};

ShapeGenerator ShapeGeneratorFactory::create(ConstructionShape shape)
{
    switch (shape) {
    case ConstructionShape_Segment:
        return std::make_shared<_SegmentGenerator>();
    case ConstructionShape_Triangle:
        return std::make_shared<_TriangleGenerator>();
    case ConstructionShape_Rectangle:
        return std::make_shared<_RectangleGenerator>();
    case ConstructionShape_Hexagon:
        return std::make_shared<_HexagonGenerator>();
    case ConstructionShape_Loop:
        return std::make_shared<_LoopGenerator>();
    case ConstructionShape_Tube:
        return std::make_shared<_TubeGenerator>();
    case ConstructionShape_Lolli:
        return std::make_shared<_LolliGenerator>();
    case ConstructionShape_SmallLolli:
        return std::make_shared<_SmallLolliGenerator>();
    case ConstructionShape_Zigzag:
        return std::make_shared<_ZigzagGenerator>();
    }
    return nullptr;
}
