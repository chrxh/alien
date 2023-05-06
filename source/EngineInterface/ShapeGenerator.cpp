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
};

class _TriangleGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
    {
        ShapeGeneratorResult result;
        auto edgeLength = std::max(2, _processedEdges + 1);
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

private:
    int _edgePos = 0;
    int _processedEdges = 0;
};

class _RectangleGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
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

private:
    int _edgePos = 0;
    int _processedEdges = 0;
};

class _HexagonGenerator : public _ShapeGenerator
{
public:
    ShapeGeneratorResult generateNextConstructionData() override
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

private:
    int _edgePos = 0;
    int _processedEdges = 0;
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
    }
    return nullptr;
}
