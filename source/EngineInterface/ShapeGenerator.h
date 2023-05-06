#pragma once

#include <memory>
#include <optional>

#include "CellFunctionConstants.h"

struct ShapeGeneratorResult
{
    float angle;
    std::optional<int> numRequiredAdditionalConnections;
};

class _ShapeGenerator
{
public:
    virtual ShapeGeneratorResult generateNextConstructionData() = 0;
};
using ShapeGenerator = std::shared_ptr<_ShapeGenerator>;

class ShapeGeneratorFactory
{
public:
    static ShapeGenerator create(ConstructionShape shape);
};
