#pragma once

#include <memory>
#include <optional>

#include "CellFunctionConstants.h"
#include "Definitions.h"

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

class ShapeGeneratorFactory
{
public:
    static ShapeGenerator create(ConstructionShape shape);
};
