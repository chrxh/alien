#pragma once

#include <vector>

#include "Base/Definitions.h"
#include "EngineInterface/ElementaryTypes.h"

struct OverlayElementDescription
{
    Enums::CellFunction::Type cellType;
    RealVector2D pos;
};

struct OverlayDescription 
{
    std::vector<OverlayElementDescription> elements;
};