#pragma once

#include <vector>

#include "Base/Definitions.h"
#include "EngineInterface/ElementaryTypes.h"

struct OverlayElementDescription
{
    uint64_t id;
    bool cell;  //false = energy particle
    Enums::CellFunction::Type cellType;
    RealVector2D pos;
    int branchNumber;
    int selected;
};

struct OverlayDescription 
{
    std::vector<OverlayElementDescription> elements;
};