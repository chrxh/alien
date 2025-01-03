#pragma once

#include <vector>

#include "Base/Definitions.h"
#include "EngineInterface/CellFunctionConstants.h"

struct OverlayElementDescription
{
    uint64_t id;
    bool cell;  //false = energy particle
    CellFunction cellType;
    RealVector2D pos;
    int selected;
};

struct OverlayDescription 
{
    std::vector<OverlayElementDescription> elements;
};