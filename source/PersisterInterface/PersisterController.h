#pragma once

#include "Definitions.h"

class _PersisterController
{
public:
    virtual ~_PersisterController() = default;

    virtual void saveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center) = 0;
};