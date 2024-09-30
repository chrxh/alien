#pragma once

#include "Definitions.h"

class _PersisterController
{
public:
    void saveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center);
};