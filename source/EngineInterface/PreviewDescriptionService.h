#pragma once

#include "Base/Singleton.h"

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"
#include "PreviewDescriptions.h"

class PreviewDescriptionService
{
    MAKE_SINGLETON(PreviewDescriptionService);
public:
    PreviewDescription convert(GenomeDescription const& genome, std::optional<int> selectedNode, SimulationParameters const& parameters);
};

