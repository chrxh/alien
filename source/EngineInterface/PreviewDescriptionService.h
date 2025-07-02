#pragma once

#include "Base/Singleton.h"

#include "GenomeDescription.h"
#include "SimulationParameters.h"
#include "PreviewDescriptions.h"

class PreviewDescriptionService
{
    MAKE_SINGLETON(PreviewDescriptionService);
public:
    PreviewDescription convert(GenomeDescription const& creature, std::optional<int> selectedNode, SimulationParameters const& parameters);
};

