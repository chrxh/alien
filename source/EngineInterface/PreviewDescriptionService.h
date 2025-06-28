#pragma once

#include "Base/Singleton.h"

#include "CreatureDescription.h"
#include "SimulationParameters.h"
#include "PreviewDescriptions.h"

class PreviewDescriptionService
{
    MAKE_SINGLETON(PreviewDescriptionService);
public:
    PreviewDescription convert(CreatureDescription const& creature, std::optional<int> selectedNode, SimulationParameters const& parameters);
};

