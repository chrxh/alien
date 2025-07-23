#pragma once

#include "Base/Singleton.h"

#include "Descriptions.h"
#include "GenomeDescription.h"
#include "SimulationParameters.h"
#include "PreviewDescriptions.h"

class PreviewDescriptionService
{
    MAKE_SINGLETON(PreviewDescriptionService);
public:
    PreviewDescription convert(CollectionDescription const& data) const;
};