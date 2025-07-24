#pragma once

#include "Base/Singleton.h"

#include "Descriptions.h"
#include "GenomeDescription.h"
#include "SimulationParameters.h"
#include "PreviewDescriptions.h"

class PreviewDescriptionConverterService
{
    MAKE_SINGLETON(PreviewDescriptionConverterService);
public:
    PreviewDescription convert(CollectionDescription const& data) const;
};