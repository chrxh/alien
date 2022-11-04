#pragma once

#include "EngineInterface/GenomeDescriptions.h"
#include "PreviewDescriptions.h"


class PreviewDescriptionConverter
{
public:
    static PreviewDescription convert(GenomeDescription const& genome);
};

