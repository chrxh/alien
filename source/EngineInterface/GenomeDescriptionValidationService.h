#pragma once

#include "Base/Singleton.h"

#include "CreatureDescription.h"

class GenomeDescriptionValidationService
{
    MAKE_SINGLETON(GenomeDescriptionValidationService);

public:
    void validateAndCorrect(CreatureDescription& genome);
};
