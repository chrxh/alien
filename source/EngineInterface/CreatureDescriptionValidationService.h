#pragma once

#include "Base/Singleton.h"

#include "GenomeDescription.h"

class CreatureDescriptionValidationService
{
    MAKE_SINGLETON(CreatureDescriptionValidationService);

public:
    void validateAndCorrect(GenomeDescription& creature);
};
