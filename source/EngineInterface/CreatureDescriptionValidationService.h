#pragma once

#include "Base/Singleton.h"

#include "CreatureDescription.h"

class CreatureDescriptionValidationService
{
    MAKE_SINGLETON(CreatureDescriptionValidationService);

public:
    void validateAndCorrect(CreatureDescription& creature);
};
