#pragma once

#include "Base/Singleton.h"

#include "GenomeDescriptions.h"

class GenomeDescriptionValidationService
{
    MAKE_SINGLETON(GenomeDescriptionValidationService);

public:
    void validateAndCorrect(GenomeDescription_New& genome);
};
