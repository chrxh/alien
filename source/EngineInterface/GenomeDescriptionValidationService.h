#pragma once

#include <Base/Singleton.h>

#include "GenomeDescription.h"

class GenomeDescriptionValidationService
{
    MAKE_SINGLETON(GenomeDescriptionValidationService);

public:
    void validateAndCorrect(GenomeDescription& genome);
};
