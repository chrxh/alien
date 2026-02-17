#pragma once

#include <Base/Singleton.h>

#include "GenomeDesc.h"

class GenomeDescValidationService
{
    MAKE_SINGLETON(GenomeDescValidationService);

public:
    void validateAndCorrect(GenomeDesc& genome);
};
