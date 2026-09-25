#pragma once

#include <Base/Singleton.h>

#include "Descs.h"
#include "GenomeDesc.h"

class DescValidationService
{
    MAKE_SINGLETON(DescValidationService);

public:
    void validateAndCorrect(GenomeDesc& genome);
    void validateAndCorrect(ExtendedObjectDesc& extendedObject);
};
