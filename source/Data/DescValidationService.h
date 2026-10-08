#pragma once

#include <map>
#include <vector>

#include <Base/Singleton.h>

#include "Descs.h"
#include "GenomeDesc.h"
#include "GenomeIssue.h"

class DescValidationService
{
    MAKE_SINGLETON(DescValidationService);

public:
    void validateAndCorrect(GenomeDesc& genome);
    void validateAndCorrect(ExtendedObjectDesc& extendedObject);

    std::vector<GenomeIssue> findGenomeIssues(GenomeDesc const& genome) const;
    std::map<int, int> fixGenomeIssues(GenomeDesc& genome) const;
};
