#pragma once

#include <vector>

#include <Base/Singleton.h>

#include "Descs.h"

class MassOperationsService
{
    MAKE_SINGLETON(MassOperationsService);

public:
    void randomizeCellColors(ContentDesc& description, std::vector<int> const& colorCodes) const;
    void randomizeGenomeColors(ContentDesc& description, std::vector<int> const& colorCodes) const;
    void randomizeEnergies(ContentDesc& description, float minEnergy, float maxEnergy) const;
    void randomizeAges(ContentDesc& description, int minAge, int maxAge) const;
    void randomizeCountdowns(ContentDesc& description, int minValue, int maxValue) const;
    void randomizeLineageIds(ContentDesc& description) const;
    void randomizeGlow(ContentDesc& description, float minGlow, float maxGlow) const;
    void setMutationRates(ContentDesc& description, MutationRatesDesc const& mutationRates) const;
};
