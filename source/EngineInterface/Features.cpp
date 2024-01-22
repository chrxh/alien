#include "Features.h"

bool Features::operator==(Features const& other) const
{
    return externalEnergyControl == other.externalEnergyControl && cellColorTransitionRules == other.cellColorTransitionRules
        && advancedAbsorptionControl == other.advancedAbsorptionControl && advancedAttackerControl == other.advancedAttackerControl;
}
