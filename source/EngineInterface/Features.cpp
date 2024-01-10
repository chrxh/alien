#include "Features.h"

bool Features::operator==(Features const& other) const
{
    return externalEnergyControl == other.externalEnergyControl && cellColorTransitionRules == other.cellColorTransitionRules && additionalAbsorptionControl == other.additionalAbsorptionControl;
}
