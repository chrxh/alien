#include "Features.h"

bool Features::operator==(Features const& other) const
{
    return genomeComplexityMeasurement == other.genomeComplexityMeasurement && externalEnergyControl == other.externalEnergyControl
        && cellColorTransitionRules == other.cellColorTransitionRules && advancedAbsorptionControl == other.advancedAbsorptionControl
        && advancedAttackerControl == other.advancedAttackerControl && cellAgeLimiter == other.cellAgeLimiter && cellGlow == other.cellGlow
        && legacyModes == other.legacyModes && customizeNeuronMutations == other.customizeNeuronMutations;
}
