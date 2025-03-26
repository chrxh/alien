#pragma once

#include <compare>

/**
 * NOTE: header is also included in kernel code
 */

struct ExpertSettingsToggles
{
    bool advancedAbsorptionControl = false;
    bool advancedAttackerControl = false;
    bool externalEnergyControl = false;
    bool customizeNeuronMutations = false;
    bool customizeDeletionMutations = false;
    bool cellColorTransitionRules = false;
    bool cellAgeLimiter = false;
    bool cellGlow = false;
    bool genomeComplexityMeasurement = false;

    bool operator==(ExpertSettingsToggles const&) const = default;
};
