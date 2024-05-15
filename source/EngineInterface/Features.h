#pragma once

struct Features
{
    bool genomeComplexityMeasurement = false;
    bool advancedAbsorptionControl = false;
    bool advancedAttackerControl = false;
    bool externalEnergyControl = false;
    bool cellColorTransitionRules = false;
    bool cellAgeLimiter = false;

    bool operator==(Features const& other) const;
};
