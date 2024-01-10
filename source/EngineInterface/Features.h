#pragma once

struct Features
{
    bool externalEnergyControl = false;
    bool cellColorTransitionRules = false;
    bool additionalAbsorptionControl = false;

    bool operator==(Features const& other) const;
};
