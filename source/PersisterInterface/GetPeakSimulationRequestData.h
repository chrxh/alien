#pragma once

#include "SharedDeserializedSimulation.h"

struct GetPeakSimulationRequestData
{
    SharedDeserializedSimulation peakDeserializedSimulation;
    float zoom = 1.0f;
    RealVector2D center;
};