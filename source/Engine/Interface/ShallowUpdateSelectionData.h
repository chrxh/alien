#pragma once

#include <optional>

#include <Base/MathTypes.h>

struct ShallowUpdateSelectionData
{
    bool considerClusters = true;
    bool glueOnContact = false;
    float posDeltaX = 0;
    float posDeltaY = 0;
    std::optional<RealVector2D> velocity;
    float angleDelta = 0;
    std::optional<float> angularVel;
};
