#pragma once

struct ShallowUpdateSelectionData
{
    bool considerClusters = true;
    float posDeltaX = 0;
    float posDeltaY = 0;
    float velDeltaX = 0;
    float velDeltaY = 0;
    float angleDelta = 0;
    float angularVelDelta = 0;
};
