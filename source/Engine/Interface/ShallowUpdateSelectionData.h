#pragma once

struct ShallowUpdateSelectionData
{
    bool considerClusters = true;
    bool glueOnContact = false;
    float posDeltaX = 0;
    float posDeltaY = 0;
    float velX = 0;
    float velY = 0;
    float angleDelta = 0;
    float angularVel = 0;
};
