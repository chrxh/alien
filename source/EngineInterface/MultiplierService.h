#pragma once

#include <Base/Definitions.h>
#include <Base/Macros.h>
#include <Base/Singleton.h>

#include "Descs.h"

class MultiplierService
{
    MAKE_SINGLETON(MultiplierService);

public:
    struct GridParameters
    {
        MEMBER(GridParameters, int, horizontalNumber, 10);
        MEMBER(GridParameters, float, horizontalDistance, 50.0f);
        MEMBER(GridParameters, float, horizontalAngleInc, 0);
        MEMBER(GridParameters, float, horizontalVelXinc, 0);
        MEMBER(GridParameters, float, horizontalVelYinc, 0);
        MEMBER(GridParameters, float, horizontalAngularVelInc, 0);
        MEMBER(GridParameters, int, verticalNumber, 10);
        MEMBER(GridParameters, float, verticalDistance, 50.0f);
        MEMBER(GridParameters, float, verticalAngleInc, 0);
        MEMBER(GridParameters, float, verticalVelXinc, 0);
        MEMBER(GridParameters, float, verticalVelYinc, 0);
        MEMBER(GridParameters, float, verticalAngularVelInc, 0);
    };
    ContentDesc multiplyInGrid(ContentDesc const& content, GridParameters const& parameters) const;

    struct RandomParameters
    {
        MEMBER(RandomParameters, int, number, 100);
        MEMBER(RandomParameters, float, minAngle, 0);
        MEMBER(RandomParameters, float, maxAngle, 360.0f);
        MEMBER(RandomParameters, float, minVelX, 0);
        MEMBER(RandomParameters, float, maxVelX, 0);
        MEMBER(RandomParameters, float, minVelY, 0);
        MEMBER(RandomParameters, float, maxVelY, 0);
        MEMBER(RandomParameters, float, minAngularVel, 0);
        MEMBER(RandomParameters, float, maxAngularVel, 0);
        MEMBER(RandomParameters, bool, overlappingCheck, false);
        MEMBER(RandomParameters, IntVector2D, maxDelta, IntVector2D());
    };
    struct RandomMultiplicationResult
    {
        ContentDesc content;
        bool overlappingCheckSuccessful = true;
    };
    RandomMultiplicationResult multiplyRandomly(ContentDesc const& content, RandomParameters const& parameters) const;
};
