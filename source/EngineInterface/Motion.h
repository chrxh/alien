#pragma once

/**
 * NOTE: header is also included in kernel code
 */

struct FluidMotion
{
    float smoothingLength = 0.8f;
    float viscosityStrength = 0.1f;
    float pressureStrength = 0.1f;

    bool operator==(FluidMotion const& other) const
    {
        return smoothingLength == other.smoothingLength && viscosityStrength == other.viscosityStrength && pressureStrength == other.pressureStrength;
    }
    bool operator!=(FluidMotion const& other) const { return !operator==(other); }
};

struct CollisionMotion
{
    float cellMaxCollisionDistance = 1.3f;
    float cellRepulsionStrength = 0.08f;

    bool operator==(CollisionMotion const& other) const
    {
        return cellMaxCollisionDistance == other.cellMaxCollisionDistance && cellRepulsionStrength == other.cellRepulsionStrength;
    }
    bool operator!=(CollisionMotion const& other) const { return !operator==(other); }
};

union MotionData
{
    FluidMotion fluidMotion;
    CollisionMotion collisionMotion;
};

using MotionType = int;
enum MotionType_
{
    MotionType_Fluid,
    MotionType_Collision
};
