#pragma once

/**
 * NOTE: header is also included in kernel code
 */

struct FluidMotion
{
    float smoothingLength = 0.8f;
    float viscosityStrength = 0.1f;
    float pressureStrength = 0.1f;

    bool operator==(FluidMotion const&) const = default;
};

struct CollisionMotion
{
    float cellMaxCollisionDistance = 1.3f;
    float cellRepulsionStrength = 0.08f;

    bool operator==(CollisionMotion const&) const = default;
};

union MotionDataAlternatives
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

struct MotionData
{
    MotionType type = MotionType_Fluid;
    MotionDataAlternatives alternatives = {FluidMotion()};

    bool operator==(MotionData const& other) const
    {
        if (type != other.type) {
            return false;
        }
        if (type == MotionType_Fluid) {
            if (alternatives.fluidMotion != other.alternatives.fluidMotion) {
                return false;
            }
        }
        if (type == MotionType_Collision) {
            if (alternatives.collisionMotion != other.alternatives.collisionMotion) {
                return false;
            }
        }
        return true;
    }
};
