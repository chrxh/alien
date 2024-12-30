#pragma once

#include "SimulationParametersTypes.h"

/**
 * NOTE: header is also included in kernel code
 */

struct CircularRadiationSource
{
    float radius = 1;

    bool operator==(CircularRadiationSource const&) const = default;
};

struct RectangularRadiationSource
{
    float width = 100.0f;
    float height = 200.0f;

    bool operator==(RectangularRadiationSource const&) const = default;
};

using RadiationSourceShapeType = int;
enum RadiationSourceShapeType_
{
    RadiationSourceShapeType_Circular,
    RadiationSourceShapeType_Rectangular
};

union RadiationSourceShapeAlternatives
{
    CircularRadiationSource circularRadiationSource;
    RectangularRadiationSource rectangularRadiationSource;
};

struct RadiationSourceShape
{
    RadiationSourceShapeType type = RadiationSourceShapeType_Circular;
    RadiationSourceShapeAlternatives alternatives = {CircularRadiationSource()};

    bool operator==(RadiationSourceShape const& other) const
    {
        if (type != other.type) {
            return false;
        }
        if (type == RadiationSourceShapeType_Circular) {
            if (alternatives.circularRadiationSource != other.alternatives.circularRadiationSource) {
                return false;
            }
        }
        if (type == RadiationSourceShapeType_Rectangular) {
            if (alternatives.rectangularRadiationSource != other.alternatives.rectangularRadiationSource) {
                return false;
            }
        }
        return true;
    }
};

struct RadiationSource
{
    Char64 name = "<unnamed>";
    int locationIndex = -1;

    float strength = 0.0f;
    bool strengthPinned = false;
    float posX = 0;
    float posY = 0;
    float velX = 0;
    float velY = 0;
    bool useAngle = false;
    float angle = 0;

    RadiationSourceShape shape;

    bool operator==(RadiationSource const& other) const = default;
};
