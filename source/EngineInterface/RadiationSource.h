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

union RadiationSourceShapeData
{
    CircularRadiationSource circularRadiationSource;
    RectangularRadiationSource rectangularRadiationSource;

    bool operator==(RectangularRadiationSource const&) const { return true; }
};

using RadiationSourceShapeType = int;
enum RadiationSourceShapeType_
{
    RadiationSourceShapeType_Circular,
    RadiationSourceShapeType_Rectangular
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

    RadiationSourceShapeType shapeType = RadiationSourceShapeType_Circular;
    RadiationSourceShapeData shapeData = {CircularRadiationSource()};

    bool operator==(RadiationSource const& other) const
    {
        if (shapeType != other.shapeType) {
            return false;
        }
        if (shapeType == RadiationSourceShapeType_Circular) {
            if (shapeData.circularRadiationSource != other.shapeData.circularRadiationSource) {
                return false;
            }
        }
        if (shapeType == RadiationSourceShapeType_Rectangular) {
            if (shapeData.rectangularRadiationSource != other.shapeData.rectangularRadiationSource) {
                return false;
            }
        }
        for (int i = 0, j = sizeof(Char64) / sizeof(char); i < j; ++i) {
            if (name[i] != other.name[i]) {
                return false;
            }
            if (name[i] == '\0') {
                break;
            }
        }
        return posX == other.posX && posY == other.posY && velX == other.velX && velY == other.velY && useAngle == other.useAngle && angle == other.angle
            && strength == other.strength && strengthPinned == other.strengthPinned;
    }
};
