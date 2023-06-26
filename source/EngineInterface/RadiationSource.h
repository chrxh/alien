#pragma once

/**
 * NOTE: header is also included in kernel code
 */

struct CircularRadiationSource
{
    float radius = 1;

    bool operator==(CircularRadiationSource const& other) const { return radius == other.radius; }
    bool operator!=(CircularRadiationSource const& other) const { return !operator==(other); }
};

struct RectangularRadiationSource
{
    float width = 100.0f;
    float height = 200.0f;

    bool operator==(RectangularRadiationSource const& other) const { return width == other.width && height == other.height; }
    bool operator!=(RectangularRadiationSource const& other) const { return !operator==(other); }
};

union RadiationSourceShapeData
{
    CircularRadiationSource circularRadiationSource;
    RectangularRadiationSource rectangularRadiationSource;
};

using RadiationSourceShapeType = int;
enum RadiationSourceShapeType_
{
    RadiationSourceShapeType_Circular,
    RadiationSourceShapeType_Rectangular
};

struct RadiationSource
{
    float posX = 0;
    float posY = 0;

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
        return posX == other.posX && posY == other.posY;
    }
    bool operator!=(RadiationSource const& other) const { return !operator==(other); }
};
