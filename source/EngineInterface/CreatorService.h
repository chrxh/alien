#pragma once

#include <cstdint>
#include <vector>

#include <Base/Definitions.h>
#include <Base/Macros.h>
#include <Base/Singleton.h>

#include "Descs.h"

struct RgbImage
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;  // 3 bytes per pixel, row by row
};

using CreationMaterial = int;
enum CreationMaterial_
{
    CreationMaterial_Solid,
    CreationMaterial_Fluid,
    CreationMaterial_FreeCell,
    CreationMaterial_EnergyParticle
};

class CreatorService
{
    MAKE_SINGLETON(CreatorService);

public:
    struct ObjectProperties
    {
        MEMBER(ObjectProperties, CreationMaterial, material, CreationMaterial_Solid);
        MEMBER(ObjectProperties, int, color, 0);
        MEMBER(ObjectProperties, float, energy, 100.0f);
        MEMBER(ObjectProperties, float, stiffness, 1.0f);
        MEMBER(ObjectProperties, float, glow, 0.0f);
        MEMBER(ObjectProperties, bool, isStatic, false);
        MEMBER(ObjectProperties, bool, sticky, false);
    };

    ContentDesc createSingleObject(ObjectProperties const& properties, RealVector2D const& pos) const;
    ContentDesc createRectangle(ObjectProperties const& properties, RealVector2D const& center, IntVector2D const& numObjects, float objectDistance) const;
    ContentDesc createHexagon(ObjectProperties const& properties, RealVector2D const& center, int layers, float objectDistance) const;
    ContentDesc createDisc(ObjectProperties const& properties, RealVector2D const& center, float outerRadius, float innerRadius, float objectDistance) const;
    ContentDesc createLine(ObjectProperties const& properties, std::vector<RealVector2D> const& points, float objectDistance) const;
    ContentDesc createCurve(ObjectProperties const& properties, std::vector<RealVector2D> const& controlPoints, float objectDistance) const;
    ContentDesc createPolygon(ObjectProperties const& properties, std::vector<RealVector2D> const& points, float objectDistance) const;
    ContentDesc createPencilDot(ObjectProperties const& properties, RealVector2D const& pos, float pencilRadius) const;
    ContentDesc createFreehandStroke(ObjectProperties const& properties, std::vector<RealVector2D> const& points, float pencilRadius) const;
    ContentDesc createPatternFromImage(RgbImage const& image, RealVector2D const& center) const;

    // Variants for arbitrary object types
    struct RectangleParameters
    {
        MEMBER(RectangleParameters, int, width, 10);
        MEMBER(RectangleParameters, int, height, 10);
        MEMBER(RectangleParameters, ObjectTypeDesc, objectType, SolidDesc());
        MEMBER(RectangleParameters, float, cellDistance, 1.0f);
        MEMBER(RectangleParameters, bool, connectObjects, true);
        MEMBER(RectangleParameters, float, stiffness, 1.0f);
        MEMBER(RectangleParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(RectangleParameters, bool, sticky, false);
        MEMBER(RectangleParameters, int, color, 0);
        MEMBER(RectangleParameters, bool, isStatic, false);
    };
    ContentDesc createRectangle(RectangleParameters const& parameters) const;

    struct HexagonParameters
    {
        MEMBER(HexagonParameters, int, layers, 10);
        MEMBER(HexagonParameters, ObjectTypeDesc, objectType, SolidDesc());
        MEMBER(HexagonParameters, float, cellDistance, 1.0f);
        MEMBER(HexagonParameters, bool, connectObjects, true);
        MEMBER(HexagonParameters, float, stiffness, 1.0f);
        MEMBER(HexagonParameters, RealVector2D, center, RealVector2D({0, 0}));
        MEMBER(HexagonParameters, bool, sticky, false);
        MEMBER(HexagonParameters, int, color, 0);
        MEMBER(HexagonParameters, bool, isStatic, false);
    };
    ContentDesc createHexagon(HexagonParameters const& parameters) const;

    std::vector<RealVector2D> calcBezierCurvePath(std::vector<RealVector2D> const& controlPoints, float objectDistance) const;

private:
    ContentDesc createObjectNetwork(ObjectProperties const& properties, std::vector<RealVector2D> const& positions, float connectionDistance) const;
    ContentDesc convertToEnergyParticles(ObjectProperties const& properties, ContentDesc const& content) const;
    ObjectTypeDesc getObjectTypeDesc(ObjectProperties const& properties) const;
};
