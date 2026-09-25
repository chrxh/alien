#pragma once

#include <vector>

#include <Base/Definitions.h>
#include <Base/Macros.h>
#include <Base/RgbImage.h>
#include <Base/Singleton.h>

#include "Colors.h"
#include "Descs.h"

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
    ContentDesc
    createFreehandStroke(ObjectProperties const& properties, std::vector<RealVector2D> const& points, float pencilRadius, IntVector2D const& worldSize) const;
    ContentDesc createPatternFromImage(RgbImage const& image, RealVector2D const& center, ColorVector<FloatColorRGB> const& customizationColors) const;

    std::vector<RealVector2D> calcBezierCurvePath(std::vector<RealVector2D> const& controlPoints, float objectDistance) const;

private:
    ContentDesc createObjectNetwork(ObjectProperties const& properties, std::vector<RealVector2D> const& positions, float connectionDistance) const;
    ContentDesc convertToEnergyParticles(ObjectProperties const& properties, ContentDesc const& content) const;
    ObjectTypeDesc getObjectTypeDesc(ObjectProperties const& properties) const;
};
