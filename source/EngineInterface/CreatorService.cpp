#include "CreatorService.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>
#include <span>

#include <boost/range/adaptor/indexed.hpp>

#include <Base/Math.h>

#include "Colors.h"
#include "DescEditService.h"
#include "NumberGenerator.h"
#include "ObjectColoring.h"
#include "SimulationFacade.h"

namespace
{
    auto constexpr MaxNumObjects = size_t{1000000};
    auto constexpr SurfaceConnectionFactor = 1.7f;
    auto constexpr GridConnectionFactor = 1.1f;
    auto constexpr PathConnectionFactor = 1.5f;
    auto constexpr PencilConnectionDistance = 1.5f;
    auto constexpr ImageConnectionDistance = 1.5f;
    auto constexpr OverlapDistance = 0.5f;
}

ContentDesc CreatorService::createSingleObject(ObjectProperties const& properties, RealVector2D const& pos) const
{
    ContentDesc result;
    if (properties._material == CreationMaterial_EnergyParticle) {
        result._energies.emplace_back(EnergyDesc().pos(pos).energy(properties._energy).color(properties._color));
    } else {
        result._objects.emplace_back(ObjectDesc()
                                         .pos(pos)
                                         .stiffness(properties._stiffness)
                                         .color(properties._color)
                                         .isStatic(properties._isStatic)
                                         .sticky(properties._sticky)
                                         .type(getObjectTypeDesc(properties)));
    }
    return result;
}

namespace
{
    std::vector<RealVector2D> distributeInRectangle(IntVector2D const& numObjects, float distance)
    {
        std::vector<RealVector2D> result;
        for (auto i : std::views::iota(0, numObjects.x)) {
            for (auto j : std::views::iota(0, numObjects.y)) {
                result.emplace_back(RealVector2D{toFloat(i) * distance, toFloat(j) * distance});
            }
        }
        return result;
    }
}

ContentDesc CreatorService::createRectangle(ObjectProperties const& properties, RealVector2D const& center, IntVector2D const& numObjects, float objectDistance)
    const
{
    if (numObjects.x <= 0 || numObjects.y <= 0) {
        return {};
    }
    auto result = createObjectNetwork(properties, distributeInRectangle(numObjects, objectDistance), objectDistance * GridConnectionFactor);
    DescEditService::get().setCenter(result, center);
    return result;
}

namespace
{
    std::vector<RealVector2D> distributeInHexagon(int layers, float distance)
    {
        std::vector<RealVector2D> result;
        auto incY = sqrt(3.0) * distance / 2.0;
        for (auto j : std::views::iota(0, layers)) {
            for (auto i : std::views::iota(-(layers - 1), layers - j)) {
                auto x = toFloat(i * distance + j * distance / 2.0);
                result.emplace_back(RealVector2D{x, toFloat(-j * incY)});
                if (j > 0) {
                    result.emplace_back(RealVector2D{x, toFloat(j * incY)});
                }
            }
        }
        return result;
    }
}

ContentDesc CreatorService::createHexagon(ObjectProperties const& properties, RealVector2D const& center, int layers, float objectDistance) const
{
    if (layers <= 0) {
        return {};
    }
    auto result = createObjectNetwork(properties, distributeInHexagon(layers, objectDistance), objectDistance * SurfaceConnectionFactor);
    DescEditService::get().setCenter(result, center);
    return result;
}

ContentDesc
CreatorService::createDisc(ObjectProperties const& properties, RealVector2D const& center, float outerRadius, float innerRadius, float objectDistance) const
{
    if (innerRadius > outerRadius || innerRadius < 0 || outerRadius < 0) {
        return {};
    }

    ContentDesc result;
    auto const objectType = getObjectTypeDesc(properties);
    auto constexpr SmallValue = 0.01f;
    for (float radius = innerRadius; radius <= outerRadius + SmallValue && result._objects.size() < MaxNumObjects; radius += objectDistance) {
        float angleInc = [&] {
            if (radius > SmallValue) {
                auto angle = asinf(objectDistance / (2 * radius)) * 2 * toFloat(Const::RadToDeg);
                return 360.0f / floorf(360.0f / angle);
            }
            return 360.0f;
        }();
        for (auto angle = 0.0; angle < 360.0f - angleInc / 2; angle += angleInc) {
            result._objects.emplace_back(ObjectDesc()
                                             .id(NumberGenerator::get().createEntityId())
                                             .stiffness(properties._stiffness)
                                             .sticky(properties._sticky)
                                             .pos(Math::unitVectorOfAngle(angle) * radius)
                                             .color(properties._color)
                                             .isStatic(properties._isStatic)
                                             .type(objectType));
        }
    }

    if (properties._material == CreationMaterial_EnergyParticle) {
        result = convertToEnergyParticles(properties, result);
    } else {
        DescEditService::get().reconnectObjects(result, objectDistance * SurfaceConnectionFactor);
    }
    DescEditService::get().setCenter(result, center);
    return result;
}

namespace
{
    std::vector<RealVector2D> distributeAlongPath(std::vector<RealVector2D> const& path, float distance)
    {
        std::vector<RealVector2D> result;
        if (path.size() < 2 || distance < NEAR_ZERO) {
            return result;
        }
        result.emplace_back(path.front());
        auto pendingDistance = distance;
        for (auto const& [from, to] : std::views::zip(path, path | std::views::drop(1))) {
            auto segmentLength = Math::length(to - from);
            if (segmentLength < NEAR_ZERO) {
                continue;
            }
            auto direction = (to - from) / segmentLength;
            auto offset = pendingDistance;
            for (; offset < segmentLength + NEAR_ZERO && result.size() < MaxNumObjects; offset += distance) {
                result.emplace_back(from + direction * offset);
            }
            pendingDistance = offset - segmentLength;
        }
        return result;
    }
}

ContentDesc CreatorService::createLine(ObjectProperties const& properties, std::vector<RealVector2D> const& points, float objectDistance) const
{
    return createObjectNetwork(properties, distributeAlongPath(points, objectDistance), objectDistance * PathConnectionFactor);
}

ContentDesc CreatorService::createCurve(ObjectProperties const& properties, std::vector<RealVector2D> const& controlPoints, float objectDistance) const
{
    return createObjectNetwork(
        properties, distributeAlongPath(calcBezierCurvePath(controlPoints, objectDistance), objectDistance), objectDistance * PathConnectionFactor);
}

namespace
{
    bool isInsidePolygon(std::vector<RealVector2D> const& closedPolygon, RealVector2D const& pos)
    {
        auto result = false;
        for (auto const& [from, to] : std::views::zip(closedPolygon, closedPolygon | std::views::drop(1))) {
            if ((from.y > pos.y) != (to.y > pos.y)) {
                auto intersectionX = from.x + (pos.y - from.y) / (to.y - from.y) * (to.x - from.x);
                if (pos.x < intersectionX) {
                    result = !result;
                }
            }
        }
        return result;
    }

    std::vector<RealVector2D> distributeHexagonallyInPolygon(std::vector<RealVector2D> const& polygon, float distance)
    {
        std::vector<RealVector2D> result;
        if (polygon.size() < 3 || distance < NEAR_ZERO) {
            return result;
        }
        auto closedPolygon = polygon;
        closedPolygon.emplace_back(polygon.front());

        auto minPos = polygon.front();
        auto maxPos = polygon.front();
        for (auto const& point : polygon) {
            minPos.x = std::min(minPos.x, point.x);
            minPos.y = std::min(minPos.y, point.y);
            maxPos.x = std::max(maxPos.x, point.x);
            maxPos.y = std::max(maxPos.y, point.y);
        }

        auto rowDistance = distance * sqrtf(3.0f) / 2;
        auto rowIndex = 0;
        for (auto y = minPos.y; y < maxPos.y + NEAR_ZERO && result.size() < MaxNumObjects; y += rowDistance) {
            auto rowOffset = rowIndex % 2 == 0 ? 0.0f : distance / 2;
            for (auto x = minPos.x + rowOffset; x < maxPos.x + NEAR_ZERO && result.size() < MaxNumObjects; x += distance) {
                if (isInsidePolygon(closedPolygon, {x, y})) {
                    result.emplace_back(RealVector2D{x, y});
                }
            }
            ++rowIndex;
        }
        return result;
    }
}

ContentDesc CreatorService::createPolygon(ObjectProperties const& properties, std::vector<RealVector2D> const& points, float objectDistance) const
{
    return createObjectNetwork(properties, distributeHexagonallyInPolygon(points, objectDistance), objectDistance * SurfaceConnectionFactor);
}

namespace
{
    std::vector<RealVector2D> distributeInCircle(RealVector2D const& center, float radius)
    {
        if (radius <= 1 + NEAR_ZERO) {
            return {center};
        }

        std::vector<RealVector2D> result;
        auto centerRow = toInt(center.y);
        auto radiusRow = toInt(radius);

        auto startYRow = centerRow - radiusRow;
        auto radiusRounded = toFloat(radiusRow);
        for (float dx = -radiusRounded; dx <= radiusRounded + NEAR_ZERO; dx += 1.0f) {
            int row = 0;
            for (float dy = -radiusRounded; dy <= radiusRounded + NEAR_ZERO; dy += 1.0f, ++row) {
                float evenRowIncrement = (startYRow + row) % 2 == 0 ? 0.5f : 0.0f;
                auto dxMod = dx + evenRowIncrement;
                if (dxMod * dxMod + dy * dy > radiusRounded * radiusRounded + NEAR_ZERO) {
                    continue;
                }
                result.emplace_back(RealVector2D{center.x + dxMod, center.y + dy});
            }
        }
        return result;
    }
}

ContentDesc CreatorService::createPencilDot(ObjectProperties const& properties, RealVector2D const& pos, float pencilRadius) const
{
    auto alignedPos = pos;
    if (pencilRadius > 1 + NEAR_ZERO) {
        alignedPos.x = toFloat(toInt(pos.x));
        alignedPos.y = toFloat(toInt(pos.y));
    }
    ContentDesc result;
    auto const objectType = getObjectTypeDesc(properties);
    for (auto const& objectPos : distributeInCircle(alignedPos, pencilRadius)) {
        result._objects.emplace_back(ObjectDesc()
                                         .pos(objectPos)
                                         .stiffness(properties._stiffness)
                                         .color(properties._color)
                                         .isStatic(properties._isStatic)
                                         .sticky(properties._sticky)
                                         .type(objectType));
    }
    return properties._material == CreationMaterial_EnergyParticle ? convertToEnergyParticles(properties, result) : result;
}

ContentDesc CreatorService::createFreehandStroke(ObjectProperties const& properties, std::vector<RealVector2D> const& points, float pencilRadius) const
{
    ContentDesc result;
    if (points.empty()) {
        return result;
    }

    DescEditService::Occupancy occupancy;
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    auto addDot = [&](RealVector2D const& pos) {
        DescEditService::get().addIfSpaceAvailable(result, occupancy, createPencilDot(properties, pos, pencilRadius), OverlapDistance, worldSize);
    };

    addDot(points.front());
    for (auto const& [from, to] : std::views::zip(points, points | std::views::drop(1))) {
        auto length = Math::length(to - from);
        for (auto delta = 1.0f; delta < length; delta += 1.0f) {
            addDot(from + (to - from) * delta / length);
        }
        addDot(to);
    }

    if (properties._material != CreationMaterial_EnergyParticle && properties._material != CreationMaterial_Fluid) {
        DescEditService::get().reconnectObjects(result, PencilConnectionDistance);
    }
    return result;
}

namespace
{
    int getMatchedColor(FloatColorRGB const& color, ColorVector<FloatColorRGB> const& customizationColors)
    {
        using Color = std::array<float, 3>;
        auto toHsv = [](FloatColorRGB const& color) {
            float h, s, v;
            ObjectColoring::rgbToHsv(color.r, color.g, color.b, h, s, v);
            return Color{h, s, v};
        };
        std::vector<Color> objectColors;
        for (auto const& customizationColor : customizationColors.values) {
            objectColors.emplace_back(toHsv(customizationColor));
        }

        std::optional<int> bestMatchIndex;
        std::optional<float> bestMatchDistance;
        auto colorHsv = toHsv(color);
        for (auto const& [index, objectColor] : objectColors | boost::adaptors::indexed(0)) {
            auto distance = colorHsv[0] - objectColor[0];
            if (distance > 0.5f) {
                distance -= 1.0f;
            }
            if (distance < -0.5f) {
                distance += 1.0f;
            }
            distance = std::abs(distance) * colorHsv[1] + std::abs(colorHsv[1] - objectColor[1]);
            if (!bestMatchDistance || *bestMatchDistance > distance) {
                bestMatchIndex = toInt(index);
                bestMatchDistance = distance;
            }
        }
        return *bestMatchIndex;
    }
}

ContentDesc CreatorService::createPatternFromImage(RgbImage const& image, RealVector2D const& center) const
{
    auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;
    ContentDesc result;
    for (auto x : std::views::iota(0, image.width)) {
        for (auto y : std::views::iota(0, image.height)) {
            auto address = (x + y * image.width) * 3;
            int r = image.pixels.at(address);
            int g = image.pixels.at(address + 1);
            int b = image.pixels.at(address + 2);
            auto xOffset = y % 2 == 0 ? 0.0f : 0.5f;
            if (r > 20 || g > 20 || b > 20) {
                auto color = FloatColorRGB{toFloat(r) / 255.0f, toFloat(g) / 255.0f, toFloat(b) / 255.0f};
                result._objects.emplace_back(ObjectDesc()
                                                 .id(NumberGenerator::get().createEntityId())
                                                 .pos({toFloat(x) + xOffset, toFloat(y)})
                                                 .color(getMatchedColor(color, customizationColors))
                                                 .isStatic(false)
                                                 .type(SolidDesc()));
            }
        }
    }

    DescEditService::get().reconnectObjects(result, ImageConnectionDistance);
    DescEditService::get().setCenter(result, center);
    return result;
}

namespace
{
    RealVector2D evaluateBezier(std::vector<RealVector2D> const& controlPoints, float t)
    {
        auto points = controlPoints;
        for (auto count = points.size(); count > 1; --count) {
            auto range = std::span(points).first(count);
            for (auto&& [current, next] : std::views::zip(range, range | std::views::drop(1))) {
                current = current * (1.0f - t) + next * t;
            }
        }
        return points.front();
    }
}

std::vector<RealVector2D> CreatorService::calcBezierCurvePath(std::vector<RealVector2D> const& controlPoints, float objectDistance) const
{
    if (controlPoints.size() < 2) {
        return controlPoints;
    }

    auto controlPolygonLength = 0.0f;
    for (auto const& [from, to] : std::views::zip(controlPoints, controlPoints | std::views::drop(1))) {
        controlPolygonLength += Math::length(to - from);
    }

    auto numSegments = std::clamp(toInt(controlPolygonLength * 4 / objectDistance), 16, 10000);
    std::vector<RealVector2D> result;
    result.reserve(numSegments + 1);
    for (auto segment : std::views::iota(0, numSegments + 1)) {
        result.emplace_back(evaluateBezier(controlPoints, toFloat(segment) / toFloat(numSegments)));
    }
    return result;
}

ContentDesc CreatorService::createObjectNetwork(ObjectProperties const& properties, std::vector<RealVector2D> const& positions, float connectionDistance) const
{
    ContentDesc result;
    auto const objectType = getObjectTypeDesc(properties);
    for (auto const& pos : positions) {
        result._objects.emplace_back(ObjectDesc()
                                         .id(NumberGenerator::get().createEntityId())
                                         .stiffness(properties._stiffness)
                                         .sticky(properties._sticky)
                                         .pos(pos)
                                         .color(properties._color)
                                         .isStatic(properties._isStatic)
                                         .type(objectType));
    }

    if (properties._material == CreationMaterial_EnergyParticle) {
        result = convertToEnergyParticles(properties, result);
    } else {
        DescEditService::get().reconnectObjects(result, connectionDistance);
    }
    return result;
}

ContentDesc CreatorService::convertToEnergyParticles(ObjectProperties const& properties, ContentDesc const& content) const
{
    ContentDesc result;
    for (auto const& object : content._objects) {
        result._energies.emplace_back(EnergyDesc().pos(object._pos).energy(properties._energy).color(properties._color));
    }
    return result;
}

ObjectTypeDesc CreatorService::getObjectTypeDesc(ObjectProperties const& properties) const
{
    switch (properties._material) {
    case CreationMaterial_Fluid:
        return FluidDesc().energy(properties._energy).glow(properties._glow);
    case CreationMaterial_FreeCell:
        return FreeCellDesc().energy(properties._energy);
    default:
        return SolidDesc().energy(properties._energy);
    }
}
