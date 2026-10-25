#include "RayTracingLayout.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <ranges>
#include <utility>

#include <Base/Math.h>

#include <Data/SimulationParameters.h>

namespace
{
    // The gas grid covers the view and a margin for the spread of the particles and the range of the lights
    auto constexpr MinGasNodeDistance = 0.25f;
    auto constexpr MaxGasBlurRadius = 32;

    // Standard deviation of the spread of the particles, like the soft disks of radius 7.5 in the rasterized view. Zoomed out, the spread
    // covers a minimum number of pixels.
    auto constexpr CloudSpread = 2.6f;
    auto constexpr MinCloudSpreadInPixels = 2.0f;

    // Zoomed out, the falloff distance of the light of the cells grows continuously, so that it covers a minimum number of nodes of the
    // irradiance grid and moving cells do not flicker. The range of the lights and the size of the clusters, which collect the light, grow
    // with it. The irradiance grid has twice the node distance of the gas grid as long as it resolves the falloff, otherwise the node distance
    // of the gas grid. The light of the radiation sources, whose rays cost far more than the light of the cells, keeps twice the node distance
    // of the gas grid, so that zoomed out it is not traced at four times the number of nodes.
    auto constexpr MinLightFalloffInIrradianceNodes = 1.5f;
    auto constexpr LightRangeInFalloffDistances = 4.0f;
    auto constexpr LightClusterSizeInFalloffDistances = 0.8f;

    // The smooth sampling of the irradiance grid reaches this number of nodes beyond a position
    auto constexpr IrradianceSamplingReach = 2.0f;

    // The shadow grid is coarser than the gas grid, since the shadows of distant objects are soft. Its node distance is a power of two multiple
    // of the minimum.
    auto constexpr MinShadowGridNodeDistance = 1.0f;

    RayTracingLayout::GasGrid calcGasGrid(RenderView const& view)
    {
        auto pixelSize = 1.0f / view.zoomFactor;
        auto cloudSpread = std::max(CloudSpread, MinCloudSpreadInPixels * pixelSize);
        auto const& visibleRect = view.visibleWorldRect;
        auto visibleExtent = std::max(visibleRect.bottomRight.x - visibleRect.topLeft.x, visibleRect.bottomRight.y - visibleRect.topLeft.y);
        if (!std::isfinite(visibleExtent) || !std::isfinite(cloudSpread)) {
            return {};
        }

        // The node distance of the gas grid is less than twice the extent of the view with the margins divided by the number of nodes. The
        // falloff distance covers the minimum number of irradiance nodes at any such node distance, so it does not change in steps with the
        // node distance.
        auto maxNodes = toFloat(RayTracingLayout::MaxGasNodes - 2);
        auto falloffPerNodeDistance = 2.0f * MinLightFalloffInIrradianceNodes;
        auto lightFalloffDistance = std::max(
            {RayTracingLayout::LightFalloffDistance,
             falloffPerNodeDistance * visibleExtent / (maxNodes - 2.0f * LightRangeInFalloffDistances * falloffPerNodeDistance),
             falloffPerNodeDistance * (visibleExtent + 6.0f * cloudSpread) / maxNodes});
        auto margin = std::max(3.0f * cloudSpread, LightRangeInFalloffDistances * lightFalloffDistance);
        auto nodeDistance = MinGasNodeDistance;
        while ((visibleExtent + 2.0f * margin) / nodeDistance > maxNodes) {
            nodeDistance *= 2.0f;
        }
        auto irradianceNodeDistance = MinLightFalloffInIrradianceNodes * 2.0f * nodeDistance <= lightFalloffDistance ? 2.0f * nodeDistance : nodeDistance;

        auto topLeft = visibleRect.topLeft - RealVector2D{margin, margin};
        auto bottomRight = visibleRect.bottomRight + RealVector2D{margin, margin};
        RealVector2D origin{std::floor(topLeft.x / nodeDistance) * nodeDistance, std::floor(topLeft.y / nodeDistance) * nodeDistance};
        IntVector2D size{
            std::min(toInt(std::ceil((bottomRight.x - origin.x) / nodeDistance)) + 1, RayTracingLayout::MaxGasNodes),
            std::min(toInt(std::ceil((bottomRight.y - origin.y) / nodeDistance)) + 1, RayTracingLayout::MaxGasNodes)};

        // The light clusters are aligned to the world, so that moving the view does not change how the light of the cells is collected
        auto lightClusterSize = LightClusterSizeInFalloffDistances * lightFalloffDistance;
        RealVector2D lightClusterOrigin{std::floor(origin.x / lightClusterSize) * lightClusterSize, std::floor(origin.y / lightClusterSize) * lightClusterSize};
        auto getNumLightClusters = [&](float gridOrigin, int numNodes, float clusterOrigin) {
            auto gridEnd = gridOrigin + toFloat(numNodes - 1) * nodeDistance;
            return std::min(toInt(std::ceil((gridEnd - clusterOrigin) / lightClusterSize)) + 1, RayTracingLayout::MaxLightClusters);
        };
        auto sourceLightNodeDistance = 2.0f * nodeDistance;
        auto getNumGridNodes = [&](int numNodes, float gridNodeDistance) {
            return std::min(toInt(std::ceil(toFloat(numNodes - 1) * nodeDistance / gridNodeDistance)) + 1, RayTracingLayout::GasIrradianceImageSize);
        };
        return RayTracingLayout::GasGrid{
            .origin = origin,
            .nodeDistance = nodeDistance,
            .size = size,
            .blurRadius = std::min(toInt(std::ceil(3.0f * cloudSpread / nodeDistance)), MaxGasBlurRadius),
            .cloudSpreadInNodes = cloudSpread / nodeDistance,
            .lightFalloffDistance = lightFalloffDistance,
            .lightRange = LightRangeInFalloffDistances * lightFalloffDistance,
            .lightClusterOrigin = lightClusterOrigin,
            .lightClusterSize = lightClusterSize,
            .numLightClusters = {getNumLightClusters(origin.x, size.x, lightClusterOrigin.x), getNumLightClusters(origin.y, size.y, lightClusterOrigin.y)},
            .irradianceNodeDistance = irradianceNodeDistance,
            .irradianceSize = {getNumGridNodes(size.x, irradianceNodeDistance), getNumGridNodes(size.y, irradianceNodeDistance)},
            .sourceLightNodeDistance = sourceLightNodeDistance,
            .sourceLightSize = {getNumGridNodes(size.x, sourceLightNodeDistance), getNumGridNodes(size.y, sourceLightNodeDistance)},
        };
    }

    // Without borderless rendering, only the world is lit
    RealRect getLitRegion(RayTracingLayout::GasGrid const& gasGrid, SimulationParameters const& parameters, IntVector2D const& worldSize)
    {
        auto gasGridExtent = RealVector2D{toFloat(gasGrid.size.x - 1), toFloat(gasGrid.size.y - 1)} * gasGrid.nodeDistance;
        RealRect result{gasGrid.origin, gasGrid.origin + gasGridExtent};
        if (!parameters.borderlessRendering.value) {
            result.topLeft = {std::max(result.topLeft.x, 0.0f), std::max(result.topLeft.y, 0.0f)};
            result.bottomRight = {std::min(result.bottomRight.x, toFloat(worldSize.x)), std::min(result.bottomRight.y, toFloat(worldSize.y))};
        }
        return result;
    }

    // A radiation source counts if the light of one of its periodic copies reaches the region. Sources with the same area and range shine as
    // one, so that their light is traced only once.
    std::vector<RayTracingLayout::SourceLight> getSourceLights(SimulationParameters const& parameters, IntVector2D const& worldSize, RealRect const& region)
    {
        auto world = toRealVector2D(worldSize);
        std::vector<RayTracingLayout::SourceLight> result;
        for (auto index : std::views::iota(0, parameters.numSources)) {
            auto brightness = parameters.sourceLightBrightness.sourceValues[index];
            auto range = parameters.sourceLightRange.sourceValues[index];
            if (brightness <= 0.0f || range <= 0.0f) {
                continue;
            }
            auto circular = parameters.sourceShapeType.sourceValues[index] == SourceShapeType_Circular;
            auto radius = parameters.sourceCircularRadius.sourceValues[index];
            auto halfExtent = circular ? RealVector2D{radius, radius} : parameters.sourceRectangularRect.sourceValues[index] / 2.0f;
            auto position = parameters.sourcePosition.sourceValues[index];
            auto reach = halfExtent + RealVector2D{range, range};
            auto reachesRegion = [&](float regionStart, float regionEnd, float center, float reach, float period) {
                auto offset = Math::modulo(center - regionStart, period);
                return offset - reach <= regionEnd - regionStart || offset + reach >= period;
            };
            if (!reachesRegion(region.topLeft.x, region.bottomRight.x, position.x, reach.x, world.x)
                || !reachesRegion(region.topLeft.y, region.bottomRight.y, position.y, reach.y, world.y)) {
                continue;
            }
            auto const& color = parameters.sourceLightColor.sourceValues[index];
            RayTracingLayout::SourceLight light{
                .center = position,
                .halfExtent = halfExtent,
                .range = range,
                .circular = circular,
                .color = {color.r * brightness, color.g * brightness, color.b * brightness},
            };
            auto sameArea = std::ranges::find_if(result, [&](RayTracingLayout::SourceLight const& other) {
                return other.center == light.center && other.halfExtent == light.halfExtent && other.range == light.range && other.circular == light.circular;
            });
            if (sameArea == result.end()) {
                result.emplace_back(light);
            } else {
                sameArea->color.r += light.color.r;
                sameArea->color.g += light.color.g;
                sameArea->color.b += light.color.b;
            }
        }
        return result;
    }

    RealRect getEnlargedRect(RealRect const& rect, float margin)
    {
        return {rect.topLeft - RealVector2D{margin, margin}, rect.bottomRight + RealVector2D{margin, margin}};
    }

    // Part between the bounds that the periodic copies of an interval around the center cover along an axis
    std::optional<std::pair<float, float>> getCoveredPart(float center, float halfWidth, float lowerBound, float upperBound, float period)
    {
        auto firstCopy = std::ceil((lowerBound - halfWidth - center) / period);
        auto lastCopy = std::floor((upperBound + halfWidth - center) / period);
        if (firstCopy > lastCopy) {
            return std::nullopt;
        }
        return std::pair{std::max(center + firstCopy * period - halfWidth, lowerBound), std::min(center + lastCopy * period + halfWidth, upperBound)};
    }

    // The rays to the radiation sources start in the given region and end on the parts of the areas of the sources within their reach
    ShadowGridLayout calcShadowGrid(RealRect const& rayStarts, std::vector<RayTracingLayout::SourceLight> const& lights, IntVector2D const& worldSize)
    {
        if (lights.empty()) {
            return {};
        }
        auto region = rayStarts;
        auto maxRayLengthPerRange = std::sqrt(1.0f + RayTracingLayout::SourceLightSpread * RayTracingLayout::SourceLightSpread);
        for (auto const& light : lights) {
            auto reachedRegion = getEnlargedRect(rayStarts, maxRayLengthPerRange * light.range);
            auto shiningPartX = getCoveredPart(light.center.x, light.halfExtent.x, reachedRegion.topLeft.x, reachedRegion.bottomRight.x, toFloat(worldSize.x));
            auto shiningPartY = getCoveredPart(light.center.y, light.halfExtent.y, reachedRegion.topLeft.y, reachedRegion.bottomRight.y, toFloat(worldSize.y));
            if (shiningPartX && shiningPartY) {
                region.topLeft = {std::min(region.topLeft.x, shiningPartX->first), std::min(region.topLeft.y, shiningPartY->first)};
                region.bottomRight = {std::max(region.bottomRight.x, shiningPartX->second), std::max(region.bottomRight.y, shiningPartY->second)};
            }
        }

        auto extent = std::max(region.bottomRight.x - region.topLeft.x, region.bottomRight.y - region.topLeft.y);
        if (!std::isfinite(extent)) {
            return {};
        }
        auto nodeDistance = MinShadowGridNodeDistance;
        while (extent / nodeDistance + 3.0f > toFloat(RayTracingLayout::MaxShadowGridNodes)) {
            nodeDistance *= 2.0f;
        }
        RealVector2D origin{std::floor(region.topLeft.x / nodeDistance) * nodeDistance, std::floor(region.topLeft.y / nodeDistance) * nodeDistance};
        return ShadowGridLayout{
            .origin = origin,
            .nodeDistance = nodeDistance,
            .size =
                {std::min(toInt(std::ceil((region.bottomRight.x - origin.x) / nodeDistance)) + 1, RayTracingLayout::MaxShadowGridNodes),
                 std::min(toInt(std::ceil((region.bottomRight.y - origin.y) / nodeDistance)) + 1, RayTracingLayout::MaxShadowGridNodes)},
        };
    }
}

RayTracingLayout RayTracingLayout::calc(RenderView const& view, SimulationParameters const& parameters, IntVector2D const& worldSize)
{
    RayTracingLayout result;
    result.gasGrid = calcGasGrid(view);
    if (result.isEmpty()) {
        return result;
    }
    auto litRegion = getLitRegion(result.gasGrid, parameters, worldSize);
    result.sourceLights = getSourceLights(parameters, worldSize, litRegion);
    auto rayStarts = getEnlargedRect(litRegion, IrradianceSamplingReach * result.gasGrid.sourceLightNodeDistance);
    result.shadowGrid = calcShadowGrid(rayStarts, result.sourceLights, worldSize);
    return result;
}

bool RayTracingLayout::isEmpty() const
{
    return gasGrid.size.x <= 0 || gasGrid.size.y <= 0;
}
