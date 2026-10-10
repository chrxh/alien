#pragma once

#include <optional>
#include <string>

#include <Base/Interface/Definitions.h>

struct SimulationDesc;

enum class McpPictureFormat
{
    Jpg,
    Png
};

class McpToolContext
{
public:
    virtual ~McpToolContext() = default;

    virtual RealVector2D getVisibleAreaCenter() const = 0;
    virtual RealVector2D getVisibleAreaSize() const = 0;
    virtual float getZoomFactor() const = 0;
    virtual void setVisibleArea(RealVector2D const& center, float zoomFactor) = 0;

    virtual void createSimulation(std::string const& projectName, IntVector2D const& worldSize) = 0;
    virtual void applySimulation(SimulationDesc const& simulation) = 0;
    virtual void onSelectionChanged() = 0;
    virtual void onNetworkResourcesChanged() = 0;

    virtual std::string createPicture(IntVector2D const& resolution, McpPictureFormat format) = 0;
    virtual std::optional<std::string> createSimulationPreviewJpg() = 0;

    virtual void showMessage(std::string const& message) = 0;
};
