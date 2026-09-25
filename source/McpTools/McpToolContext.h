#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <Base/Definitions.h>

#include <EngineInterface/CreatorService.h>

// Functionality of the application in which the MCP server runs
class McpToolContext
{
public:
    virtual ~McpToolContext() = default;

    virtual RealVector2D getVisibleAreaCenter() const = 0;
    virtual RealVector2D getVisibleAreaSize() const = 0;

    virtual void createSimulation(std::string const& projectName, IntVector2D const& worldSize) = 0;
    virtual void onSelectionChanged() = 0;
    virtual std::optional<RgbImage> loadImage(std::filesystem::path const& path) const = 0;

    virtual void showMessage(std::string const& message) = 0;
};
