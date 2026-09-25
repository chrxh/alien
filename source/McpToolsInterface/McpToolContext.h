#pragma once

#include <string>

#include <Base/Definitions.h>

class McpToolContext
{
public:
    virtual ~McpToolContext() = default;

    virtual RealVector2D getVisibleAreaCenter() const = 0;
    virtual RealVector2D getVisibleAreaSize() const = 0;

    virtual void createSimulation(std::string const& projectName, IntVector2D const& worldSize) = 0;
    virtual void onSelectionChanged() = 0;

    virtual void showMessage(std::string const& message) = 0;
};
