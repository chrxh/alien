#pragma once

#include <string>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/CreatorService.h>
#include <EngineInterface/Descs.h>

#include <Network/McpServer.h>
#include <McpToolsInterface/McpToolContext.h>

class McpCreatorTools
{
    MAKE_SINGLETON(McpCreatorTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult createObject(boost::json::object const& arguments) const;
    McpToolResult createRectangle(boost::json::object const& arguments) const;
    McpToolResult createHexagon(boost::json::object const& arguments) const;
    McpToolResult createDisc(boost::json::object const& arguments) const;
    McpToolResult drawFreehand(boost::json::object const& arguments) const;
    McpToolResult createLine(boost::json::object const& arguments) const;
    McpToolResult createCurve(boost::json::object const& arguments) const;
    McpToolResult createPolygon(boost::json::object const& arguments) const;
    McpToolResult createPatternFromImage(boost::json::object const& arguments) const;

    CreatorService::ObjectProperties getObjectProperties(boost::json::object const& arguments) const;
    RealVector2D getCenter(boost::json::object const& arguments) const;
    std::vector<RealVector2D> getPointsInsideWorld(boost::json::object const& arguments, std::string const& key, size_t minNumPoints) const;
    void checkInsideWorld(RealVector2D const& pos) const;
    void checkNumObjects(float estimatedNumObjects) const;

    McpToolResult addToSimulation(ContentDesc&& content, CreatorService::ObjectProperties const& properties, std::string const& shape) const;

    McpToolContext* _context = nullptr;
};
