#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Data/GenomeDesc.h>

#include <Network/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

class McpGenomeTools
{
    MAKE_SINGLETON(McpGenomeTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

    GenomeDesc getGenomeFromArguments(boost::json::object const& arguments) const;
    GenomeDesc getGenomeOfObject(uint64_t objectId) const;

private:
    McpToolResult getGenome(boost::json::object const& arguments) const;
    McpToolResult saveGenome(boost::json::object const& arguments) const;
    McpToolResult createSeed(boost::json::object const& arguments) const;
    McpToolResult injectGenome(boost::json::object const& arguments) const;

    McpToolContext* _context = nullptr;
};
