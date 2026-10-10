#pragma once

#include <vector>

#include <Base/Interface/Singleton.h>

#include <Data/Interface/GenomeDesc.h>

#include <Network/Interface/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

class McpGenomeTools
{
    MAKE_SINGLETON(McpGenomeTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult getGenome(boost::json::object const& arguments) const;
    McpToolResult saveGenome(boost::json::object const& arguments) const;
    McpToolResult createSeed(boost::json::object const& arguments) const;
    McpToolResult injectGenome(boost::json::object const& arguments) const;

    GenomeDesc getGenomeFromArguments(boost::json::object const& arguments) const;
    GenomeDesc getGenomeOfObject(uint64_t objectId) const;

    McpToolContext* _context = nullptr;
};
