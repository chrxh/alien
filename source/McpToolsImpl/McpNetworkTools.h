#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Network/Definitions.h>
#include <Network/McpServer.h>
#include <McpToolsInterface/McpToolContext.h>

#include <PersisterInterface/DownloadCache.h>

#include "McpPersisterTask.h"

class McpNetworkTools
{
    MAKE_SINGLETON(McpNetworkTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);
    void process();

private:
    void listResources(boost::json::object const& arguments, McpToolCompletion const& completion);
    void downloadResource(boost::json::object const& arguments, McpToolCompletion const& completion);
    void uploadSimulation(boost::json::object const& arguments, McpToolCompletion const& completion);

    McpToolResult createResourceList(boost::json::object const& arguments) const;
    DownloadCache getDownloadCache();

    McpToolContext* _context = nullptr;
    McpPersisterTask _listTask;
    McpPersisterTask _downloadTask;
    McpPersisterTask _uploadTask;
    std::vector<NetworkResourceRawTO> _resources;
    DownloadCache _downloadCache;
};
