#include "McpNetworkTools.h"

#include <algorithm>
#include <filesystem>
#include <ranges>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/Interface/StringHelper.h>
#include <Base/Interface/VersionParserService.h>

#include <Network/Interface/McpArguments.h>
#include <Network/Interface/McpJson.h>
#include <Network/Interface/McpSchema.h>
#include <Network/Interface/NetworkResourceRawTO.h>
#include <Network/Interface/NetworkService.h>

#include <Persister/Interface/PersisterFacade.h>
#include <Persister/Interface/SerializerService.h>

namespace
{
    auto constexpr DefaultMaxResults = 50;
    auto constexpr MaxResults = 500;
    auto constexpr MaxDescriptionLength = 200;
    auto constexpr GenomeFileExtension = ".genome";

    std::string getResourceTypeName(NetworkResourceType type)
    {
        return type == NetworkResourceType_Genome ? "genome" : "simulation";
    }

    std::string getWorkspaceName(WorkspaceType type)
    {
        switch (type) {
        case WorkspaceType_Public:
            return "community";
        case WorkspaceType_AlienProject:
            return "alien_project";
        default:
            return "private";
        }
    }
}

std::vector<McpTool> McpNetworkTools::getTools(McpToolContext& context)
{
    _context = &context;

    return {
        McpTool{
            .name = "list_network_resources",
            .description = "Fetches the simulations and genomes available on the ALIEN server: shared by the community, provided by the ALIEN project "
                           "and, if the user is logged in, stored in the private workspace of the user. The newest resources come first. Call it "
                           "before download_network_resource.",
            .inputSchema = McpSchema::object({
                {"resource_type", McpSchema::enumeration("Only list resources of this type", {"simulation", "genome"})},
                {"workspace", McpSchema::enumeration("Only list resources of this workspace", {"community", "alien_project", "private"})},
                {"filter", McpSchema::string("Only list resources whose name, user name, description or timestamp contains this text")},
                {"max_results", McpSchema::integer(std::format("Maximum number of resources to return, default: {}", DefaultMaxResults), 1, MaxResults)},
            }),
            .deferredHandler = [this](boost::json::object const& arguments, McpToolCompletion const& completion) { listResources(arguments, completion); },
        },
        McpTool{
            .name = "download_network_resource",
            .description = "Downloads a simulation or genome from the ALIEN server. A simulation replaces the current simulation in ALIEN. A genome is "
                           "saved to a genome file (*.genome), from which create_seed can create creatures.",
            .inputSchema = McpSchema::object(
                {
                    {"resource_id", McpSchema::string("Id of the resource as returned by list_network_resources")},
                    {"genome_file_path", McpSchema::string("For genomes: absolute path of the genome file to write, must end with .genome")},
                    {"overwrite", McpSchema::boolean("For genomes: overwrite an existing file. Default: false")},
                },
                {"resource_id"}),
            .deferredHandler = [this](boost::json::object const& arguments, McpToolCompletion const& completion) { downloadResource(arguments, completion); },
        },
        McpTool{
            .name = "upload_simulation",
            .description = "Uploads the current simulation to the private workspace of the logged-in user on the ALIEN server. The simulation is not "
                           "shared with the community. The user has to be logged in via ALIEN.",
            .inputSchema = McpSchema::object(
                {
                    {"name", McpSchema::string("Name of the simulation")},
                    {"description", McpSchema::string("Description of the simulation")},
                    {"folder", McpSchema::string("Folder, subfolders are separated by '/'. Default: no folder")},
                },
                {"name"}),
            .deferredHandler = [this](boost::json::object const& arguments, McpToolCompletion const& completion) { uploadSimulation(arguments, completion); },
        },
    };
}

void McpNetworkTools::process()
{
    _listTask.process();
    _downloadTask.process();
    _uploadTask.process();
}

void McpNetworkTools::listResources(boost::json::object const& arguments, McpToolCompletion const& completion)
{
    createResourceList(arguments);
    _listTask.execute(
        [](SenderId const& senderId) {
            return _PersisterFacade::get()->scheduleGetNetworkResources(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, GetNetworkResourcesRequestData());
        },
        [this, arguments](PersisterRequestId const& requestId) {
            _resources = _PersisterFacade::get()->fetchGetNetworkResourcesData(requestId).resourceTOs;
            std::ranges::sort(_resources, std::ranges::greater(), [](NetworkResourceRawTO const& resource) { return resource->timestamp; });
            return createResourceList(arguments);
        },
        completion);
}

McpToolResult McpNetworkTools::createResourceList(boost::json::object const& arguments) const
{
    auto resourceType = McpArguments::getOptionalString(arguments, "resource_type");
    auto workspace = McpArguments::getOptionalString(arguments, "workspace");
    auto filter = McpArguments::getOptionalString(arguments, "filter");
    auto maxResults = McpArguments::getOptionalInt(arguments, "max_results", 1, MaxResults).value_or(DefaultMaxResults);

    auto matches = [&](NetworkResourceRawTO const& resource) {
        return (!resourceType || getResourceTypeName(resource->resourceType) == *resourceType)
            && (!workspace || getWorkspaceName(resource->workspaceType) == *workspace)
            && (!filter || resource->matchWithFilter(*filter) || StringHelper::containsCaseInsensitive(resource->description, *filter));
    };

    boost::json::array resources;
    auto numMatches = 0;
    for (auto const& resource : _resources | std::views::filter(matches)) {
        ++numMatches;
        if (toInt(resources.size()) >= maxResults) {
            continue;
        }
        auto description = resource->description;
        if (description.size() > MaxDescriptionLength) {
            description.resize(MaxDescriptionLength);
            description.append(" ...");
        }
        boost::json::object entry{
            {"id", resource->id},
            {"type", getResourceTypeName(resource->resourceType)},
            {"name", resource->resourceName},
            {"description", description},
            {"user", resource->userName},
            {"workspace", getWorkspaceName(resource->workspaceType)},
            {"timestamp", resource->timestamp},
            {"version", resource->version},
            {"downloads", resource->numDownloads},
            {"likes", resource->getTotalLikes()},
        };
        if (resource->resourceType == NetworkResourceType_Simulation) {
            entry["world_width"] = resource->width;
            entry["world_height"] = resource->height;
            entry["objects"] = resource->particles;
        } else {
            entry["nodes"] = resource->particles;
        }
        resources.emplace_back(std::move(entry));
    }
    return {.text = McpJson::serialize(boost::json::object{{"matching_resources", numMatches}, {"resources", std::move(resources)}})};
}

void McpNetworkTools::downloadResource(boost::json::object const& arguments, McpToolCompletion const& completion)
{
    auto resourceId = McpArguments::getString(arguments, "resource_id");
    auto findResult = std::ranges::find_if(_resources, [&](NetworkResourceRawTO const& resource) { return resource->id == resourceId; });
    if (findResult == _resources.end()) {
        throw std::invalid_argument(std::format("The resource '{}' is unknown. Call list_network_resources first.", resourceId));
    }
    auto const& resource = *findResult;

    std::optional<std::filesystem::path> genomePath;
    std::string genomeFilePath;
    if (resource->resourceType == NetworkResourceType_Genome) {
        genomeFilePath = McpArguments::getString(arguments, "genome_file_path");
        if (!genomeFilePath.ends_with(GenomeFileExtension)) {
            throw std::invalid_argument(std::format("The genome file name must end with '{}'.", GenomeFileExtension));
        }
        genomePath = McpArguments::getFilePath(arguments, "genome_file_path");
        if (std::filesystem::exists(*genomePath) && !McpArguments::getOptionalBool(arguments, "overwrite").value_or(false)) {
            throw std::invalid_argument(std::format("The file '{}' already exists. Set 'overwrite' to replace it.", genomeFilePath));
        }
    }

    auto requestData = DownloadNetworkResourceRequestData{
        .resourceId = resource->id,
        .resourceName = resource->resourceName,
        .resourceVersion = resource->version,
        .resourceType = resource->resourceType,
        .downloadCache = getDownloadCache(),
    };
    _downloadTask.execute(
        [requestData](SenderId const& senderId) {
            return _PersisterFacade::get()->scheduleDownloadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [this, genomePath, genomeFilePath](PersisterRequestId const& requestId) {
            auto data = _PersisterFacade::get()->fetchDownloadNetworkResourcesData(requestId);
            auto versionWarning = VersionParserService::get().isVersionNewer(data.resourceVersion)
                ? std::format(" Warning: it was created with the newer ALIEN version {} and might not work as expected.", data.resourceVersion)
                : std::string();

            if (data.resourceType == NetworkResourceType_Simulation) {
                auto const& simulation = std::get<SimulationDesc>(data.resourceData);
                _context->applySimulation(simulation);
                _context->onSelectionChanged();
                _context->showMessage(data.resourceName);
                return McpToolResult{
                    .text = std::format(
                        "Downloaded and loaded the simulation '{}' with a world size of {} x {}. The simulation is paused.{}",
                        data.resourceName,
                        simulation._worldSize.x,
                        simulation._worldSize.y,
                        versionWarning)};
            }
            if (!SerializerService::get().serializeGenomeToFile(*genomePath, std::get<GenomeDesc>(data.resourceData))) {
                throw std::runtime_error(std::format("The genome could not be saved to '{}'.", genomeFilePath));
            }
            _context->showMessage("Genome downloaded");
            return McpToolResult{.text = std::format("Downloaded the genome '{}' and saved it to '{}'.{}", data.resourceName, genomeFilePath, versionWarning)};
        },
        completion);
    _context->showMessage("Downloading ...");
}

void McpNetworkTools::uploadSimulation(boost::json::object const& arguments, McpToolCompletion const& completion)
{
    auto name = McpArguments::getString(arguments, "name");
    if (name.empty()) {
        throw std::invalid_argument("'name' must not be empty.");
    }
    auto description = McpArguments::getOptionalString(arguments, "description").value_or("");
    auto folder = McpArguments::getOptionalString(arguments, "folder").value_or("");
    if (!NetworkService::get().isLoggedIn()) {
        throw std::runtime_error("The user is not logged in. Uploading requires the user to log in via ALIEN.");
    }
    auto previewJpg = _context->createSimulationPreviewJpg();
    if (!previewJpg) {
        throw std::runtime_error("The preview picture of the simulation could not be created.");
    }

    auto requestData = UploadNetworkResourceRequestData{
        .folderName = folder,
        .resourceWithoutFolderName = name,
        .resourceDescription = description,
        .workspaceType = WorkspaceType_Private,
        .downloadCache = getDownloadCache(),
        .data =
            UploadNetworkResourceRequestData::SimulationData{.zoom = _context->getZoomFactor(), .center = _context->getVisibleAreaCenter(), .jpg = *previewJpg},
    };
    _uploadTask.execute(
        [requestData](SenderId const& senderId) {
            return _PersisterFacade::get()->scheduleUploadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [this, name](PersisterRequestId const& requestId) {
            _PersisterFacade::get()->fetchUploadNetworkResourcesData(requestId);
            _context->onNetworkResourcesChanged();
            _context->showMessage("Simulation uploaded");
            return McpToolResult{.text = std::format("Uploaded the simulation '{}' to the private workspace.", name)};
        },
        completion);
    _context->showMessage("Uploading ...");
}

DownloadCache McpNetworkTools::getDownloadCache()
{
    if (!_downloadCache) {
        _downloadCache = std::make_shared<_DownloadCache>();
    }
    return _downloadCache;
}
