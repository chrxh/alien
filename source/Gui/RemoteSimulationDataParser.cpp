#include "RemoteSimulationDataParser.h"

std::vector<RemoteSimulationData> RemoteSimulationDataParser::decode(boost::property_tree::ptree tree)
{
    std::vector<RemoteSimulationData> result;
    for (auto const& [key, subTree] : tree) {
        RemoteSimulationData entry;
        entry.id = subTree.get<std::string>("id");
        entry.userName = subTree.get<std::string>("userName");
        entry.simName = subTree.get<std::string>("simulationName");
        entry.description= subTree.get<std::string>("description");
        entry.width = subTree.get<int>("width");
        entry.height = subTree.get<int>("height");
        entry.particles = subTree.get<int>("particles");
        entry.version = subTree.get<std::string>("version");
        entry.timestamp = subTree.get<std::string>("timestamp");
        entry.contentSize = std::stoll(subTree.get<std::string>("contentSize"));
        result.emplace_back(entry);
    }
    return result;
}
