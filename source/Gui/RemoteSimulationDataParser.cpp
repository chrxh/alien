#include "RemoteSimulationDataParser.h"

std::vector<RemoteSimulationData> RemoteSimulationDataParser::decode(boost::property_tree::ptree tree)
{
    std::vector<RemoteSimulationData> result;
    for (auto const& [key, subTree] : tree) {
        RemoteSimulationData entry;
        entry.id = subTree.get<std::string>("id");
        entry.simName = subTree.get<std::string>("simulationName");
        entry.userName = subTree.get<std::string>("userName");
        entry.width = subTree.get<int>("width");
        entry.height = subTree.get<int>("height");
        entry.version = subTree.get<std::string>("version");
        entry.timestamp = subTree.get<std::string>("timestamp");
        result.emplace_back(entry);
    }
    return result;
}
