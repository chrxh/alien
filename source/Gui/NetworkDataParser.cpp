#include "NetworkDataParser.h"

std::vector<RemoteSimulationData> NetworkDataParser::decodeRemoteSimulationData(boost::property_tree::ptree tree)
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
        entry.likes = subTree.get<int>("likes");
        entry.numDownloads = subTree.get<int>("numDownloads");
        entry.fromRelease = subTree.get<int>("fromRelease") == 1;
        result.emplace_back(entry);
    }
    return result;
}

std::vector<UserData> NetworkDataParser::decodeUserData(boost::property_tree::ptree tree)
{
    std::vector<UserData> result;
    for (auto const& [key, subTree] : tree) {
        UserData entry;
        entry.userName = subTree.get<std::string>("userName");
        entry.starsEarned = subTree.get<int>("starsEarned");
        entry.starsGiven = subTree.get<int>("starsGiven");
        entry.timestamp = subTree.get<std::string>("timestamp");
        result.emplace_back(entry);
    }
    return result;
}
