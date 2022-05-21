#include "RemoteSimulationDataParser.h"

std::vector<RemoteSimulationData> RemoteSimulationDataParser::decode(boost::property_tree::ptree tree)
{
    std::vector<RemoteSimulationData> result;

    for (auto const& [key, subTree] : tree) {
    }
    
    return result;
}
