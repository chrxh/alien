#include "ChangeDescriptions.h"


DataChangeDescription::DataChangeDescription(DataDescription const & dataBefore, DataDescription const & dataAfter)
{
	vector<uint64_t> clusterIdsAfter;
	std::transform(dataAfter.clusters.begin(), dataAfter.clusters.end(), clusterIdsAfter.begin(), [](auto const& cluster) {
		return cluster.id;
	});
	unordered_set<uint64_t> clusterIdSetAfter(clusterIdsAfter.begin(), clusterIdsAfter.end());

	for (auto const& cluster : dataBefore.clusters) {
		if(clusterIdSetAfter.find(cluster.id) == clusterIdSetAfter.end()) {
		}
	}
}
