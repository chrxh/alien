#include "ChangeDescriptions.h"


DataChangeDescription::DataChangeDescription(DataDescription const & dataBefore, DataDescription const & dataAfter)
{
	vector<uint64_t> clusterIdsAfter;
	std::transform(dataAfter.clusters.begin(), dataAfter.clusters.end(), clusterIdsAfter.begin(), [](auto const& cluster) {
		return cluster.id;
	});
	unordered_set<uint64_t> clusterIdSetAfter(clusterIdsAfter.begin(), clusterIdsAfter.end());

	for (auto const& clusterBefore : dataBefore.clusters) {
		if(clusterIdSetAfter.find(clusterBefore.id) == clusterIdSetAfter.end()) {
			addDeletedCluster(clusterBefore.id);
		}
		else {
			clusterIdSetAfter.erase(clusterBefore.id);
		}
	}
}
