#pragma once

#include "ModelBasic/Definitions.h"
#include "Definitions.h"
#include "AccessTOs.cuh"

class DataConverter
{
public:
	DataConverter(DataAccessTO& dataTO, NumberGenerator* numberGen, SimulationParameters const& parameters);

	void updateData(DataChangeDescription const& data);

	DataDescription getDataDescription() const;

private:
	void addCluster(ClusterDescription const& clusterDesc);
	void addParticle(ParticleDescription const& particleDesc);

	void markDelCluster(uint64_t clusterId);
	void markDelParticle(uint64_t particleId);

	void markModifyCluster(ClusterChangeDescription const& clusterDesc);
	void markModifyParticle(ParticleChangeDescription const& particleDesc);

	void processDeletions();
	void processModifications();
	void addCell(CellDescription const& cellToAdd, ClusterDescription const& cluster, ClusterAccessTO& cudaCluster,
		unordered_map<uint64_t, int>& cellIndexTOByIds);
	void setConnections(CellDescription const& cellToAdd, CellAccessTO& cellTO, unordered_map<uint64_t, int> const& cellIndexByIds);

	void applyChangeDescription(ParticleChangeDescription const& particleChanges, ParticleAccessTO& particle);
	void applyChangeDescription(ClusterChangeDescription const& clusterChanges, ClusterAccessTO& cluster);
	void applyChangeDescription(CellChangeDescription const& cellChanges, CellAccessTO& cell);

private:
	DataAccessTO& _dataTO;
	NumberGenerator* _numberGen;
	SimulationParameters _parameters;

	std::unordered_set<uint64_t> _clusterIdsToDelete;
	std::unordered_map<uint64_t, ClusterChangeDescription> _clusterToModifyById;
	std::unordered_map<uint64_t, CellChangeDescription> _cellToModifyById;
	std::unordered_set<uint64_t> _particleIdsToDelete;
	std::unordered_map<uint64_t, ParticleChangeDescription> _particleToModifyById;
};
