#pragma once

#include "ModelBasic/Definitions.h"
#include "Definitions.h"
#include "CudaInterface.cuh"

class DataConverter
{
public:
	DataConverter(SimulationAccessTO* cudaData, NumberGenerator* numberGen);

	void updateData(DataChangeDescription const& data);

	DataDescription getDataDescription(IntRect const& requiredRect) const;

private:
	void addCluster(ClusterDescription const& clusterDesc);
	void addParticle(ParticleDescription const& particleDesc);

	void markDelCluster(uint64_t clusterId);
	void markDelParticle(uint64_t particleId);

	void markModifyCluster(ClusterChangeDescription const& clusterDesc);
	void markModifyParticle(ParticleChangeDescription const& particleDesc);

	void processDeletionsAndModifications();
	void addCell(CellDescription const& cellToAdd, ClusterDescription const& cluster, ClusterAccessTO& cudaCluster,
		unordered_map<uint64_t, int>& cellIndexTOByIds);
	void setConnections(CellDescription const& cellToAdd, CellAccessTO& cellTO, unordered_map<uint64_t, int> const& cellIndexByIds);

	void applyChangeDescription(ParticleAccessTO& particle, ParticleChangeDescription const& particleChanges);
	void applyChangeDescription(ClusterAccessTO& cluster, ClusterChangeDescription const& clusterChanges);
	void applyChangeDescription(CellAccessTO& cell, CellChangeDescription const& cellChanges);

private:
	SimulationAccessTO* _simulationTO;
	NumberGenerator* _numberGen;

	std::unordered_set<uint64_t> _clusterIdsToDelete;
	std::unordered_map<uint64_t, ClusterChangeDescription> _clusterToModifyById;
	std::unordered_map<uint64_t, CellChangeDescription> _cellToModifyById;
	std::unordered_set<uint64_t> _particleIdsToDelete;
	std::unordered_map<uint64_t, ParticleChangeDescription> _particleToModifyById;
};
