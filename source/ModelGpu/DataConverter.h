#pragma once

#include "ModelBasic/Definitions.h"
#include "Definitions.h"
#include "CudaInterface.cuh"

class DataConverter
{
public:
	DataConverter(SimulationDataForAccess& cudaData, NumberGenerator* numberGen);

	void updateData(DataChangeDescription const& data);

	SimulationDataForAccess getGpuData() const;
	DataDescription getDataDescription(IntRect const& requiredRect) const;

private:
	void addCluster(ClusterDescription const& clusterDesc);
	void addParticle(ParticleDescription const& particleDesc);

	void markDelCluster(uint64_t clusterId);
	void markDelParticle(uint64_t particleId);

	void markModifyCluster(ClusterChangeDescription const& clusterDesc);
	void markModifyParticle(ParticleChangeDescription const& particleDesc);

	void processDeletionsAndModifications();
	void addCell(CellDescription const& cellToAdd, ClusterDescription const& cluster, ClusterData& cudaCluster
		, unordered_map<uint64_t, CellData*>& cellByIds);
	void resolveConnections(CellDescription const& cellToAdd, unordered_map<uint64_t, CellData*> const& cellByIds
		, CellData& cudaCell);

	void applyChangeDescription(ParticleData& particle, ParticleChangeDescription const& particleChanges);
	void applyChangeDescription(ClusterData& cluster, ClusterChangeDescription const& clusterChanges);
	void applyChangeDescription(CellData& cell, CellChangeDescription const& cellChanges, ClusterChangeDescription const& clusterChanges);

	void updateAngularMass(ClusterData& cluster);

private:
	SimulationDataForAccess& _cudaData;
	NumberGenerator* _numberGen;

	std::unordered_set<uint64_t> _clusterIdsToDelete;
	std::unordered_map<uint64_t, ClusterChangeDescription> _clusterToModifyById;
	std::unordered_map<uint64_t, CellChangeDescription> _cellToModifyById;
	std::unordered_set<uint64_t> _particleIdsToDelete;
	std::unordered_map<uint64_t, ParticleChangeDescription> _particleToModifyById;
};
