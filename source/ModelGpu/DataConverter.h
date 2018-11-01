#pragma once

#include "ModelBasic/Definitions.h"
#include "Definitions.h"
#include "CudaInterface.cuh"

class DataConverter
{
public:
	DataConverter(SimulationDataForAccess& cudaData, NumberGenerator* numberGen);

	void addCluster(ClusterDescription const& clusterDesc);
	void addParticle(ParticleDescription const& particleDesc);
	void delCluster(uint64_t clusterId);
	void delParticle(uint64_t particleId);

	void finalize();

	SimulationDataForAccess getGpuData() const;
	DataDescription getDataDescription(IntRect const& requiredRect) const;

private:
	void addCell(CellDescription const& cellToAdd, ClusterDescription const& cluster, ClusterData& cudaCluster
		, unordered_map<uint64_t, CellData*>& cellByIds);
	void resolveConnections(CellDescription const& cellToAdd, unordered_map<uint64_t, CellData*> const& cellByIds
		, CellData& cudaCell);

	void updateAngularMass(ClusterData& cluster);

private:
	SimulationDataForAccess& _cudaData;
	NumberGenerator* _numberGen;

	std::unordered_set<uint64_t> _clusterIdsToDelete;
	std::unordered_set<uint64_t> _particleIdsToDelete;
};
