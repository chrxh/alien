#pragma once

#include "ModelBasic/Definitions.h"
#include "Definitions.h"
#include "CudaInterface.cuh"

class DataConverter
{
public:
	DataConverter(SimulationDataForAccess& cudaData, NumberGenerator* numberGen);

	void add(ClusterDescription const& clusterToAdd);

	SimulationDataForAccess getResult() const;

private:
	void addCell(CellDescription const& cellToAdd, ClusterDescription const& cluster, ClusterData& cudaCluster
		, unordered_map<uint64_t, CellData*>& cellByIds);
	void resolveConnections(CellDescription const& cellToAdd, unordered_map<uint64_t, CellData*> const& cellByIds
		, CellData& cudaCell);

	void updateAngularMass(ClusterData& cluster);

private:
	SimulationDataForAccess& _cudaData;
	NumberGenerator* _numberGen;
};
