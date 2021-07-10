#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineGpuKernels/AccessTOs.cuh"
#include "Definitions.h"

class DataConverter
{
public:
	DataConverter(DataAccessTO& dataTO, NumberGenerator* numberGen, SimulationParameters const& parameters, CudaConstants const& cudaConstants);

	void updateData(DataChangeDescription const& data);

	DataDescription getDataDescription() const;

private:
    ClusterDescription scanAndCreateClusterDescription(int startCellIndex, std::set<int>& freeCellIndices) const;
    CellDescription createCellDescription(int cellIndex) const;

	void addCell(CellChangeDescription const& cellToAdd, unordered_map<uint64_t, int>& cellIndexTOByIds);
    void addParticle(ParticleDescription const& particleDesc);

	void markDelCell(uint64_t cellId);
	void markDelParticle(uint64_t particleId);

	void markModifyCell(CellChangeDescription const& clusterDesc);
	void markModifyParticle(ParticleChangeDescription const& particleDesc);

	void processDeletions();
	void processModifications();
	void setConnections(CellChangeDescription const& cellToAdd, unordered_map<uint64_t, int> const& cellIndexByIds);

	void applyChangeDescription(ParticleChangeDescription const& particleChanges, ParticleAccessTO& particle);
	void applyChangeDescription(CellChangeDescription const& cellChanges, CellAccessTO& cell);

    int convertStringAndReturnStringIndex(QString const& s);

private:
	DataAccessTO& _dataTO;
	NumberGenerator* _numberGen;
	SimulationParameters _parameters;
    CudaConstants _cudaConstants;

	std::unordered_set<uint64_t> _cellIdsToDelete;
	std::unordered_set<uint64_t> _particleIdsToDelete;
	std::unordered_map<uint64_t, CellChangeDescription> _cellToModifyById;
	std::unordered_map<uint64_t, ParticleChangeDescription> _particleToModifyById;
};
