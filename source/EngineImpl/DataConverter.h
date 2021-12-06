#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/ChangeDescriptions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineGpuKernels/AccessTOs.cuh"
#include "Definitions.h"

#include <unordered_map>

class DataConverter
{
public:
    DataConverter(
        DataAccessTO& dataTO,
        SimulationParameters const& parameters,
        GpuSettings const& gpuConstants);

	void updateData(DataChangeDescription const& data);

	DataDescription getDataDescription() const;

private:
	struct CreateClusterReturnData
    {
        ClusterDescription cluster;
        std::unordered_map<int, int> cellTOIndexToCellDescIndex;
	};
    CreateClusterReturnData scanAndCreateClusterDescription(
        int startCellIndex,
        std::unordered_set<int>& freeCellIndices) const;
    CellDescription createCellDescription(int cellIndex) const;

	void addCell(CellChangeDescription const& cellToAdd, unordered_map<uint64_t, int>& cellIndexTOByIds);
    void addParticle(ParticleDescription const& particleDesc);

	void setConnections(CellChangeDescription const& cellToAdd, unordered_map<uint64_t, int> const& cellIndexByIds);

    int convertStringAndReturnStringIndex(std::string const& s);

private:
	DataAccessTO& _dataTO;
	SimulationParameters _parameters;
    GpuSettings _gpuConstants;

	std::unordered_set<uint64_t> _cellIdsToDelete;
	std::unordered_set<uint64_t> _particleIdsToDelete;
	std::unordered_map<uint64_t, CellChangeDescription> _cellToModifyById;
	std::unordered_map<uint64_t, ParticleChangeDescription> _particleToModifyById;
};
