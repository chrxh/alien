#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineGpuKernels/AccessTOs.cuh"
#include "Definitions.h"

#include <unordered_map>

class DataConverter
{
public:
    DataConverter(SimulationParameters const& parameters, GpuSettings const& gpuConstants);

    enum class SortTokens {No, Yes};
    DataDescription convertAccessTOtoDataDescription(DataAccessTO const& dataTO, SortTokens sortTokens = SortTokens::No)
        const;
    OverlayDescription convertAccessTOtoOverlayDescription(DataAccessTO const& dataTO) const;
    void convertDataDescriptionToAccessTO(DataAccessTO& result, DataDescription const& description) const;
    void convertCellDescriptionToAccessTO(DataAccessTO& result, CellDescription const& cell) const;
    void convertParticleDescriptionToAccessTO(DataAccessTO& result, ParticleDescription const& particle) const;

private:
	struct CreateClusterReturnData
    {
        ClusterDescription cluster;
        std::unordered_map<int, int> cellTOIndexToCellDescIndex;
	};
    CreateClusterReturnData scanAndCreateClusterDescription(
        DataAccessTO const& dataTO,
        int startCellIndex,
        std::unordered_set<int>& freeCellIndices) const;
    CellDescription createCellDescription(DataAccessTO const& dataTO, int cellIndex) const;

	void addCell(
        DataAccessTO const& dataTO, CellDescription const& cellToAdd, std::unordered_map<uint64_t, int>& cellIndexTOByIds) const;
    void addParticle(DataAccessTO const& dataTO, ParticleDescription const& particleDesc) const;

	void setConnections(
        DataAccessTO const& dataTO, CellDescription const& cellToAdd, std::unordered_map<uint64_t, int> const& cellIndexByIds) const;

    int convertStringAndReturnStringIndex(DataAccessTO const& dataTO, std::string const& s) const;

private:
	SimulationParameters _parameters;
    GpuSettings _gpuConstants;
};
