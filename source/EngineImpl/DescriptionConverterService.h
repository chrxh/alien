#pragma once

#include <unordered_map>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineGpuKernels/ObjectTO.cuh"
#include "Definitions.h"

class DescriptionConverterService
{
public:
    DescriptionConverterService(SimulationParameters const& parameters);

    ClusteredDataDescription convertTOtoClusteredDataDescription(DataTO const& dataTO) const;
    DataDescription convertTOtoDataDescription(DataTO const& dataTO) const;
    OverlayDescription convertTOtoOverlayDescription(DataTO const& dataTO) const;
    void convertDescriptionToTO(DataTO& result, ClusteredDataDescription const& description) const;
    void convertDescriptionToTO(DataTO& result, DataDescription const& description) const;
    void convertDescriptionToTO(DataTO& result, CellDescription const& cell) const;
    void convertDescriptionToTO(DataTO& result, ParticleDescription const& particle) const;

private:
	struct CreateClusterReturnData
    {
        ClusterDescription cluster;
        std::unordered_map<int, int> cellTOIndexToCellDescIndex;
	};
    CreateClusterReturnData scanAndCreateClusterDescription(
        DataTO const& dataTO,
        int startCellIndex,
        std::unordered_set<int>& freeCellIndices) const;
    CellDescription createCellDescription(DataTO const& dataTO, int cellIndex) const;

	void addCell(
        DataTO const& dataTO, CellDescription const& cellToAdd, std::unordered_map<uint64_t, int>& cellIndexTOByIds) const;
    void addParticle(DataTO const& dataTO, ParticleDescription const& particleDesc) const;

	void setConnections(
        DataTO const& dataTO, CellDescription const& cellToAdd, std::unordered_map<uint64_t, int> const& cellIndexByIds) const;

private:
	SimulationParameters _parameters;
};
