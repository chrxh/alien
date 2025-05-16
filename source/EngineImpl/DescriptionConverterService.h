#pragma once

#include <unordered_map>

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineGpuKernels/ObjectTO.cuh"
#include "EngineGpuKernels/Definitions.h"
#include "Definitions.h"

class DescriptionConverterService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(DescriptionConverterService);

public:
    ClusteredDataDescription convertTOtoClusteredDataDescription(DataTO const& dataTO) const;
    DataDescription convertTOtoDataDescription(DataTO const& dataTO) const;
    OverlayDescription convertTOtoOverlayDescription(DataTO const& dataTO) const;
    DataTO convertDescriptionToTO(ClusteredDataDescription const& description) const;
    DataTO convertDescriptionToTO(DataDescription const& description) const;
    DataTO convertDescriptionToTO(CellDescription const& cell) const;
    DataTO convertDescriptionToTO(ParticleDescription const& particle) const;

private:
    DescriptionConverterService();

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
        std::vector<CellTO>& cellTOs,
        std::vector<uint8_t>& heap,
        CellDescription const& cellToAdd,
        std::unordered_map<uint64_t, uint64_t>& cellIndexTOByIds) const;
    void addParticle(std::vector<ParticleTO>& particleTOs, ParticleDescription const& particleDesc) const;

	void setConnections(std::vector<CellTO>& cellTOs, CellDescription const& cellToAdd, std::unordered_map<uint64_t, uint64_t> const& cellIndexByIds) const;

    DataTO provideDataTO(std::vector<CellTO> const& cellTOs, std::vector<ParticleTO> const& particleTOs, std::vector<uint8_t> const& heap) const;

private:
    mutable DataTOProvider _dataTOProvider;
};
