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
    CollectionDescription convertTOtoDescription(CollectionTO const& collectionTO) const;
    OverlayDescription convertTOtoOverlayDescription(CollectionTO const& collectionTO) const;
    CollectionTO convertDescriptionToTO(CollectionDescription const& description) const;
    CollectionTO convertDescriptionToTO(CellDescription const& cell) const;
    CollectionTO convertDescriptionToTO(ParticleDescription const& particle) const;

private:
    DescriptionConverterService();

    CellDescription createCellDescription(CollectionTO const& collectionTO, int cellIndex, std::unordered_map<uint64_t, uint64_t> const& genomeIdByTOIndex) const;
    GenomeDescription_New createGenomeDescription(CollectionTO const& collectionTO, int genomeIndex, std::unordered_map<uint64_t, uint64_t>& genomeIdByTOIndex) const;

    void convertGenomeToTO(
        std::vector<GenomeTO>& genomeTOs,
        std::vector<GeneTO>& geneTOs,
        std::vector<NodeTO>& nodeTOs,
        std::vector<uint8_t>& heap,
        GenomeDescription_New const& genomeDesc,
        std::unordered_map<uint64_t, uint64_t>& genomeTOIndexById) const;
    void convertCellToTO(
        std::vector<CellTO>& cellTOs,
        std::vector<uint8_t>& heap,
        std::unordered_map<uint64_t, uint64_t>& cellTOIndexById,
        CellDescription const& cellToAdd,
        std::unordered_map<uint64_t, uint64_t> const& genomeTOIndexById) const;
    void addParticle(std::vector<ParticleTO>& particleTOs, ParticleDescription const& particleDesc) const;

	void setConnections(std::vector<CellTO>& cellTOs, CellDescription const& cellToAdd, std::unordered_map<uint64_t, uint64_t> const& cellIndexByIds) const;

    CollectionTO provideDataTO(
        std::vector<GenomeTO> const& genomeTOs,
        std::vector<GeneTO> const& geneTOs,
        std::vector<NodeTO> const& nodeTOs,
        std::vector<CellTO> const& cellTOs,
        std::vector<ParticleTO> const& particleTOs,
        std::vector<uint8_t> const& heap) const;

private:
    mutable CollectionTOProvider _collectionTOProvider;
};
