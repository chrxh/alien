#pragma once

#include <unordered_map>

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Description.h>
#include <EngineInterface/SimulationParameters.h>

#include <EngineGpuKernels/Definitions.h>
#include <EngineGpuKernels/TO.cuh>

#include "Definitions.h"

class DescriptionConverterService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(DescriptionConverterService);

public:
    Description convertTOtoDescription(TO const& to) const;
    TO convertDescriptionToTO(Description const& description) const;
    TO convertDescriptionToTO(CellDescription const& cell) const;
    TO convertDescriptionToTO(ParticleDescription const& particle) const;
    TO convertDescriptionToTO(uint64_t creatureId, GenomeDescription const& genome) const;

private:
    DescriptionConverterService();

    CellDescription createCellDescription(TO const& to, int cellIndex) const;
    NodeDescription createNodeDescription(TO const& to, NodeTO const* nodeTO) const;
    GenomeDescription createGenomeDescription(TO const& to, int genomeIndex) const;
    CreatureDescription createCreatureDescription(TO const& to, int creatureIndex) const;
    ParticleDescription createParticleDescription(TO const& to, int particleIndex) const;

    void convertGenomeToTO(
        std::vector<GenomeTO>& genomeTOs,
        std::vector<GeneTO>& geneTOs,
        std::vector<NodeTO>& nodeTOs,
        std::vector<uint8_t>& heap,
        GenomeDescription const& genome,
        std::unordered_map<uint64_t, uint64_t>& genomeTOIndexById) const;

    void convertCreatureToTO(
        std::vector<CreatureTO>& creatureTOs,
        CreatureDescription const& creatureDesc,
        std::unordered_map<uint64_t, uint64_t> const& genomeTOIndexById,
        std::unordered_map<uint64_t, uint64_t>& creatureTOIndexById) const;
    void convertCellToTO(
        std::vector<CellTO>& cellTOs,
        std::vector<uint8_t>& heap,
        std::unordered_map<uint64_t, uint64_t>& cellTOIndexById,
        CellDescription const& cellToAdd,
        std::optional<uint64_t> const& creatureId,
        std::unordered_map<uint64_t, uint64_t> const& creatureTOIndexById) const;
    void addParticle(std::vector<ParticleTO>& particleTOs, ParticleDescription const& particleDesc) const;

    void setConnections(std::vector<CellTO>& cellTOs, CellDescription const& cellToAdd, std::unordered_map<uint64_t, uint64_t> const& cellIndexByIds) const;

    TO provideDataTO(
        std::vector<CreatureTO> const& creatureTOs,
        std::vector<GenomeTO> const& genomeTOs,
        std::vector<GeneTO> const& geneTOs,
        std::vector<NodeTO> const& nodeTOs,
        std::vector<CellTO> const& cellTOs,
        std::vector<ParticleTO> const& particleTOs,
        std::vector<uint8_t> const& heap) const;

private:
    mutable TOProvider _collectionTOProvider;
};
