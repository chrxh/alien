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
    TO convertDescriptionToTO(ObjectDescription const& cell) const;
    TO convertDescriptionToTO(EnergyDescription const& particle) const;
    TO convertDescriptionToTO(uint64_t creatureId, GenomeDescription const& genome) const;

private:
    DescriptionConverterService();

    ObjectDescription createObjectDescription(TO const& to, int objectIndex) const;
    NodeDescription createNodeDescription(TO const& to, NodeTO const* nodeTO) const;
    GenomeDescription createGenomeDescription(TO const& to, int genomeIndex) const;
    CreatureDescription createCreatureDescription(TO const& to, int creatureIndex) const;
    EnergyDescription createEnergyDescription(TO const& to, int particleIndex) const;

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
        std::vector<ObjectTO>& objectTOs,
        std::vector<uint8_t>& heap,
        std::unordered_map<uint64_t, uint64_t>& objectTOIndexById,
        ObjectDescription const& cellToAdd,
        std::optional<uint64_t> const& creatureId,
        std::unordered_map<uint64_t, uint64_t> const& creatureTOIndexById) const;
    void addParticle(std::vector<EnergyTO>& particleTOs, EnergyDescription const& particleDesc) const;

    void setConnections(std::vector<ObjectTO>& objectTOs, ObjectDescription const& cellToAdd, std::unordered_map<uint64_t, uint64_t> const& objectIndexByIds) const;

    TO provideDataTO(
        std::vector<CreatureTO> const& creatureTOs,
        std::vector<GenomeTO> const& genomeTOs,
        std::vector<GeneTO> const& geneTOs,
        std::vector<NodeTO> const& nodeTOs,
        std::vector<ObjectTO> const& objectTOs,
        std::vector<EnergyTO> const& particleTOs,
        std::vector<uint8_t> const& heap) const;

private:
    mutable TOProvider _collectionTOProvider;
};
