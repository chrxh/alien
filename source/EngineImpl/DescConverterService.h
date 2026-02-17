#pragma once

#include <unordered_map>

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationParameters.h>

#include <EngineGpuKernels/Definitions.h>
#include <EngineGpuKernels/TOs.cuh>

#include "Definitions.h"

class DescConverterService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(DescConverterService);

public:
    Desc convertTOtoDescription(TOs const& to) const;
    TOs convertDescriptionToTO(Desc const& description) const;
    TOs convertDescriptionToTO(ExtendedObjectDesc const& extendedObject) const;
    TOs convertDescriptionToTO(EnergyDesc const& particle) const;
    TOs convertDescriptionToTO(uint64_t creatureId, GenomeDesc const& genome) const;

private:
    DescConverterService();

    ObjectDesc createObjectDesc(TOs const& to, int objectIndex) const;
    NodeDesc createNodeDesc(TOs const& to, NodeTO const* nodeTO) const;
    GenomeDesc createGenomeDesc(TOs const& to, int genomeIndex) const;
    CreatureDesc createCreatureDesc(TOs const& to, int creatureIndex) const;
    EnergyDesc createEnergyDesc(TOs const& to, int particleIndex) const;

    void convertGenomeToTO(
        std::vector<GenomeTO>& genomeTOs,
        std::vector<GeneTO>& geneTOs,
        std::vector<NodeTO>& nodeTOs,
        std::vector<uint8_t>& heap,
        GenomeDesc const& genome,
        std::unordered_map<uint64_t, uint64_t>& genomeTOIndexById) const;

    void convertCreatureToTO(
        std::vector<CreatureTO>& creatureTOs,
        CreatureDesc const& creatureDesc,
        std::unordered_map<uint64_t, uint64_t> const& genomeTOIndexById,
        std::unordered_map<uint64_t, uint64_t>& creatureTOIndexById) const;
    void convertObjectToTO(
        std::vector<ObjectTO>& objectTOs,
        std::vector<uint8_t>& heap,
        std::unordered_map<uint64_t, uint64_t>& objectTOIndexById,
        ObjectDesc const& cellToAdd,
        std::unordered_map<uint64_t, uint64_t> const& creatureTOIndexById) const;
    void addParticle(std::vector<EnergyTO>& particleTOs, EnergyDesc const& particleDesc) const;

    void setConnections(std::vector<ObjectTO>& objectTOs, ObjectDesc const& cellToAdd, std::unordered_map<uint64_t, uint64_t> const& objectIndexByIds) const;

    TOs provideDataTO(
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
