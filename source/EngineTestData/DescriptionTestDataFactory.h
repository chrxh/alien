#pragma once

#include <vector>

#include "Base/Singleton.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/GenomeDescription.h"

class DescriptionTestDataFactory
{
    MAKE_SINGLETON(DescriptionTestDataFactory);

public:
    struct CellParameter
    {
        CellType cellType;
        MuscleMode muscleMode;
    };
    CellDescription createNonDefaultCellDescription(CellParameter cellParameter) const;
    ParticleDescription createNonDefaultParticleDescription() const;

    struct NodeParameter
    {
        CellTypeGenome cellTypeGenome;
        MuscleMode muscleMode;
    };
    NodeDescription createNonDefaultNodeDescription(NodeParameter nodeParameter) const;
    CreatureDescription createNonDefaultCreatureDescription(NodeParameter nodeParameter) const;

    bool compare(Description left, Description right) const;
    bool compare(CellDescription left, CellDescription right) const;
    bool compare(ParticleDescription left, ParticleDescription right) const;

private:
    CellTypeDescription createNonDefaultCellTypeDescription(CellParameter cellParameter) const;

    CellTypeGenomeDescription createNonDefaultCellTypeGenomeDescription(NodeParameter cellParameter) const;
};
