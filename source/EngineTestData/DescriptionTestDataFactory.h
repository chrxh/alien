#pragma once

#include <vector>

#include "Base/Singleton.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/CreatureDescription.h"

class DescriptionTestDataFactory
{
    MAKE_SINGLETON(DescriptionTestDataFactory);

public:
    struct CellParameter
    {
        CellType cellType;
        MuscleMode muscleMode;
    };
    CellDescription createRandomCellDescription(CellParameter cellParameter) const;
    ParticleDescription createRandomParticleDescription() const;

    struct NodeParameter
    {
        CellTypeGenome cellTypeGenome;
        MuscleMode muscleMode;
    };
    CreatureDescription createRandomCreatureDescription(NodeParameter nodeParameter) const;

    bool compare(CollectionDescription left, CollectionDescription right) const;
    bool compare(CellDescription left, CellDescription right) const;
    bool compare(ParticleDescription left, ParticleDescription right) const;

private:
    CellTypeDescription createRandomCellTypeDescription(CellParameter cellParameter) const;

    NodeDescription createRandomNodeDescription(NodeParameter nodeParameter) const;
    CellTypeGenomeDescription_New createRandomCellTypeGenomeDescription(NodeParameter cellParameter) const;
};
