#pragma once

#include <vector>

#include "Base/Singleton.h"
#include "EngineInterface/Descriptions.h"
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
    CellDescription createRandomCellDescription(CellParameter cellParameter) const;
    ParticleDescription createRandomParticleDescription() const;

    struct NodeParameter
    {
        CellTypeGenome cellTypeGenome;
        MuscleMode muscleMode;
    };
    NodeDescription createRandomNodeDescription(NodeParameter nodeParameter) const;
    CreatureDescription createRandomCreatureDescription(NodeParameter nodeParameter) const;

    bool compare(CollectionDescription left, CollectionDescription right) const;
    bool compare(CellDescription left, CellDescription right) const;
    bool compare(ParticleDescription left, ParticleDescription right) const;

private:
    float getRandomFloat(float min = -1.0f, float max = 1.0f) const;
    RealVector2D getRandomFloat2(float min = -1.0f, float max = 1.0f) const;
    int getRandomInt(int min = 0, int max = 42) const;

    CellTypeDescription createRandomCellTypeDescription(CellParameter cellParameter) const;

    CellTypeGenomeDescription createRandomCellTypeGenomeDescription(NodeParameter cellParameter) const;
};
