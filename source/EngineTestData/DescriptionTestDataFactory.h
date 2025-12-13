#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/GenomeDescription.h>

class DescriptionTestDataFactory
{
    MAKE_SINGLETON(DescriptionTestDataFactory);

public:
    struct CellParameter
    {
        CellType cellType;
        std::optional<MuscleMode> muscleMode;
        std::optional<SensorMode> sensorMode;
    };
    std::vector<CellParameter> getAllCellParameters() const;
    CellDescription createNonDefaultCellDescription(CellParameter cellParameter) const;
    ParticleDescription createNonDefaultParticleDescription() const;

    struct NodeParameter
    {
        CellTypeGenome cellTypeGenome;
        std::optional<MuscleMode> muscleMode;
        std::optional<SensorMode> sensorMode;
    };
    std::vector<NodeParameter> getAllNodeParameters() const;
    NodeDescription createNonDefaultNodeDescription(NodeParameter nodeParameter) const;
    std::pair<CreatureDescription, GenomeDescription> createNonDefaultCreatureDescription(NodeParameter nodeParameter) const;

    bool compare(Description left, Description right) const;
    bool compare(CellDescription left, CellDescription right) const;
    bool compare(ParticleDescription left, ParticleDescription right) const;
    bool compare(CellDescription const& cell, NodeDescription const& node) const;

private:
    CellTypeDescription createNonDefaultCellTypeDescription(CellParameter cellParameter) const;

    CellTypeGenomeDescription createNonDefaultCellTypeGenomeDescription(NodeParameter cellParameter) const;
};
