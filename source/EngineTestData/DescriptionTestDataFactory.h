#pragma once

#include <variant>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/GenomeDescription.h>

class DescriptionTestDataFactory
{
    MAKE_SINGLETON(DescriptionTestDataFactory);

public:
    struct MuscleModeWrapper { MuscleMode value; };
    struct SensorModeWrapper { SensorMode value; };
    struct ReconnectorModeWrapper { ReconnectorMode value; };
    struct MemoryModeWrapper { MemoryMode value; };
    struct CommunicatorModeWrapper { CommunicatorMode value; };

    using CellTypeMode = std::variant<std::monostate, MuscleModeWrapper, SensorModeWrapper, ReconnectorModeWrapper, MemoryModeWrapper, CommunicatorModeWrapper>;

    struct CellParameter
    {
        CellType cellType;
        CellTypeMode mode = std::monostate{};
    };
    std::vector<CellParameter> getAllCellParameters() const;
    CellDescription createNonDefaultCellDescription(CellParameter cellParameter) const;
    ParticleDescription createNonDefaultParticleDescription() const;

    struct NodeParameter
    {
        CellTypeGenome cellTypeGenome;
        CellTypeMode mode = std::monostate{};
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
