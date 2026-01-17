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

    struct ObjectParameter
    {
        ObjectType objectType;
        CellType cellType;
        CellTypeMode mode = std::monostate{};
    };
    std::vector<ObjectParameter> getAllObjectParameters() const;
    ObjectDescription createNonDefaultObjectDescription(ObjectParameter objectParameter) const;
    EnergyDescription createNonDefaultEnergyDescription() const;

    struct NodeParameter
    {
        CellTypeGenome cellTypeGenome;
        CellTypeMode mode = std::monostate{};
    };
    std::vector<NodeParameter> getAllNodeParameters() const;
    NodeDescription createNonDefaultNodeDescription(NodeParameter nodeParameter) const;
    std::pair<CreatureDescription, GenomeDescription> createNonDefaultCreatureDescription(NodeParameter nodeParameter) const;

    bool compare(Description left, Description right) const;
    bool compare(ObjectDescription left, ObjectDescription right) const;
    bool compare(EnergyDescription left, EnergyDescription right) const;
    bool compare(ObjectDescription const& object, NodeDescription const& node) const;

private:
    CellTypeDescription createNonDefaultCellTypeDescription(ObjectParameter objectParameter) const;

    CellTypeGenomeDescription createNonDefaultCellTypeGenomeDescription(NodeParameter objectParameter) const;
};
