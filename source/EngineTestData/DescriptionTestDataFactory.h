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
    ObjectDesc createNonDefaultObjectDesc(ObjectParameter objectParameter) const;
    EnergyDesc createNonDefaultEnergyDesc() const;

    struct NodeParameter
    {
        CellTypeGenome cellTypeGenome;
        CellTypeMode mode = std::monostate{};
    };
    std::vector<NodeParameter> getAllNodeParameters() const;
    NodeDesc createNonDefaultNodeDesc(NodeParameter nodeParameter) const;
    std::pair<CreatureDesc, GenomeDesc> createNonDefaultCreatureDesc(NodeParameter nodeParameter) const;

    bool compare(Desc left, Desc right) const;
    bool compare(ObjectDesc left, ObjectDesc right) const;
    bool compare(EnergyDesc left, EnergyDesc right) const;
    bool compare(ObjectDesc const& object, NodeDesc const& node) const;

private:
    CellTypeDesc createNonDefaultCellTypeDesc(ObjectParameter objectParameter) const;

    CellTypeGenomeDesc createNonDefaultCellTypeGenomeDesc(NodeParameter objectParameter) const;
};
