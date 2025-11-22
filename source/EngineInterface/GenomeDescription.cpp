#include "GenomeDescription.h"

#include "NumberGenerator.h"

NeuralNetworkGenomeDescription::NeuralNetworkGenomeDescription()
{
    _weights.resize(MAX_CHANNELS * MAX_CHANNELS, 0);
    _biases.resize(MAX_CHANNELS, 0);
    _activationFunctions.resize(MAX_CHANNELS, ActivationFunction_Identity);
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        _weights[i * MAX_CHANNELS + i] = 1.0f;
    }
}

NeuralNetworkGenomeDescription& NeuralNetworkGenomeDescription::weight(int row, int col, float value)
{
    _weights[row * MAX_CHANNELS + col] = value;
    return *this;
}

GenomeDescription::GenomeDescription()
{
    _id = NumberGenerator::get().createId();
}

GenomeDescription GenomeDescription::id(uint64_t id)
{
    NumberGenerator::get().adaptMaxIds({.entityId = id});
    _id = id;
    return *this;
}

SensorMode SensorGenomeDescription::getMode() const
{
    if (std::holds_alternative<DetectEnergyGenomeDescription>(_mode)) {
        return SensorMode_DetectEnergy;
    } else if (std::holds_alternative<DetectStructureGenomeDescription>(_mode)) {
        return SensorMode_DetectStructure;
    } else if (std::holds_alternative<DetectFreeCellGenomeDescription>(_mode)) {
        return SensorMode_DetectFreeCell;
    } else if (std::holds_alternative<DetectCreatureGenomeDescription>(_mode)) {
        return SensorMode_DetectCreature;
    }
    CHECK(false);
}

MuscleMode MuscleGenomeDescription::getMode() const
{
    if (std::holds_alternative<AutoBendingGenomeDescription>(_mode)) {
        return MuscleMode_AutoBending;
    } else if (std::holds_alternative<ManualBendingGenomeDescription>(_mode)) {
        return MuscleMode_ManualBending;
    } else if (std::holds_alternative<AngleBendingGenomeDescription>(_mode)) {
        return MuscleMode_AngleBending;
    } else if (std::holds_alternative<AutoCrawlingGenomeDescription>(_mode)) {
        return MuscleMode_AutoCrawling;
    } else if (std::holds_alternative<ManualCrawlingGenomeDescription>(_mode)) {
        return MuscleMode_ManualCrawling;
    } else if (std::holds_alternative<DirectMovementGenomeDescription>(_mode)) {
        return MuscleMode_DirectMovement;
    }
    CHECK(false);
}

CellTypeGenome NodeDescription::getCellType() const
{
    if (std::holds_alternative<BaseGenomeDescription>(_cellType)) {
        return CellTypeGenome_Base;
    } else if (std::holds_alternative<DepotGenomeDescription>(_cellType)) {
        return CellTypeGenome_Depot;
    } else if (std::holds_alternative<ConstructorGenomeDescription>(_cellType)) {
        return CellTypeGenome_Constructor;
    } else if (std::holds_alternative<SensorGenomeDescription>(_cellType)) {
        return CellTypeGenome_Sensor;
    } else if (std::holds_alternative<GeneratorGenomeDescription>(_cellType)) {
        return CellTypeGenome_Generator;
    } else if (std::holds_alternative<AttackerGenomeDescription>(_cellType)) {
        return CellTypeGenome_Attacker;
    } else if (std::holds_alternative<InjectorGenomeDescription>(_cellType)) {
        return CellTypeGenome_Injector;
    } else if (std::holds_alternative<MuscleGenomeDescription>(_cellType)) {
        return CellTypeGenome_Muscle;
    } else if (std::holds_alternative<DefenderGenomeDescription>(_cellType)) {
        return CellTypeGenome_Defender;
    } else if (std::holds_alternative<ReconnectorGenomeDescription>(_cellType)) {
        return CellTypeGenome_Reconnector;
    } else if (std::holds_alternative<DetonatorGenomeDescription>(_cellType)) {
        return CellTypeGenome_Detonator;
    }
    CHECK(false);
}
