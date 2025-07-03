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
    if (std::holds_alternative<BaseGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Base;
    } else if (std::holds_alternative<DepotGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Depot;
    } else if (std::holds_alternative<ConstructorGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Constructor;
    } else if (std::holds_alternative<SensorGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Sensor;
    } else if (std::holds_alternative<GeneratorGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Generator;
    } else if (std::holds_alternative<AttackerGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Attacker;
    } else if (std::holds_alternative<InjectorGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Injector;
    } else if (std::holds_alternative<MuscleGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Muscle;
    } else if (std::holds_alternative<DefenderGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Defender;
    } else if (std::holds_alternative<ReconnectorGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Reconnector;
    } else if (std::holds_alternative<DetonatorGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Detonator;
    }
    CHECK(false);
}
