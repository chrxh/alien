#include "GenomeDescriptions.h"

#include "NumberGenerator.h"

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
    } else if (std::holds_alternative<ConstructorGenomeDescription_New>(_cellTypeData)) {
        return CellTypeGenome_Constructor;
    } else if (std::holds_alternative<SensorGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Sensor;
    } else if (std::holds_alternative<OscillatorGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Oscillator;
    } else if (std::holds_alternative<AttackerGenomeDescription>(_cellTypeData)) {
        return CellTypeGenome_Attacker;
    } else if (std::holds_alternative<InjectorGenomeDescription_New>(_cellTypeData)) {
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

GenomeDescription_New::GenomeDescription_New()
{
    _id = NumberGenerator::get().createObjectId();
}
