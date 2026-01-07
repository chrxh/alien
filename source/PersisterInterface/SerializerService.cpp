#include "SerializerService.h"

#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <boost/algorithm/string/split.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/adaptors.hpp>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>

#include <Base/LoggingService.h>
#include <Base/Resources.h>
#include <Base/VersionParserService.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationParameters.h>

#include "SettingsParserService.h"

#include <zstr.hpp>

#define SPLIT_SERIALIZATION(Classname) \
    template <class Archive> \
    void save(Archive& ar, Classname const& data) \
    { \
        loadSave(SerializationTask::Save, ar, const_cast<Classname&>(data)); \
    } \
    template <class Archive> \
    void load(Archive& ar, Classname& data) \
    { \
        loadSave(SerializationTask::Load, ar, data); \
    }

namespace
{
    enum class SerializationTask
    {
        Load,
        Save
    };
}

namespace cereal
{
    using VariantData = std::variant<
        int,
        float,
        bool,
        double,
        std::string,
        uint64_t,
        uint32_t,
        uint16_t,
        uint8_t,
        int64_t,
        int16_t,
        int8_t,
        RealVector2D,
        std::optional<bool>,
        std::optional<uint64_t>,
        std::optional<uint8_t>,
        std::optional<int8_t>,
        std::optional<int>,
        std::optional<float>,
        std::optional<RealVector2D>,
        std::vector<bool>,
        std::vector<uint8_t>,
        std::vector<int8_t>,
        std::vector<int>,
        std::vector<float>,
        std::vector<std::vector<uint8_t>>,
        std::vector<std::vector<int8_t>>,
        std::vector<std::vector<int>>,
        std::vector<std::vector<float>>>;

    template <class Archive>
    std::unordered_map<int, VariantData> getLoadSaveMap(SerializationTask task, Archive& ar)
    {
        std::unordered_map<int, VariantData> loadSaveMap;
        if (task == SerializationTask::Load) {
            ar(loadSaveMap);
        }
        return loadSaveMap;
    }
    template <typename T>
    void loadSave(SerializationTask task, std::unordered_map<int, VariantData>& loadSaveMap, int key, T& value, T const& defaultValue)
    {
        if (task == SerializationTask::Load) {
            auto findResult = loadSaveMap.find(key);
            if (findResult != loadSaveMap.end()) {
                auto variantData = findResult->second;
                value = std::get<T>(variantData);
            } else {
                value = defaultValue;
            }
        } else {
            loadSaveMap.emplace(key, value);
        }
    }
    template <class Archive>
    void processLoadSaveMap(SerializationTask task, Archive& ar, std::unordered_map<int, VariantData>& loadSaveMap)
    {
        if (task == SerializationTask::Save) {
            ar(loadSaveMap);
        }
    }

    template <class Archive>
    void serialize(Archive& ar, IntVector2D& data)
    {
        ar(data.x, data.y);
    }
    template <class Archive>
    void serialize(Archive& ar, RealVector2D& data)
    {
        ar(data.x, data.y);
    }
}

/************************************************************************/
/* Genome data                                                          */
/************************************************************************/
namespace
{
    auto constexpr Id_Genome_Id = 0;
    auto constexpr Id_Genome_Name = 1;
    auto constexpr Id_Genome_FrontAngle = 2;

    auto constexpr Id_Gene_Name = 0;
    auto constexpr Id_Gene_Shape = 1;
    auto constexpr Id_Gene_NumBranches = 2;
    auto constexpr Id_Gene_Separation = 3;
    auto constexpr Id_Gene_AngleAlignment = 4;
    auto constexpr Id_Gene_Stiffness = 5;
    auto constexpr Id_Gene_ConnectionDistance = 6;
    auto constexpr Id_Gene_NumRepetitions = 7;

    auto constexpr Id_Node_ReferenceAngle = 0;
    auto constexpr Id_Node_Color = 1;
    auto constexpr Id_Node_NumAdditionalConnections = 2;

    auto constexpr Id_SignalRestrictionGenome_Mode = 0;  // Replaces Active (0=inactive, 1=active, 2=conditional)
    auto constexpr Id_SignalRestrictionGenome_BaseAngle = 1;
    auto constexpr Id_SignalRestrictionGenome_OpeneningAngle = 2;

    auto constexpr Id_NeuralNetworkGenome_Weights = 0;
    auto constexpr Id_NeuralNetworkGenome_Biases = 1;
    auto constexpr Id_NeuralNetworkGenome_ActivationFunctions = 2;

    auto constexpr Id_DepotGenome_Mode = 0;
    auto constexpr Id_DepotGenome_storageLimit = 1;
    auto constexpr Id_DepotGenome_InitialStoredUsableEnergy = 2;

    auto constexpr Id_DefenderGenome_Mode = 0;

    auto constexpr Id_ConstructorGenome_AutoTriggerInterval = 0;
    auto constexpr Id_ConstructorGenome_GeneIndex = 1;
    auto constexpr Id_ConstructorGenome_ConstructionActivationTime = 2;
    auto constexpr Id_ConstructorGenome_ConstructionAngle = 3;
    auto constexpr Id_ConstructorGenome_ProvideEnergy = 4;

    auto constexpr Id_SensorGenome_AutoTriggerInterval = 0;
    auto constexpr Id_SensorGenome_MinRange = 1;
    auto constexpr Id_SensorGenome_MaxRange = 2;

    auto constexpr Id_SensorModeGenome_DetectEnergy_MinDensity = 0;

    auto constexpr Id_SensorModeGenome_DetectFreeCell_MinDensity = 0;
    auto constexpr Id_SensorModeGenome_DetectFreeCell_RestrictToColor = 1;

    auto constexpr Id_SensorModeGenome_DetectCreature_MinNumCells = 0;
    auto constexpr Id_SensorModeGenome_DetectCreature_MaxNumCells = 1;
    auto constexpr Id_SensorModeGenome_DetectCreature_RestrictToColor = 2;
    auto constexpr Id_SensorModeGenome_DetectCreature_RestrictToLineage = 3;

    auto constexpr Id_MuscleModeGenome_AutoBending_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleModeGenome_AutoBending_ForwardBackwardRatio = 4;

    auto constexpr Id_MuscleModeGenome_ManualBending_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleModeGenome_ManualBending_ForwardBackwardRatio = 1;

    auto constexpr Id_MuscleModeGenome_AngleBending_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleModeGenome_AngleBending_AttractionRepulsionRatio = 1;

    auto constexpr Id_MuscleModeGenome_AutoCrawling_MaxDistanceDeviation = 0;
    auto constexpr Id_MuscleModeGenome_AutoCrawling_ForwardBackwardRatio = 1;

    auto constexpr Id_MuscleModeGenome_ManualCrawling_MaxDistanceDeviation = 0;
    auto constexpr Id_MuscleModeGenome_ManualCrawling_ForwardBackwardRatio = 1;

    auto constexpr Id_GeneratorGenome_AutoTriggerInterval = 0;
    auto constexpr Id_GeneratorGenome_PulseType = 1;
    auto constexpr Id_GeneratorGenome_AlternationInterval = 2;

    auto constexpr Id_AttackerModeGenome_FreeCell_RestrictToColor = 0;

    auto constexpr Id_AttackerModeGenome_Creature_MinNumCells = 0;
    auto constexpr Id_AttackerModeGenome_Creature_MaxNumCells = 1;
    auto constexpr Id_AttackerModeGenome_Creature_RestrictToColor = 2;
    auto constexpr Id_AttackerModeGenome_Creature_RestrictToLineage = 3;

    auto constexpr Id_InjectorGenome_GeneIndex = 0;

    auto constexpr Id_ReconnectorModeGenome_FreeCell_RestrictToColor = 0;

    auto constexpr Id_ReconnectorModeGenome_Creature_MinNumCells = 0;
    auto constexpr Id_ReconnectorModeGenome_Creature_MaxNumCells = 1;
    auto constexpr Id_ReconnectorModeGenome_Creature_RestrictToColor = 2;
    auto constexpr Id_ReconnectorModeGenome_Creature_RestrictToLineage = 3;

    auto constexpr Id_DetonatorGenome_Countdown = 0;

    auto constexpr Id_DigestorGenome_RawEnergyConductivity = 0;

    auto constexpr Id_SignalEntryGenome_Channels = 0;

    auto constexpr Id_SignalDelayGenome_Delay = 0;

    auto constexpr Id_SignalRecorderGenome_ReadOnly = 0;
    auto constexpr Id_SignalRecorderGenome_NumSavedSignalEntries = 1;

    auto constexpr Id_SignalStorageGenome_ReadOnly = 0;

    auto constexpr Id_SignalIntegratorGenome_NewSignalWeight = 0;

    auto constexpr Id_MemoryGenome_ChannelBitMask = 0;

    auto constexpr Id_SenderGenome_Range = 0;
    auto constexpr Id_SenderGenome_MaxTimesSent = 1;

    auto constexpr Id_ReceiverGenome_ChannelBitMask = 0;
    auto constexpr Id_ReceiverGenome_RestrictToColor = 1;
    auto constexpr Id_ReceiverGenome_RestrictToLineage = 2;
}

namespace cereal
{
    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetworkGenomeDescription& data)
    {
        NeuralNetworkGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_NeuralNetworkGenome_Weights, data._weights, defaultObject._weights);
        loadSave(task, auxiliaries, Id_NeuralNetworkGenome_Biases, data._biases, defaultObject._biases);
        loadSave(task, auxiliaries, Id_NeuralNetworkGenome_ActivationFunctions, data._activationFunctions, defaultObject._activationFunctions);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(NeuralNetworkGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, BaseGenomeDescription& data)
    {
        BaseGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(BaseGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DepotGenomeDescription& data)
    {
        DepotGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DepotGenome_storageLimit, data._storageLimit, defaultObject._storageLimit);
        loadSave(task, auxiliaries, Id_DepotGenome_InitialStoredUsableEnergy, data._initialStoredUsableEnergy, defaultObject._initialStoredUsableEnergy);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DepotGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorGenomeDescription& data)
    {
        ConstructorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ConstructorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        loadSave(task, auxiliaries, Id_ConstructorGenome_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        loadSave(
            task, auxiliaries, Id_ConstructorGenome_ConstructionActivationTime, data._constructionActivationTime, defaultObject._constructionActivationTime);
        loadSave(task, auxiliaries, Id_ConstructorGenome_ConstructionAngle, data._constructionAngle, defaultObject._constructionAngle);
        loadSave(task, auxiliaries, Id_ConstructorGenome_ProvideEnergy, data._provideEnergy, defaultObject._provideEnergy);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ConstructorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TelemetryGenomeDescription& data)
    {
        //TelemetryGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(TelemetryGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectEnergyGenomeDescription& data)
    {
        DetectEnergyGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectEnergy_MinDensity, data._minDensity, defaultObject._minDensity);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectEnergyGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectStructureGenomeDescription& data)
    {
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectStructureGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectFreeCellGenomeDescription& data)
    {
        DetectFreeCellGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectFreeCell_MinDensity, data._minDensity, defaultObject._minDensity);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectFreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectFreeCellGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectCreatureGenomeDescription& data)
    {
        DetectCreatureGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectCreature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectCreature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectCreature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectCreature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectCreatureGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorGenomeDescription& data)
    {
        SensorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        loadSave(task, auxiliaries, Id_SensorGenome_MinRange, data._minRange, defaultObject._minRange);
        loadSave(task, auxiliaries, Id_SensorGenome_MaxRange, data._maxRange, defaultObject._maxRange);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(SensorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeneratorGenomeDescription& data)
    {
        GeneratorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_GeneratorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        loadSave(task, auxiliaries, Id_GeneratorGenome_PulseType, data._pulseType, defaultObject._pulseType);
        loadSave(task, auxiliaries, Id_GeneratorGenome_AlternationInterval, data._alternationInterval, defaultObject._alternationInterval);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(GeneratorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackFreeCellGenomeDescription& data)
    {
        AttackFreeCellGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_AttackerModeGenome_FreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackFreeCellGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackCreatureGenomeDescription& data)
    {
        AttackCreatureGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_AttackerModeGenome_Creature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_AttackerModeGenome_Creature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_AttackerModeGenome_Creature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_AttackerModeGenome_Creature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackCreatureGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerGenomeDescription& data)
    {
        AttackerGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(AttackerGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorGenomeDescription& data)
    {
        InjectorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_InjectorGenome_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(InjectorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoBendingGenomeDescription& data)
    {
        AutoBendingGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_AutoBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_AutoBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AutoBendingGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualBendingGenomeDescription& data)
    {
        ManualBendingGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_ManualBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_ManualBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ManualBendingGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AngleBendingGenomeDescription& data)
    {
        AngleBendingGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_AngleBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(
            task,
            auxiliaries,
            Id_MuscleModeGenome_AngleBending_AttractionRepulsionRatio,
            data._attractionRepulsionRatio,
            defaultObject._attractionRepulsionRatio);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AngleBendingGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoCrawlingGenomeDescription& data)
    {
        AutoCrawlingGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_AutoCrawling_MaxDistanceDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_AutoCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AutoCrawlingGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualCrawlingGenomeDescription& data)
    {
        ManualCrawlingGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_ManualCrawling_MaxDistanceDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_ManualCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ManualCrawlingGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DirectMovementGenomeDescription& data)
    {
        DirectMovementGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DirectMovementGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleGenomeDescription& data)
    {
        MuscleGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(MuscleGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderGenomeDescription& data)
    {
        DefenderGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DefenderGenome_Mode, data._mode, defaultObject._mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DefenderGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectStructureGenomeDescription& data)
    {
        ReconnectStructureGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectStructureGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectFreeCellGenomeDescription& data)
    {
        ReconnectFreeCellGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_FreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectFreeCellGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectCreatureGenomeDescription& data)
    {
        ReconnectCreatureGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_Creature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_Creature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_Creature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_Creature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectCreatureGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorGenomeDescription& data)
    {
        ReconnectorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(ReconnectorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorGenomeDescription& data)
    {
        DetonatorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DetonatorGenome_Countdown, data._countdown, defaultObject._countdown);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetonatorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DigestorGenomeDescription& data)
    {
        DigestorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DigestorGenome_RawEnergyConductivity, data._rawEnergyConductivity, defaultObject._rawEnergyConductivity);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DigestorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDelayGenomeDescription& data)
    {
        SignalDelayGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalDelayGenome_Delay, data._delay, defaultObject._delay);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalDelayGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRecorderGenomeDescription& data)
    {
        SignalRecorderGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalRecorderGenome_ReadOnly, data._readOnly, defaultObject._readOnly);
        loadSave(task, auxiliaries, Id_SignalRecorderGenome_NumSavedSignalEntries, data._numWrittenSignalEntries, defaultObject._numWrittenSignalEntries);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRecorderGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalStorageGenomeDescription& data)
    {
        SignalStorageGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalStorageGenome_ReadOnly, data._readOnly, defaultObject._readOnly);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalStorageGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalIntegratorGenomeDescription& data)
    {
        SignalIntegratorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalIntegratorGenome_NewSignalWeight, data._newSignalWeight, defaultObject._newSignalWeight);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalIntegratorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalEntryGenomeDescription& data)
    {
        SignalEntryGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalEntryGenome_Channels, data._channels, defaultObject._channels);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalEntryGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MemoryGenomeDescription& data)
    {
        MemoryGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MemoryGenome_ChannelBitMask, data._channelBitMask, defaultObject._channelBitMask);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
        ar(data._signalEntries);
    }
    SPLIT_SERIALIZATION(MemoryGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SenderGenomeDescription& data)
    {
        SenderGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SenderGenome_Range, data._range, defaultObject._range);
        loadSave(task, auxiliaries, Id_SenderGenome_MaxTimesSent, data._maxTimesSent, defaultObject._maxTimesSent);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SenderGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReceiverGenomeDescription& data)
    {
        ReceiverGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReceiverGenome_ChannelBitMask, data._channelBitMask, defaultObject._channelBitMask);
        loadSave(task, auxiliaries, Id_ReceiverGenome_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_ReceiverGenome_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReceiverGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CommunicatorGenomeDescription& data)
    {
        CommunicatorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(CommunicatorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRestrictionGenomeDescription& data)
    {
        SignalRestrictionGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        
        // Backward compatibility: old files stored a bool at key 0 (true = active, false = inactive)
        // New files store a SignalRestrictionMode (uint8_t) at the same key
        if (task == SerializationTask::Load) {
            auto findResult = auxiliaries.find(Id_SignalRestrictionGenome_Mode);
            if (findResult != auxiliaries.end()) {
                auto& variantData = findResult->second;
                if (std::holds_alternative<bool>(variantData)) {
                    // Old file format: bool (true = active, false = inactive)
                    data._mode = std::get<bool>(variantData) ? SignalRestrictionMode_Active : SignalRestrictionMode_Inactive;
                } else if (std::holds_alternative<uint8_t>(variantData)) {
                    // New file format: SignalRestrictionMode (uint8_t)
                    auto modeValue = std::get<uint8_t>(variantData);
                    data._mode = (modeValue < SignalRestrictionMode_Count) ? modeValue : defaultObject._mode;
                } else {
                    data._mode = defaultObject._mode;
                }
            } else {
                data._mode = defaultObject._mode;
            }
        } else {
            // Save using the new uint8_t format
            auxiliaries.emplace(Id_SignalRestrictionGenome_Mode, data._mode);
        }
        
        loadSave(task, auxiliaries, Id_SignalRestrictionGenome_BaseAngle, data._baseAngle, defaultObject._baseAngle);
        loadSave(task, auxiliaries, Id_SignalRestrictionGenome_OpeneningAngle, data._openingAngle, defaultObject._openingAngle);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRestrictionGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NodeDescription& data)
    {
        NodeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Node_ReferenceAngle, data._referenceAngle, defaultObject._referenceAngle);
        loadSave(task, auxiliaries, Id_Node_Color, data._color, defaultObject._color);
        loadSave(task, auxiliaries, Id_Node_NumAdditionalConnections, data._numAdditionalConnections, defaultObject._numAdditionalConnections);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._neuralNetwork, data._cellType, data._signalRestriction);
    }
    SPLIT_SERIALIZATION(NodeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeneDescription& data)
    {
        GeneDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Gene_Name, data._name, defaultObject._name);
        loadSave(task, auxiliaries, Id_Gene_Shape, data._shape, defaultObject._shape);
        loadSave(task, auxiliaries, Id_Gene_NumBranches, data._numBranches, defaultObject._numBranches);
        loadSave(task, auxiliaries, Id_Gene_Separation, data._separation, defaultObject._separation);
        loadSave(task, auxiliaries, Id_Gene_AngleAlignment, data._angleAlignment, defaultObject._angleAlignment);
        loadSave(task, auxiliaries, Id_Gene_Stiffness, data._stiffness, defaultObject._stiffness);
        loadSave(task, auxiliaries, Id_Gene_ConnectionDistance, data._connectionDistance, defaultObject._connectionDistance);
        loadSave(task, auxiliaries, Id_Gene_NumRepetitions, data._numConcatenations, defaultObject._numConcatenations);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._nodes);
    }
    SPLIT_SERIALIZATION(GeneDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GenomeDescription& data)
    {
        GenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Genome_Id, data._id, defaultObject._id);
        loadSave(task, auxiliaries, Id_Genome_Name, data._name, defaultObject._name);
        loadSave(task, auxiliaries, Id_Genome_FrontAngle, data._frontAngle, defaultObject._frontAngle);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._genes);
    }
    SPLIT_SERIALIZATION(GenomeDescription)
}

/************************************************************************/
/* Objects data                                                         */
/************************************************************************/
namespace
{
    auto constexpr Id_Particle_Id = 0;
    auto constexpr Id_Particle_Pos = 1;
    auto constexpr Id_Particle_Vel = 2;
    auto constexpr Id_Particle_Energy = 3;
    auto constexpr Id_Particle_Color = 4;

    auto constexpr Id_Creature_Id = 0;
    auto constexpr Id_Creature_AncestorId = 1;
    auto constexpr Id_Creature_Generation = 2;
    auto constexpr Id_Creature_LineageId = 3;
    auto constexpr Id_Creature_NumCells = 4;
    auto constexpr Id_Creature_FrontAngleId = 5;
    auto constexpr Id_Creature_GenomeId = 6;

    auto constexpr Id_Cell_Id = 0;
    auto constexpr Id_Cell_Energy = 1;
    auto constexpr Id_Cell_RawEnergy = 20;
    auto constexpr Id_Cell_Pos = 2;
    auto constexpr Id_Cell_Vel = 3;
    auto constexpr Id_Cell_Stiffness = 4;
    auto constexpr Id_Cell_Color = 5;
    auto constexpr Id_Cell_Fixed = 6;
    auto constexpr Id_Cell_Age = 7;
    auto constexpr Id_Cell_CellState = 8;
    auto constexpr Id_Cell_ActivationTime = 9;
    auto constexpr Id_Cell_CellTriggered = 10;
    auto constexpr Id_Cell_NodeIndex = 12;
    auto constexpr Id_Cell_ParentNodeIndex = 13;
    auto constexpr Id_Cell_GeneIndex = 14;
    auto constexpr Id_Cell_SignalState = 15;
    auto constexpr Id_Cell_AngleToFront = 16;
    auto constexpr Id_Cell_Sticky = 17;
    auto constexpr Id_Cell_FrontAngleId = 18;
    auto constexpr Id_Cell_IsFrontAngleRefCell = 19;

    auto constexpr Id_Signal_Channels = 0;
    auto constexpr Id_Signal_NumTimesSent = 1;

    auto constexpr Id_SignalRestriction_Mode = 0;  // Replaces Active (0=inactive, 1=active, 2=conditional)
    auto constexpr Id_SignalRestriction_BaseAngle = 1;
    auto constexpr Id_SignalRestriction_OpeningAngle = 2;

    auto constexpr Id_Connection_CellId = 0;
    auto constexpr Id_Connection_Distance = 1;
    auto constexpr Id_Connection_AngleFromPrevious = 2;

    auto constexpr Id_NeuralNetwork_Weights = 0;
    auto constexpr Id_NeuralNetwork_Biases = 1;
    auto constexpr Id_NeuralNetwork_ActivationFunctions = 2;

    auto constexpr Id_Constructor_AutoTriggerInterval = 0;
    auto constexpr Id_Constructor_ConstructionActivationTime = 1;
    auto constexpr Id_Constructor_GeneIndex = 2;
    auto constexpr Id_Constructor_CurrentNodeIndex = 3;
    auto constexpr Id_Constructor_CurrentConcatenation = 4;
    auto constexpr Id_Constructor_LastConstructedCellId = 5;
    auto constexpr Id_Constructor_CurrentBranch = 6;
    auto constexpr Id_Constructor_ConstructionAngle = 7;
    auto constexpr Id_Constructor_ProvideEnergy = 8;

    auto constexpr Id_Defender_Mode = 0;

    auto constexpr Id_Muscle_LastMovementX = 4;
    auto constexpr Id_Muscle_LastMovementY = 5;

    auto constexpr Id_MuscleMode_AutoBending_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleMode_AutoBending_ForwardBackwardRatio = 6;
    auto constexpr Id_MuscleMode_AutoBending_InitialAngle = 7;
    auto constexpr Id_MuscleMode_AutoBending_Forward = 8;
    auto constexpr Id_MuscleMode_AutoBending_Activation = 9;
    auto constexpr Id_MuscleMode_AutoBending_ActivationCountdown = 10;
    auto constexpr Id_MuscleMode_AutoBending_ImpulseAlreadyApplied = 12;

    auto constexpr Id_MuscleMode_ManualBending_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleMode_ManualBending_ForwardBackwardRatio = 1;
    auto constexpr Id_MuscleMode_ManualBending_InitialAngle = 2;
    auto constexpr Id_MuscleMode_ManualBending_ImpulseAlreadyApplied = 4;
    auto constexpr Id_MuscleMode_ManualBending_LastAngleDelta = 5;

    auto constexpr Id_MuscleMode_AngleBending_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleMode_AngleBending_AttractionRepulsionRatio = 1;
    auto constexpr Id_MuscleMode_AngleBending_InitialAngle = 2;

    auto constexpr Id_MuscleMode_AutoCrawling_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleMode_AutoCrawling_ForwardBackwardRatio = 1;
    auto constexpr Id_MuscleMode_AutoCrawling_InitialDistance = 2;
    auto constexpr Id_MuscleMode_AutoCrawling_Forward = 3;
    auto constexpr Id_MuscleMode_AutoCrawling_Activation = 4;
    auto constexpr Id_MuscleMode_AutoCrawling_ActivationCountdown = 5;
    auto constexpr Id_MuscleMode_AutoCrawling_LastActualDistance = 6;
    auto constexpr Id_MuscleMode_AutoCrawling_ImpulseAlreadyApplied = 7;

    auto constexpr Id_MuscleMode_ManualCrawling_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleMode_ManualCrawling_ForwardBackwardRatio = 1;
    auto constexpr Id_MuscleMode_ManualCrawling_InitialDistance = 2;
    auto constexpr Id_MuscleMode_ManualCrawling_LastActualDistance = 3;
    auto constexpr Id_MuscleMode_ManualCrawling_LastDistanceDelta = 4;
    auto constexpr Id_MuscleMode_ManualCrawling_ImpulseAlreadyApplied = 5;

    auto constexpr Id_Injector_GeneIndex = 0;

    auto constexpr Id_Generator_AutoTriggerInterval = 0;
    auto constexpr Id_Generator_PulseType = 1;
    auto constexpr Id_Generator_AlternationMode = 2;
    auto constexpr Id_Generator_NumPulses = 3;

    auto constexpr Id_AttackerMode_FreeCell_RestrictToColor = 0;

    auto constexpr Id_AttackerMode_Creature_MinNumCells = 0;
    auto constexpr Id_AttackerMode_Creature_MaxNumCells = 1;
    auto constexpr Id_AttackerMode_Creature_RestrictToColor = 2;
    auto constexpr Id_AttackerMode_Creature_RestrictToLineage = 3;

    auto constexpr Id_Sensor_MinRange = 0;
    auto constexpr Id_Sensor_MaxRange = 1;
    auto constexpr Id_Sensor_AutoTriggerInterval = 2;

    auto constexpr Id_SensorMode_DetectEnergy_MinDensity = 0;

    auto constexpr Id_SensorMode_DetectFreeCell_MinDensity = 0;
    auto constexpr Id_SensorMode_DetectFreeCell_RestrictToColor = 1;

    auto constexpr Id_SensorMode_SensorLastMatch_CreatureId = 0;
    auto constexpr Id_SensorMode_SensorLastMatch_Pos = 1;

    auto constexpr Id_SensorMode_DetectCreature_MinNumCells = 0;
    auto constexpr Id_SensorMode_DetectCreature_MaxNumCells = 1;
    auto constexpr Id_SensorMode_DetectCreature_RestrictToColor = 2;
    auto constexpr Id_SensorMode_DetectCreature_RestrictToLineage = 3;

    auto constexpr Id_Depot_storageLimit = 1;
    auto constexpr Id_Depot_StoredUsableEnergy = 2;

    auto constexpr Id_ReconnectorMode_FreeCell_RestrictToColor = 0;

    auto constexpr Id_ReconnectorMode_Creature_MinNumCells = 0;
    auto constexpr Id_ReconnectorMode_Creature_MaxNumCells = 1;
    auto constexpr Id_ReconnectorMode_Creature_RestrictToColor = 2;
    auto constexpr Id_ReconnectorMode_Creature_RestrictToLineage = 3;

    auto constexpr Id_Detonator_State = 0;
    auto constexpr Id_Detonator_Countdown = 1;

    auto constexpr Id_Digestor_RawEnergyConductivity = 0;

    auto constexpr Id_SignalEntry_Channels = 0;

    auto constexpr Id_SignalDelay_Delay = 0;
    auto constexpr Id_SignalDelay_NumMemoryEntriesInitialized = 1;
    auto constexpr Id_SignalDelay_RingBufferIndex = 2;

    auto constexpr Id_SignalRecorder_ReadOnly = 0;
    auto constexpr Id_SignalRecorder_State = 1;
    auto constexpr Id_SignalRecorder_NumSavedSignalEntries = 2;
    auto constexpr Id_SignalRecorder_NumReadSignalEntries = 3;

    auto constexpr Id_SignalStorage_ReadOnly = 0;

    auto constexpr Id_SignalIntegrator_NewSignalWeight = 0;

    auto constexpr Id_Memory_ChannelBitMask = 0;

    auto constexpr Id_Sender_Range = 0;
    auto constexpr Id_Sender_MaxTimesSent = 1;

    auto constexpr Id_Receiver_ChannelBitMask = 0;
    auto constexpr Id_Receiver_RestrictToColor = 1;
    auto constexpr Id_Receiver_RestrictToLineage = 2;
}

namespace cereal
{


    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConnectionDescription& data)
    {
        ConnectionDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Connection_CellId, data._cellId, defaultObject._cellId);
        loadSave(task, auxiliaries, Id_Connection_Distance, data._distance, defaultObject._distance);
        loadSave(task, auxiliaries, Id_Connection_AngleFromPrevious, data._angleFromPrevious, defaultObject._angleFromPrevious);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ConnectionDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDescription& data)
    {
        SignalDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Signal_Channels, data._channels, defaultObject._channels);
        loadSave(task, auxiliaries, Id_Signal_NumTimesSent, data._numTimesSent, defaultObject._numTimesSent);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRestrictionDescription& data)
    {
        SignalRestrictionDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        
        // Backward compatibility: old files stored a bool at key 0 (true = active, false = inactive)
        // New files store a SignalRestrictionMode (uint8_t) at the same key
        if (task == SerializationTask::Load) {
            auto findResult = auxiliaries.find(Id_SignalRestriction_Mode);
            if (findResult != auxiliaries.end()) {
                auto& variantData = findResult->second;
                if (std::holds_alternative<bool>(variantData)) {
                    // Old file format: bool (true = active, false = inactive)
                    data._mode = std::get<bool>(variantData) ? SignalRestrictionMode_Active : SignalRestrictionMode_Inactive;
                } else if (std::holds_alternative<uint8_t>(variantData)) {
                    // New file format: SignalRestrictionMode (uint8_t)
                    auto modeValue = std::get<uint8_t>(variantData);
                    data._mode = (modeValue < SignalRestrictionMode_Count) ? modeValue : defaultObject._mode;
                } else {
                    data._mode = defaultObject._mode;
                }
            } else {
                data._mode = defaultObject._mode;
            }
        } else {
            // Save using the new uint8_t format
            auxiliaries.emplace(Id_SignalRestriction_Mode, data._mode);
        }
        
        loadSave(task, auxiliaries, Id_SignalRestriction_BaseAngle, data._baseAngle, defaultObject._baseAngle);
        loadSave(task, auxiliaries, Id_SignalRestriction_OpeningAngle, data._openingAngle, defaultObject._openingAngle);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRestrictionDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetworkDescription& data)
    {
        NeuralNetworkDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_NeuralNetwork_Weights, data._weights, defaultObject._weights);
        loadSave(task, auxiliaries, Id_NeuralNetwork_Biases, data._biases, defaultObject._biases);
        loadSave(task, auxiliaries, Id_NeuralNetwork_ActivationFunctions, data._activationFunctions, defaultObject._activationFunctions);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(NeuralNetworkDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, BaseDescription& data)
    {
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(BaseDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DepotDescription& data)
    {
        DepotDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Depot_storageLimit, data._storageLimit, defaultObject._storageLimit);
        loadSave(task, auxiliaries, Id_Depot_StoredUsableEnergy, data._storedUsableEnergy, defaultObject._storedUsableEnergy);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DepotDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorDescription& data)
    {
        ConstructorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Constructor_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        loadSave(task, auxiliaries, Id_Constructor_ConstructionActivationTime, data._constructionActivationTime, defaultObject._constructionActivationTime);
        loadSave(task, auxiliaries, Id_Constructor_ConstructionAngle, data._constructionAngle, defaultObject._constructionAngle);
        loadSave(task, auxiliaries, Id_Constructor_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        loadSave(task, auxiliaries, Id_Constructor_LastConstructedCellId, data._lastConstructedCellId, defaultObject._lastConstructedCellId);
        loadSave(task, auxiliaries, Id_Constructor_CurrentNodeIndex, data._currentNodeIndex, defaultObject._currentNodeIndex);
        loadSave(task, auxiliaries, Id_Constructor_CurrentConcatenation, data._currentConcatenation, defaultObject._currentConcatenation);
        loadSave(task, auxiliaries, Id_Constructor_CurrentBranch, data._currentBranch, defaultObject._currentBranch);
        loadSave(task, auxiliaries, Id_Constructor_ProvideEnergy, data._provideEnergy, defaultObject._provideEnergy);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ConstructorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TelemetryDescription& data)
    {
        //TelemetryDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(TelemetryDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectEnergyDescription& data)
    {
        DetectEnergyDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorMode_DetectEnergy_MinDensity, data._minDensity, defaultObject._minDensity);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectEnergyDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectStructureDescription& data)
    {
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectStructureDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectFreeCellDescription& data)
    {
        DetectFreeCellDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorMode_DetectFreeCell_MinDensity, data._minDensity, defaultObject._minDensity);
        loadSave(task, auxiliaries, Id_SensorMode_DetectFreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectFreeCellDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectCreatureDescription& data)
    {
        DetectCreatureDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorMode_DetectCreature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_SensorMode_DetectCreature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_SensorMode_DetectCreature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_SensorMode_DetectCreature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectCreatureDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorLastMatchDescription& data)
    {
        SensorLastMatchDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorMode_SensorLastMatch_CreatureId, data._creatureId, defaultObject._creatureId);
        loadSave(task, auxiliaries, Id_SensorMode_SensorLastMatch_Pos, data._pos, defaultObject._pos);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SensorLastMatchDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorDescription& data)
    {
        SensorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Sensor_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        loadSave(task, auxiliaries, Id_Sensor_MinRange, data._minRange, defaultObject._minRange);
        loadSave(task, auxiliaries, Id_Sensor_MaxRange, data._maxRange, defaultObject._maxRange);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode, data._lastMatch);
    }
    SPLIT_SERIALIZATION(SensorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeneratorDescription& data)
    {
        GeneratorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Generator_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        loadSave(task, auxiliaries, Id_Generator_PulseType, data._pulseType, defaultObject._pulseType);
        loadSave(task, auxiliaries, Id_Generator_AlternationMode, data._alternationInterval, defaultObject._alternationInterval);
        loadSave(task, auxiliaries, Id_Generator_NumPulses, data._numPulses, defaultObject._numPulses);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(GeneratorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackFreeCellDescription& data)
    {
        AttackFreeCellDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_AttackerMode_FreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackFreeCellDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackCreatureDescription& data)
    {
        AttackCreatureDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_AttackerMode_Creature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_AttackerMode_Creature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_AttackerMode_Creature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_AttackerMode_Creature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackCreatureDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerDescription& data)
    {
        AttackerDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(AttackerDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorDescription& data)
    {
        InjectorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Injector_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(InjectorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoBendingDescription& data)
    {
        AutoBendingDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_InitialAngle, data._initialAngle, defaultObject._initialAngle);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_Forward, data._forward, defaultObject._forward);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_Activation, data._activation, defaultObject._activation);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_ActivationCountdown, data._activationCountdown, defaultObject._activationCountdown);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_ImpulseAlreadyApplied, data._impulseAlreadyApplied, defaultObject._impulseAlreadyApplied);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AutoBendingDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualBendingDescription& data)
    {
        ManualBendingDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualBending_InitialAngle, data._initialAngle, defaultObject._initialAngle);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualBending_LastAngleDelta, data._lastAngleDelta, defaultObject._lastAngleDelta);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualBending_ImpulseAlreadyApplied, data._impulseAlreadyApplied, defaultObject._impulseAlreadyApplied);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ManualBendingDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AngleBendingDescription& data)
    {
        AngleBendingDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_AngleBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(
            task, auxiliaries, Id_MuscleMode_AngleBending_AttractionRepulsionRatio, data._attractionRepulsionRatio, defaultObject._attractionRepulsionRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_AngleBending_InitialAngle, data._initialAngle, defaultObject._initialAngle);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AngleBendingDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoCrawlingDescription& data)
    {
        AutoCrawlingDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_MaxAngleDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_InitialDistance, data._initialDistance, defaultObject._initialDistance);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_LastActualDistance, data._lastActualDistance, defaultObject._lastActualDistance);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_Forward, data._forward, defaultObject._forward);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_Activation, data._activation, defaultObject._activation);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_ActivationCountdown, data._activationCountdown, defaultObject._activationCountdown);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_ImpulseAlreadyApplied, data._impulseAlreadyApplied, defaultObject._impulseAlreadyApplied);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AutoCrawlingDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualCrawlingDescription& data)
    {
        ManualCrawlingDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_MaxAngleDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_InitialDistance, data._initialDistance, defaultObject._initialDistance);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_LastActualDistance, data._lastActualDistance, defaultObject._lastActualDistance);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_LastDistanceDelta, data._lastDistanceDelta, defaultObject._lastDistanceDelta);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_ImpulseAlreadyApplied, data._impulseAlreadyApplied, defaultObject._impulseAlreadyApplied);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ManualCrawlingDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DirectMovementDescription& data)
    {
        DirectMovementDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DirectMovementDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleDescription& data)
    {
        MuscleDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Muscle_LastMovementX, data._lastMovementX, defaultObject._lastMovementX);
        loadSave(task, auxiliaries, Id_Muscle_LastMovementY, data._lastMovementY, defaultObject._lastMovementY);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(MuscleDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderDescription& data)
    {
        DefenderDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Defender_Mode, data._mode, defaultObject._mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DefenderDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectStructureDescription& data)
    {
        ReconnectStructureDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectStructureDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectFreeCellDescription& data)
    {
        ReconnectFreeCellDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReconnectorMode_FreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectFreeCellDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectCreatureDescription& data)
    {
        ReconnectCreatureDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReconnectorMode_Creature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_ReconnectorMode_Creature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_ReconnectorMode_Creature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_ReconnectorMode_Creature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectCreatureDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorDescription& data)
    {
        ReconnectorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(ReconnectorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorDescription& data)
    {
        DetonatorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Detonator_State, data._state, defaultObject._state);
        loadSave(task, auxiliaries, Id_Detonator_Countdown, data._countdown, defaultObject._countdown);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetonatorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DigestorDescription& data)
    {
        DigestorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Digestor_RawEnergyConductivity, data._rawEnergyConductivity, defaultObject._rawEnergyConductivity);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DigestorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDelayDescription& data)
    {
        SignalDelayDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalDelay_Delay, data._delay, defaultObject._delay);
        loadSave(task, auxiliaries, Id_SignalDelay_NumMemoryEntriesInitialized, data._numSignalEntriesInitialized, defaultObject._numSignalEntriesInitialized);
        loadSave(task, auxiliaries, Id_SignalDelay_RingBufferIndex, data._ringBufferIndex, defaultObject._ringBufferIndex);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalDelayDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRecorderDescription& data)
    {
        SignalRecorderDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalRecorder_ReadOnly, data._readOnly, defaultObject._readOnly);
        loadSave(task, auxiliaries, Id_SignalRecorder_State, data._state, defaultObject._state);
        loadSave(task, auxiliaries, Id_SignalRecorder_NumSavedSignalEntries, data._numWrittenSignalEntries, defaultObject._numWrittenSignalEntries);
        loadSave(task, auxiliaries, Id_SignalRecorder_NumReadSignalEntries, data._numReadSignalEntries, defaultObject._numReadSignalEntries);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRecorderDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalStorageDescription& data)
    {
        SignalStorageDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalStorage_ReadOnly, data._readOnly, defaultObject._readOnly);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalStorageDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalIntegratorDescription& data)
    {
        SignalIntegratorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalIntegrator_NewSignalWeight, data._newSignalWeight, defaultObject._newSignalWeight);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalIntegratorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalEntryDescription& data)
    {
        SignalEntryDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalEntry_Channels, data._channels, defaultObject._channels);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalEntryDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MemoryDescription& data)
    {
        MemoryDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Memory_ChannelBitMask, data._channelBitMask, defaultObject._channelBitMask);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
        ar(data._signalEntries);
    }
    SPLIT_SERIALIZATION(MemoryDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SenderDescription& data)
    {
        SenderDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Sender_Range, data._range, defaultObject._range);
        loadSave(task, auxiliaries, Id_Sender_MaxTimesSent, data._maxTimesSent, defaultObject._maxTimesSent);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SenderDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReceiverDescription& data)
    {
        ReceiverDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Receiver_ChannelBitMask, data._channelBitMask, defaultObject._channelBitMask);
        loadSave(task, auxiliaries, Id_Receiver_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_Receiver_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReceiverDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CommunicatorDescription& data)
    {
        CommunicatorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(CommunicatorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellDescription& data)
    {
        CellDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Cell_Id, data._id, defaultObject._id);
        loadSave(task, auxiliaries, Id_Cell_Energy, data._usableEnergy, defaultObject._usableEnergy);
        loadSave(task, auxiliaries, Id_Cell_RawEnergy, data._rawEnergy, defaultObject._rawEnergy);
        loadSave(task, auxiliaries, Id_Cell_Pos, data._pos, defaultObject._pos);
        loadSave(task, auxiliaries, Id_Cell_Vel, data._vel, defaultObject._vel);
        loadSave(task, auxiliaries, Id_Cell_Stiffness, data._stiffness, defaultObject._stiffness);
        loadSave(task, auxiliaries, Id_Cell_Color, data._color, defaultObject._color);
        loadSave(task, auxiliaries, Id_Cell_AngleToFront, data._frontAngle, defaultObject._frontAngle);
        loadSave(task, auxiliaries, Id_Cell_Fixed, data._fixed, defaultObject._fixed);
        loadSave(task, auxiliaries, Id_Cell_Sticky, data._sticky, defaultObject._sticky);
        loadSave(task, auxiliaries, Id_Cell_Age, data._age, defaultObject._age);
        loadSave(task, auxiliaries, Id_Cell_CellState, data._cellState, defaultObject._cellState);
        loadSave(task, auxiliaries, Id_Cell_ActivationTime, data._activationTime, defaultObject._activationTime);
        loadSave(task, auxiliaries, Id_Cell_CellTriggered, data._cellTriggered, defaultObject._cellTriggered);
        loadSave(task, auxiliaries, Id_Cell_NodeIndex, data._nodeIndex, defaultObject._nodeIndex);
        loadSave(task, auxiliaries, Id_Cell_ParentNodeIndex, data._parentNodeIndex, defaultObject._parentNodeIndex);
        loadSave(task, auxiliaries, Id_Cell_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        loadSave(task, auxiliaries, Id_Cell_SignalState, data._signalState, defaultObject._signalState);
        loadSave(task, auxiliaries, Id_Cell_FrontAngleId, data._frontAngleId, defaultObject._frontAngleId);
        loadSave(task, auxiliaries, Id_Cell_IsFrontAngleRefCell, data._headCell, defaultObject._headCell);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._connections, data._cellType, data._signal, data._signalRestriction, data._neuralNetwork);
    }
    SPLIT_SERIALIZATION(CellDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CreatureDescription& data)
    {
        CreatureDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Creature_Id, data._id, defaultObject._id);
        loadSave(task, auxiliaries, Id_Creature_AncestorId, data._ancestorId, defaultObject._ancestorId);
        loadSave(task, auxiliaries, Id_Creature_Generation, data._generation, defaultObject._generation);
        loadSave(task, auxiliaries, Id_Creature_LineageId, data._lineageId, defaultObject._lineageId);
        loadSave(task, auxiliaries, Id_Creature_NumCells, data._numCells, defaultObject._numCells);
        loadSave(task, auxiliaries, Id_Creature_FrontAngleId, data._frontAngleId, defaultObject._frontAngleId);
        loadSave(task, auxiliaries, Id_Creature_GenomeId, data._genomeId, defaultObject._genomeId);

        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._cells);
    }
    SPLIT_SERIALIZATION(CreatureDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ParticleDescription& data)
    {
        ParticleDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Particle_Id, data._id, defaultObject._id);
        loadSave(task, auxiliaries, Id_Particle_Pos, data._pos, defaultObject._pos);
        loadSave(task, auxiliaries, Id_Particle_Vel, data._vel, defaultObject._vel);
        loadSave(task, auxiliaries, Id_Particle_Energy, data._energy, defaultObject._energy);
        loadSave(task, auxiliaries, Id_Particle_Color, data._color, defaultObject._color);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ParticleDescription)

    template <class Archive>
    void serialize(Archive& ar, Description& description)
    {
        ar(description._cells, description._particles, description._creatures, description._genomes);
    }
}

bool SerializerService::serializeSimulationToFiles(std::filesystem::path const& filename, DeserializedSimulation const& data) const
{
    try {
        log(Priority::Important, "save simulation to " + filename.string());
        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));
        std::filesystem::path statisticsFilename(filename);
        statisticsFilename.replace_extension(std::filesystem::path(".statistics.csv"));

        {
            zstr::ofstream stream(filename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeDescription(data.mainData, stream);
        }
        {
            std::ofstream stream(settingsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeSettings(data.auxiliaryData, stream);
        }
        {
            std::ofstream stream(statisticsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeStatistics(data.statistics, stream);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeSimulationFromFiles(DeserializedSimulation& data, std::filesystem::path const& filename) const
{
    try {
        log(Priority::Important, "load simulation from " + filename.string());
        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));
        std::filesystem::path statisticsFilename(filename);
        statisticsFilename.replace_extension(std::filesystem::path(".statistics.csv"));

        if (!deserializeDescription(data.mainData, filename)) {
            return false;
        }
        {
            std::ifstream stream(settingsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeSettings(data.auxiliaryData, stream);
        }
        {
            std::ifstream stream(statisticsFilename.string(), std::ios::binary);
            if (!stream) {
                return true;
            }
            deserializeStatistics(data.statistics, stream);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deleteSimulation(std::filesystem::path const& filename) const
{
    try {
        log(Priority::Important, "delete simulation " + filename.string());
        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));
        std::filesystem::path statisticsFilename(filename);
        statisticsFilename.replace_extension(std::filesystem::path(".statistics.csv"));

        if (!std::filesystem::remove(filename)) {
            return false;
        }
        if (!std::filesystem::remove(settingsFilename)) {
            return false;
        }
        if (!std::filesystem::remove(statisticsFilename)) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeSimulationToStrings(SerializedSimulation& output, DeserializedSimulation const& input) const
{
    try {
        {
            std::stringstream stdStream;
            zstr::ostream stream(stdStream, std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeDescription(input.mainData, stream);
            stream.flush();
            output.mainData = stdStream.str();
        }
        {
            std::stringstream stream;
            serializeSettings(input.auxiliaryData, stream);
            output.auxiliaryData = stream.str();
        }
        {
            std::stringstream stream;
            serializeStatistics(input.statistics, stream);
            output.statistics = stream.str();
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeSimulationFromStrings(DeserializedSimulation& output, SerializedSimulation const& input) const
{
    try {
        {
            std::stringstream stdStream(input.mainData);
            zstr::istream stream(stdStream, std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeDescription(output.mainData, stream);
        }
        {
            std::stringstream stream(input.auxiliaryData);
            deserializeSettings(output.auxiliaryData, stream);
        }
        {
            std::stringstream stream(input.statistics);
            deserializeStatistics(output.statistics, stream);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeGenomeToFile(std::filesystem::path const& filename, GenomeDescription const& genome) const
{
    try {
        log(Priority::Important, "save genome to " + filename.string());
        //wrap constructor cell around genome
        Description description;
        if (!wrapGenome(description, genome)) {
            return false;
        }

        zstr::ofstream stream(filename.string(), std::ios::binary);
        if (!stream) {
            return false;
        }
        serializeDescription(description, stream);

        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeGenomeFromFile(GenomeDescription& genome, std::filesystem::path const& filename) const
{
    try {
        log(Priority::Important, "load genome from " + filename.string());
        Description description;
        if (!deserializeDescription(description, filename)) {
            return false;
        }
        if (!unwrapGenome(genome, description)) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeGenomeToString(std::string& output, std::vector<uint8_t> const& input) const
{
    try {
        std::stringstream stdStream;
        zstr::ostream stream(stdStream, std::ios::binary);
        if (!stream) {
            return false;
        }

        Description description;
        //if (!wrapGenome(description, input)) {
        //    return false;
        //}

        serializeDescription(description, stream);
        stream.flush();
        output = stdStream.str();
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeGenomeFromString(std::vector<uint8_t>& output, std::string const& input) const
{
    try {
        std::stringstream stdStream(input);
        zstr::istream stream(stdStream, std::ios::binary);
        if (!stream) {
            return false;
        }

        Description description;
        deserializeDescription(description, stream);

        //if (!unwrapGenome(output, description)) {
        //    return false;
        //}
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeSimulationParametersToFile(std::filesystem::path const& filename, SimulationParameters const& parameters) const
{
    try {
        log(Priority::Important, "save simulation parameters to " + filename.string());
        std::ofstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        serializeSimulationParameters(parameters, stream);
        stream.close();
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::filesystem::path const& filename) const
{
    try {
        log(Priority::Important, "load simulation parameters from " + filename.string());
        std::ifstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        deserializeSimulationParameters(parameters, stream);
        stream.close();
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeStatisticsToFile(std::filesystem::path const& filename, StatisticsHistoryData const& statistics) const
{
    try {
        log(Priority::Important, "save statistics history to " + filename.string());
        std::ofstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        serializeStatistics(statistics, stream);
        stream.close();
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeContentToFile(std::filesystem::path const& filename, Description const& content) const
{
    try {
        zstr::ofstream fileStream(filename.string(), std::ios::binary);
        if (!fileStream) {
            return false;
        }
        serializeDescription(content, fileStream);

        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeContentFromFile(Description& content, std::filesystem::path const& filename) const
{
    try {
        if (!deserializeDescription(content, filename)) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

void SerializerService::serializeDescription(Description const& description, std::ostream& stream) const
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(Const::ProgramVersion);
    archive(description);
}

bool SerializerService::deserializeDescription(Description& description, std::filesystem::path const& filename) const
{
    zstr::ifstream stream(filename.string(), std::ios::binary);
    if (!stream) {
        return false;
    }
    deserializeDescription(description, stream);
    return true;
}

void SerializerService::deserializeDescription(Description& description, std::istream& stream) const
{
    cereal::PortableBinaryInputArchive archive(stream);
    std::string version;
    archive(version);

    if (!VersionParserService::get().isVersionValid(version)) {
        throw std::runtime_error("No version detected.");
    }
    if (VersionParserService::get().isVersionOutdated(version)) {
        throw std::runtime_error("Version not supported.");
    }
    archive(description);
}

void SerializerService::serializeSettings(SettingsForSerialization const& settings, std::ostream& stream) const
{
    boost::property_tree::json_parser::write_json(stream, SettingsParserService::get().encodeSettings(settings));
}

void SerializerService::deserializeSettings(SettingsForSerialization& settings, std::istream& stream) const
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    settings = SettingsParserService::get().decodeSettings(tree);
}

void SerializerService::serializeSimulationParameters(SimulationParameters const& parameters, std::ostream& stream) const
{
    boost::property_tree::json_parser::write_json(stream, SettingsParserService::get().encodeSimulationParameters(parameters));
}

void SerializerService::deserializeSimulationParameters(SimulationParameters& parameters, std::istream& stream) const
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    parameters = SettingsParserService::get().decodeSimulationParameters(tree);
}

namespace
{
    std::string toString(double value)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(9) << value;
        return ss.str();
    }

    struct ColumnDescription
    {
        std::string name;
        bool colorDependent = false;
    };
    std::vector<ColumnDescription> const ColumnDescriptions = {
        {"Time step", false},
        {"Cells", true},
        {"Self-replicators", true},
        {"Viruses", true},
        {"Free cells", true},
        {"Energy particles", true},
        {"Average genome cells", true},
        {"Total energy", true},
        {"Created cells", true},
        {"Attacks", true},
        {"Muscle activities", true},
        {"Depot activities", true},
        {"Defender activities", true},
        {"Injection activities", true},
        {"Completed injections", true},
        {"Generator pulses", true},
        {"Neuron activities", true},
        {"Sensor activities", true},
        {"Sensor matches", true},
        {"Reconnector creations", true},
        {"Reconnector deletions", true},
        {"Detonations", true},
        {"Colonies", true},
        {"Genome complexity average", true},
        {"Genome complexity Maximum", true},
        {"Genome complexity variance", true},
        {"System clock", false}};

    std::variant<DataPoint*, double*> getDataRef(int colIndex, DataPointCollection& dataPoints)
    {
        if (colIndex == 0) {
            return &dataPoints.time;
        } else if (colIndex == 1) {
            return &dataPoints.numCells;
        } else if (colIndex == 2) {
            return &dataPoints.numSelfReplicators;
        } else if (colIndex == 3) {
            return &dataPoints.numViruses;
        } else if (colIndex == 4) {
            return &dataPoints.numFreeCells;
        } else if (colIndex == 5) {
            return &dataPoints.numParticles;
        } else if (colIndex == 6) {
            return &dataPoints.averageGenomeCells;
        } else if (colIndex == 7) {
            return &dataPoints.totalEnergy;
        } else if (colIndex == 8) {
            return &dataPoints.numCreatedCells;
        } else if (colIndex == 9) {
            return &dataPoints.numAttacks;
        } else if (colIndex == 10) {
            return &dataPoints.numMuscleActivities;
        } else if (colIndex == 11) {
            return &dataPoints.numDefenderActivities;
        } else if (colIndex == 12) {
            return &dataPoints.numDepotActivities;
        } else if (colIndex == 13) {
            return &dataPoints.numInjectionActivities;
        } else if (colIndex == 14) {
            return &dataPoints.numCompletedInjections;
        } else if (colIndex == 15) {
            return &dataPoints.numGeneratorPulses;
        } else if (colIndex == 16) {
            return &dataPoints.numNeuronActivities;
        } else if (colIndex == 17) {
            return &dataPoints.numSensorActivities;
        } else if (colIndex == 18) {
            return &dataPoints.numSensorMatches;
        } else if (colIndex == 19) {
            return &dataPoints.numReconnectorCreated;
        } else if (colIndex == 20) {
            return &dataPoints.numReconnectorRemoved;
        } else if (colIndex == 21) {
            return &dataPoints.numDetonations;
        } else if (colIndex == 22) {
            return &dataPoints.numColonies;
        } else if (colIndex == 23) {
            return &dataPoints.averageNumCells;
        } else if (colIndex == 24) {
            return &dataPoints.maxNumCellsOfColonies;
        } else if (colIndex == 25) {
            return &dataPoints.varianceNumCells;
        } else if (colIndex == 26) {
            return &dataPoints.systemClock;
        }
        THROW_NOT_IMPLEMENTED();
    }

    void load(int startIndex, std::vector<std::string>& serializedData, double& value)
    {
        if (startIndex < serializedData.size()) {
            value = std::stod(serializedData.at(startIndex));
        }
    }

    void save(std::vector<std::string>& serializedData, double& value)
    {
        serializedData.emplace_back(toString(value));
    }

    void load(int startIndex, std::vector<std::string>& serializedData, DataPoint& dataPoint)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            auto index = startIndex + i;
            if (index < serializedData.size()) {
                dataPoint.values[i] = std::stod(serializedData.at(index));
            }
        }
        if (startIndex + 7 < serializedData.size()) {
            dataPoint.summedValues = std::stod(serializedData.at(startIndex + 7));
        }
    }

    void save(std::vector<std::string>& serializedData, DataPoint& dataPoint)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            serializedData.emplace_back(toString(dataPoint.values[i]));
        }
        serializedData.emplace_back(toString(dataPoint.summedValues));
    }

    struct ParsedColumnInfo
    {
        std::string name;
        std::optional<int> colIndex;
        int size = 0;
    };
    void load(std::vector<ParsedColumnInfo> const& colInfos, std::vector<std::string>& serializedData, DataPointCollection& dataPoints)
    {
        int startIndex = 0;
        for (auto const& colInfo : colInfos) {
            if (!colInfo.colIndex.has_value()) {
                startIndex += colInfo.size;
                continue;
            }
            auto col = colInfo.colIndex.value();
            auto data = getDataRef(col, dataPoints);
            if (std::holds_alternative<DataPoint*>(data)) {
                load(startIndex, serializedData, *std::get<DataPoint*>(data));
            }
            if (std::holds_alternative<double*>(data)) {
                load(startIndex, serializedData, *std::get<double*>(data));
            }
            startIndex += colInfo.size;
        }
    }

    void save(std::vector<std::string>& serializedData, DataPointCollection& dataPoints)
    {
        int index = 0;
        for (auto const& column : ColumnDescriptions) {
            auto data = getDataRef(index, dataPoints);
            if (std::holds_alternative<DataPoint*>(data)) {
                save(serializedData, *std::get<DataPoint*>(data));
            }
            if (std::holds_alternative<double*>(data)) {
                save(serializedData, *std::get<double*>(data));
            }
            ++index;
        }
    }

    std::string getPrincipalPart(std::string const& colName)
    {
        auto colNameTrimmed = boost::algorithm::trim_left_copy(colName);
        size_t pos = colNameTrimmed.find(" (");
        if (pos != std::string::npos) {
            return colNameTrimmed.substr(0, pos);
        } else {
            return colNameTrimmed;
        }
    }

    std::optional<int> getColumnIndex(std::string const& colName)
    {
        for (auto const& [index, colDescription] : ColumnDescriptions | boost::adaptors::indexed(0)) {
            if (colDescription.name == colName) {
                return toInt(index);
            }
        }
        return std::nullopt;
    }
}

void SerializerService::serializeStatistics(StatisticsHistoryData const& statistics, std::ostream& stream) const
{
    //header row
    auto writeLabelAllColors = [&stream](auto const& name) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (i != 0) {
                stream << ",";
            }
            stream << name << " (color " << i << ")";
        }
        stream << "," << name << " (accumulated)";
    };

    int index = 0;
    for (auto const& [colName, colorDependent] : ColumnDescriptions) {
        if (index != 0) {
            stream << ", ";
        }
        if (!colorDependent) {
            stream << colName;
        } else {
            writeLabelAllColors(colName);
        }
        ++index;
    }
    stream << std::endl;

    //content
    for (auto dataPoints : statistics) {
        std::vector<std::string> entries;
        save(entries, dataPoints);
        stream << boost::join(entries, ",") << std::endl;
    }
}

void SerializerService::deserializeStatistics(StatisticsHistoryData& statistics, std::istream& stream) const
{
    statistics.clear();

    std::vector<std::vector<std::string>> data;

    // header line
    std::string header;
    std::getline(stream, header);

    std::vector<std::string> colNames;
    boost::split(colNames, header, boost::is_any_of(","));

    std::vector<ParsedColumnInfo> colInfos;
    for (auto const& colName : colNames) {
        auto principalPart = getPrincipalPart(colName);

        if (colInfos.empty()) {
            colInfos.emplace_back(principalPart, getColumnIndex(principalPart), 1);
        } else {
            auto& lastColInfo = colInfos.back();
            if (lastColInfo.name == principalPart) {
                ++lastColInfo.size;
            } else {
                colInfos.emplace_back(principalPart, getColumnIndex(principalPart), 1);
            }
        }
    }

    // data lines
    while (std::getline(stream, header)) {
        std::vector<std::string> entries;
        boost::split(entries, header, boost::is_any_of(","));

        DataPointCollection dataPoints;
        load(colInfos, entries, dataPoints);

        statistics.emplace_back(dataPoints);
    }
}

bool SerializerService::wrapGenome(Description& output, GenomeDescription const& input) const
{
    output.clear();
    output._genomes.emplace_back(input);
    output._creatures.emplace_back(CreatureDescription().genomeId(input._id));
    return true;
}

bool SerializerService::unwrapGenome(GenomeDescription& output, Description& input) const
{
    if (input._genomes.size() != 1) {
        return false;
    }
    output = input._genomes.front();
    return true;
}
