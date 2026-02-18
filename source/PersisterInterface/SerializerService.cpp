#include "SerializerService.h"

#include <cmath>
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

#include <EngineInterface/Desc.h>
#include <EngineInterface/GenomeDesc.h>
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
        std::vector<RealVector2D>,
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

    // Specialized overload for std::vector<NeuralNetWeight> - converts to/from std::vector<int8_t> for serialization
    void loadSave(
        SerializationTask task,
        std::unordered_map<int, VariantData>& loadSaveMap,
        int key,
        std::vector<NeuralNetWeight>& value,
        std::vector<NeuralNetWeight> const& defaultValue)
    {
        if (task == SerializationTask::Load) {
            auto findResult = loadSaveMap.find(key);
            if (findResult != loadSaveMap.end()) {
                auto variantData = findResult->second;
                auto& int8Vec = std::get<std::vector<int8_t>>(variantData);
                value.resize(int8Vec.size());
                for (size_t i = 0; i < int8Vec.size(); ++i) {
                    value[i] = NeuralNetWeight::fromRawValue(static_cast<uint8_t>(int8Vec[i]));
                }
            } else {
                value = defaultValue;
            }
        } else {
            std::vector<int8_t> int8Vec(value.size());
            for (size_t i = 0; i < value.size(); ++i) {
                int8Vec[i] = value[i].rawValue;
            }
            loadSaveMap.emplace(key, int8Vec);
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
    auto constexpr Id_Genome_LineageId = 3;
    auto constexpr Id_Genome_ActivationFunctionMutationRate = 4;

    auto constexpr Id_NeuronMutationRate_Probability = 0;
    auto constexpr Id_NeuronMutationRate_Sigma = 1;

    auto constexpr Id_ConnectionMutationRate_Probability = 0;
    auto constexpr Id_ConnectionMutationRate_Sigma = 1;

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

    auto constexpr Id_NeuralNetGenome_Weights = 0;
    auto constexpr Id_NeuralNetGenome_Biases = 1;
    auto constexpr Id_NeuralNetGenome_ActivationFunctions = 2;
    auto constexpr Id_NeuralNetGenome_ConnectionWeights = 3;

    auto constexpr Id_DepotGenome_storageLimit = 0;
    auto constexpr Id_DepotGenome_InitialStoredUsableEnergy = 1;

    auto constexpr Id_DefenderGenome_Mode = 0;

    auto constexpr Id_ConstructorGenome_AutoTriggerInterval = 0;
    auto constexpr Id_ConstructorGenome_GeneIndex = 1;
    auto constexpr Id_ConstructorGenome_ConstructionActivationTime = 2;
    auto constexpr Id_ConstructorGenome_ConstructionAngle = 3;
    auto constexpr Id_ConstructorGenome_ProvideEnergy = 4;

    auto constexpr Id_SensorGenome_AutoTrigger = 0;
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

    auto constexpr Id_GeneratorGenome_Additive = 0;
    auto constexpr Id_GeneratorGenome_ValueOffset = 1;
    auto constexpr Id_GeneratorGenome_TimeOffset = 2;

    auto constexpr Id_GeneratorModeGenome_SquareSignal_Amplitude = 0;
    auto constexpr Id_GeneratorModeGenome_SquareSignal_Period = 1;

    auto constexpr Id_GeneratorModeGenome_SawtoothSignal_Amplitude = 0;
    auto constexpr Id_GeneratorModeGenome_SawtoothSignal_Period = 1;

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

    auto constexpr Id_ReceiverGenome_RestrictToColor = 1;
    auto constexpr Id_ReceiverGenome_RestrictToLineage = 2;
}

namespace cereal
{
    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetGenomeDesc& data)
    {
        NeuralNetGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_NeuralNetGenome_Weights, data._weights, defaultObject._weights);
        loadSave(task, auxiliaries, Id_NeuralNetGenome_Biases, data._biases, defaultObject._biases);
        loadSave(task, auxiliaries, Id_NeuralNetGenome_ActivationFunctions, data._activationFunctions, defaultObject._activationFunctions);
        loadSave(task, auxiliaries, Id_NeuralNetGenome_ConnectionWeights, data._connectionWeights, defaultObject._connectionWeights);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(NeuralNetGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, BaseGenomeDesc& data)
    {
        BaseGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(BaseGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DepotGenomeDesc& data)
    {
        DepotGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DepotGenome_storageLimit, data._storageLimit, defaultObject._storageLimit);
        loadSave(task, auxiliaries, Id_DepotGenome_InitialStoredUsableEnergy, data._initialStoredUsableEnergy, defaultObject._initialStoredUsableEnergy);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DepotGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorGenomeDesc& data)
    {
        ConstructorGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ConstructorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        loadSave(task, auxiliaries, Id_ConstructorGenome_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        loadSave(
            task, auxiliaries, Id_ConstructorGenome_ConstructionActivationTime, data._constructionActivationTime, defaultObject._constructionActivationTime);
        loadSave(task, auxiliaries, Id_ConstructorGenome_ConstructionAngle, data._constructionAngle, defaultObject._constructionAngle);
        loadSave(task, auxiliaries, Id_ConstructorGenome_ProvideEnergy, data._provideEnergy, defaultObject._provideEnergy);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ConstructorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TelemetryGenomeDesc& data)
    {
        //TelemetryGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(TelemetryGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectEnergyGenomeDesc& data)
    {
        DetectEnergyGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectEnergy_MinDensity, data._minDensity, defaultObject._minDensity);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectEnergyGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectStructureGenomeDesc& data)
    {
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectStructureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectFreeCellGenomeDesc& data)
    {
        DetectFreeCellGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectFreeCell_MinDensity, data._minDensity, defaultObject._minDensity);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectFreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectFreeCellGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectCreatureGenomeDesc& data)
    {
        DetectCreatureGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectCreature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectCreature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectCreature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_SensorModeGenome_DetectCreature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectCreatureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorGenomeDesc& data)
    {
        SensorGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorGenome_AutoTrigger, data._autoTrigger, defaultObject._autoTrigger);
        loadSave(task, auxiliaries, Id_SensorGenome_MinRange, data._minRange, defaultObject._minRange);
        loadSave(task, auxiliaries, Id_SensorGenome_MaxRange, data._maxRange, defaultObject._maxRange);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(SensorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SquareSignalGenomeDesc& data)
    {
        SquareSignalGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_GeneratorModeGenome_SquareSignal_Amplitude, data._amplitude, defaultObject._amplitude);
        loadSave(task, auxiliaries, Id_GeneratorModeGenome_SquareSignal_Period, data._period, defaultObject._period);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SquareSignalGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SawtoothSignalGenomeDesc& data)
    {
        SawtoothSignalGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_GeneratorModeGenome_SawtoothSignal_Amplitude, data._amplitude, defaultObject._amplitude);
        loadSave(task, auxiliaries, Id_GeneratorModeGenome_SawtoothSignal_Period, data._period, defaultObject._period);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SawtoothSignalGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeneratorGenomeDesc& data)
    {
        GeneratorGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_GeneratorGenome_Additive, data._additive, defaultObject._additive);
        loadSave(task, auxiliaries, Id_GeneratorGenome_ValueOffset, data._valueOffset, defaultObject._valueOffset);
        loadSave(task, auxiliaries, Id_GeneratorGenome_TimeOffset, data._timeOffset, defaultObject._timeOffset);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(GeneratorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackFreeCellGenomeDesc& data)
    {
        AttackFreeCellGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_AttackerModeGenome_FreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackFreeCellGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackCreatureGenomeDesc& data)
    {
        AttackCreatureGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackCreatureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerGenomeDesc& data)
    {
        AttackerGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(AttackerGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorGenomeDesc& data)
    {
        InjectorGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_InjectorGenome_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(InjectorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoBendingGenomeDesc& data)
    {
        AutoBendingGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_AutoBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_AutoBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AutoBendingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualBendingGenomeDesc& data)
    {
        ManualBendingGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_ManualBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_ManualBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ManualBendingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AngleBendingGenomeDesc& data)
    {
        AngleBendingGenomeDesc defaultObject;
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
    SPLIT_SERIALIZATION(AngleBendingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoCrawlingGenomeDesc& data)
    {
        AutoCrawlingGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_AutoCrawling_MaxDistanceDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_AutoCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AutoCrawlingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualCrawlingGenomeDesc& data)
    {
        ManualCrawlingGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_ManualCrawling_MaxDistanceDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        loadSave(task, auxiliaries, Id_MuscleModeGenome_ManualCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ManualCrawlingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DirectMovementGenomeDesc& data)
    {
        DirectMovementGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DirectMovementGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleGenomeDesc& data)
    {
        MuscleGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(MuscleGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderGenomeDesc& data)
    {
        DefenderGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DefenderGenome_Mode, data._mode, defaultObject._mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DefenderGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectStructureGenomeDesc& data)
    {
        ReconnectStructureGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectStructureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectFreeCellGenomeDesc& data)
    {
        ReconnectFreeCellGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_FreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectFreeCellGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectCreatureGenomeDesc& data)
    {
        ReconnectCreatureGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_Creature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_Creature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_Creature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_ReconnectorModeGenome_Creature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectCreatureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorGenomeDesc& data)
    {
        ReconnectorGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(ReconnectorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorGenomeDesc& data)
    {
        DetonatorGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DetonatorGenome_Countdown, data._countdown, defaultObject._countdown);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetonatorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DigestorGenomeDesc& data)
    {
        DigestorGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DigestorGenome_RawEnergyConductivity, data._rawEnergyConductivity, defaultObject._rawEnergyConductivity);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DigestorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDelayGenomeDesc& data)
    {
        SignalDelayGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalDelayGenome_Delay, data._delay, defaultObject._delay);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalDelayGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRecorderGenomeDesc& data)
    {
        SignalRecorderGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalRecorderGenome_ReadOnly, data._readOnly, defaultObject._readOnly);
        loadSave(task, auxiliaries, Id_SignalRecorderGenome_NumSavedSignalEntries, data._numWrittenSignalEntries, defaultObject._numWrittenSignalEntries);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRecorderGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalStorageGenomeDesc& data)
    {
        SignalStorageGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalStorageGenome_ReadOnly, data._readOnly, defaultObject._readOnly);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalStorageGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalIntegratorGenomeDesc& data)
    {
        SignalIntegratorGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalIntegratorGenome_NewSignalWeight, data._newSignalWeight, defaultObject._newSignalWeight);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalIntegratorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalEntryGenomeDesc& data)
    {
        SignalEntryGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalEntryGenome_Channels, data._channels, defaultObject._channels);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalEntryGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MemoryGenomeDesc& data)
    {
        MemoryGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MemoryGenome_ChannelBitMask, data._channelBitMask, defaultObject._channelBitMask);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
        ar(data._signalEntries);
    }
    SPLIT_SERIALIZATION(MemoryGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SenderGenomeDesc& data)
    {
        SenderGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SenderGenome_Range, data._range, defaultObject._range);
        loadSave(task, auxiliaries, Id_SenderGenome_MaxTimesSent, data._maxTimesSent, defaultObject._maxTimesSent);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SenderGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReceiverGenomeDesc& data)
    {
        ReceiverGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReceiverGenome_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_ReceiverGenome_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReceiverGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CommunicatorGenomeDesc& data)
    {
        CommunicatorGenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(CommunicatorGenomeDesc)

    // Dummy struct for backward compatibility with old files that have SignalRestrictionGenomeDesc
    struct SignalRestrictionGenomeDescLegacy
    {
        uint8_t _mode = 0;
        float _baseAngle = 0.0f;
        float _openingAngle = 0.0f;
    };

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRestrictionGenomeDescLegacy& data)
    {
        SignalRestrictionGenomeDescLegacy defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);

        // For backward compatibility, read any mode format but discard
        if (task == SerializationTask::Load) {
            auto findResult = auxiliaries.find(Id_SignalRestrictionGenome_Mode);
            if (findResult != auxiliaries.end()) {
                auto& variantData = findResult->second;
                if (std::holds_alternative<bool>(variantData)) {
                    data._mode = std::get<bool>(variantData) ? 1 : 0;
                } else if (std::holds_alternative<uint8_t>(variantData)) {
                    data._mode = std::get<uint8_t>(variantData);
                }
            }
        }

        loadSave(task, auxiliaries, Id_SignalRestrictionGenome_BaseAngle, data._baseAngle, defaultObject._baseAngle);
        loadSave(task, auxiliaries, Id_SignalRestrictionGenome_OpeneningAngle, data._openingAngle, defaultObject._openingAngle);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRestrictionGenomeDescLegacy)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NodeDesc& data)
    {
        NodeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Node_ReferenceAngle, data._referenceAngle, defaultObject._referenceAngle);
        loadSave(task, auxiliaries, Id_Node_Color, data._color, defaultObject._color);
        loadSave(task, auxiliaries, Id_Node_NumAdditionalConnections, data._numAdditionalConnections, defaultObject._numAdditionalConnections);
        processLoadSaveMap(task, ar, auxiliaries);

        // For backward compatibility, read the legacy signal restriction and discard
        SignalRestrictionGenomeDescLegacy legacySignalRestriction;
        ar(data._neuralNetwork, data._cellType, data._constructor, legacySignalRestriction);
    }
    SPLIT_SERIALIZATION(NodeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeneDesc& data)
    {
        GeneDesc defaultObject;
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
    SPLIT_SERIALIZATION(GeneDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuronMutationRateDesc& data)
    {
        NeuronMutationRateDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_NeuronMutationRate_Probability, data._probability, defaultObject._probability);
        loadSave(task, auxiliaries, Id_NeuronMutationRate_Sigma, data._sigma, defaultObject._sigma);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(NeuronMutationRateDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConnectionMutationRateDesc& data)
    {
        ConnectionMutationRateDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ConnectionMutationRate_Probability, data._probability, defaultObject._probability);
        loadSave(task, auxiliaries, Id_ConnectionMutationRate_Sigma, data._sigma, defaultObject._sigma);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ConnectionMutationRateDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GenomeDesc& data)
    {
        GenomeDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Genome_Id, data._id, defaultObject._id);
        loadSave(task, auxiliaries, Id_Genome_Name, data._name, defaultObject._name);
        loadSave(task, auxiliaries, Id_Genome_LineageId, data._lineageId, defaultObject._lineageId);
        loadSave(task, auxiliaries, Id_Genome_FrontAngle, data._frontAngle, defaultObject._frontAngle);
        loadSave(
            task, auxiliaries, Id_Genome_ActivationFunctionMutationRate, data._activationFunctionMutationRate, defaultObject._activationFunctionMutationRate);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._genes);
        ar(data._neuronMutationRate1);
        ar(data._neuronMutationRate2);
        ar(data._connectionMutationRate1);
        ar(data._connectionMutationRate2);
    }
    SPLIT_SERIALIZATION(GenomeDesc)
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
    auto constexpr Id_Creature_NumCells = 4;
    auto constexpr Id_Creature_FrontAngleId = 5;
    auto constexpr Id_Creature_GenomeId = 6;
    auto constexpr Id_Creature_HaveMutationsApplied = 7;

    auto constexpr Id_Structure_Energy = 0;

    auto constexpr Id_FreeCell_Energy = 0;
    auto constexpr Id_FreeCell_Age = 1;

    auto constexpr Id_Cell_UsableEnergy = 0;
    auto constexpr Id_Cell_RawEnergy = 1;
    auto constexpr Id_Cell_ReservedEnergy = 17;
    auto constexpr Id_Cell_Age = 2;
    auto constexpr Id_Cell_CellState = 3;
    auto constexpr Id_Cell_ActivationTime = 4;
    auto constexpr Id_Cell_NodeIndex = 6;
    auto constexpr Id_Cell_ParentNodeIndex = 7;
    auto constexpr Id_Cell_GeneIndex = 8;
    auto constexpr Id_Cell_AngleToFront = 10;
    auto constexpr Id_Cell_FrontAngleId = 11;
    auto constexpr Id_Cell_IsFrontAngleRefCell = 12;
    auto constexpr Id_Cell_CreatureId = 13;
    auto constexpr Id_Cell_Event = 14;
    auto constexpr Id_Cell_EventCounter = 15;
    auto constexpr Id_Cell_EventPos = 16;

    auto constexpr Id_Object_Id = 0;
    auto constexpr Id_Object_Pos = 2;
    auto constexpr Id_Object_Vel = 3;
    auto constexpr Id_Object_Stiffness = 4;
    auto constexpr Id_Object_Color = 5;
    auto constexpr Id_Object_Fixed = 6;
    auto constexpr Id_Object_Sticky = 17;

    auto constexpr Id_Signal_Channels = 0;
    auto constexpr Id_Signal_NumTimesSent = 1;

    auto constexpr Id_SignalRestriction_Mode = 0;  // Replaces Active (0=inactive, 1=active, 2=conditional)
    auto constexpr Id_SignalRestriction_BaseAngle = 1;
    auto constexpr Id_SignalRestriction_OpeningAngle = 2;

    auto constexpr Id_Connection_ObjectId = 0;
    auto constexpr Id_Connection_Distance = 1;
    auto constexpr Id_Connection_AngleFromPrevious = 2;

    auto constexpr Id_NeuralNet_Weights = 0;
    auto constexpr Id_NeuralNet_Biases = 1;
    auto constexpr Id_NeuralNet_ActivationFunctions = 2;
    auto constexpr Id_NeuralNet_ConnectionWeights = 3;

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

    auto constexpr Id_MuscleMode_ManualBending_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleMode_ManualBending_ForwardBackwardRatio = 1;
    auto constexpr Id_MuscleMode_ManualBending_InitialAngle = 2;
    auto constexpr Id_MuscleMode_ManualBending_LastAngleDelta = 5;

    auto constexpr Id_MuscleMode_AngleBending_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleMode_AngleBending_AttractionRepulsionRatio = 1;
    auto constexpr Id_MuscleMode_AngleBending_InitialAngle = 2;

    auto constexpr Id_MuscleMode_AutoCrawling_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleMode_AutoCrawling_ForwardBackwardRatio = 1;
    auto constexpr Id_MuscleMode_AutoCrawling_InitialDistance = 2;
    auto constexpr Id_MuscleMode_AutoCrawling_Forward = 3;
    auto constexpr Id_MuscleMode_AutoCrawling_LastActualDistance = 6;

    auto constexpr Id_MuscleMode_ManualCrawling_MaxAngleDeviation = 0;
    auto constexpr Id_MuscleMode_ManualCrawling_ForwardBackwardRatio = 1;
    auto constexpr Id_MuscleMode_ManualCrawling_InitialDistance = 2;
    auto constexpr Id_MuscleMode_ManualCrawling_LastActualDistance = 3;
    auto constexpr Id_MuscleMode_ManualCrawling_LastDistanceDelta = 4;

    auto constexpr Id_Injector_GeneIndex = 0;

    auto constexpr Id_Generator_Additive = 0;
    auto constexpr Id_Generator_NumPulses = 1;
    auto constexpr Id_Generator_ValueOffset = 2;
    auto constexpr Id_Generator_TimeOffset = 3;

    auto constexpr Id_GeneratorMode_SquareSignal_Amplitude = 0;
    auto constexpr Id_GeneratorMode_SquareSignal_Period = 1;

    auto constexpr Id_GeneratorMode_SawtoothSignal_Amplitude = 0;
    auto constexpr Id_GeneratorMode_SawtoothSignal_Period = 1;

    auto constexpr Id_AttackerMode_FreeCell_RestrictToColor = 0;

    auto constexpr Id_Sensor_MinRange = 0;
    auto constexpr Id_Sensor_MaxRange = 1;
    auto constexpr Id_Sensor_AutoTrigger = 2;

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

    auto constexpr Id_Receiver_RestrictToColor = 1;
    auto constexpr Id_Receiver_RestrictToLineage = 2;
}

namespace cereal
{


    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConnectionDesc& data)
    {
        ConnectionDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Connection_ObjectId, data._objectId, defaultObject._objectId);
        loadSave(task, auxiliaries, Id_Connection_Distance, data._distance, defaultObject._distance);
        loadSave(task, auxiliaries, Id_Connection_AngleFromPrevious, data._angleFromPrevious, defaultObject._angleFromPrevious);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ConnectionDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDesc& data)
    {
        SignalDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Signal_Channels, data._channels, defaultObject._channels);
        loadSave(task, auxiliaries, Id_Signal_NumTimesSent, data._numTimesSent, defaultObject._numTimesSent);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalDesc)

    // Dummy struct for backward compatibility with old files that have SignalRestrictionDesc
    struct SignalRestrictionDescLegacy
    {
        uint8_t _mode = 0;
        float _baseAngle = 0.0f;
        float _openingAngle = 0.0f;
    };

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRestrictionDescLegacy& data)
    {
        SignalRestrictionDescLegacy defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);

        // For backward compatibility, read any mode format but discard
        if (task == SerializationTask::Load) {
            auto findResult = auxiliaries.find(Id_SignalRestriction_Mode);
            if (findResult != auxiliaries.end()) {
                auto& variantData = findResult->second;
                if (std::holds_alternative<bool>(variantData)) {
                    data._mode = std::get<bool>(variantData) ? 1 : 0;
                } else if (std::holds_alternative<uint8_t>(variantData)) {
                    data._mode = std::get<uint8_t>(variantData);
                }
            }
        }

        loadSave(task, auxiliaries, Id_SignalRestriction_BaseAngle, data._baseAngle, defaultObject._baseAngle);
        loadSave(task, auxiliaries, Id_SignalRestriction_OpeningAngle, data._openingAngle, defaultObject._openingAngle);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRestrictionDescLegacy)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetDesc& data)
    {
        NeuralNetDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_NeuralNet_Weights, data._weights, defaultObject._weights);
        loadSave(task, auxiliaries, Id_NeuralNet_Biases, data._biases, defaultObject._biases);
        loadSave(task, auxiliaries, Id_NeuralNet_ActivationFunctions, data._activationFunctions, defaultObject._activationFunctions);
        loadSave(task, auxiliaries, Id_NeuralNet_ConnectionWeights, data._connectionWeights, defaultObject._connectionWeights);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(NeuralNetDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, BaseDesc& data)
    {
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(BaseDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DepotDesc& data)
    {
        DepotDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Depot_storageLimit, data._storageLimit, defaultObject._storageLimit);
        loadSave(task, auxiliaries, Id_Depot_StoredUsableEnergy, data._storedUsableEnergy, defaultObject._storedUsableEnergy);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DepotDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorDesc& data)
    {
        ConstructorDesc defaultObject;
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
    SPLIT_SERIALIZATION(ConstructorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TelemetryDesc& data)
    {
        //TelemetryDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(TelemetryDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectEnergyDesc& data)
    {
        DetectEnergyDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorMode_DetectEnergy_MinDensity, data._minDensity, defaultObject._minDensity);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectEnergyDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectStructureDesc& data)
    {
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectStructureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectFreeCellDesc& data)
    {
        DetectFreeCellDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorMode_DetectFreeCell_MinDensity, data._minDensity, defaultObject._minDensity);
        loadSave(task, auxiliaries, Id_SensorMode_DetectFreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectFreeCellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectCreatureDesc& data)
    {
        DetectCreatureDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorMode_DetectCreature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_SensorMode_DetectCreature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_SensorMode_DetectCreature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_SensorMode_DetectCreature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetectCreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorLastMatchDesc& data)
    {
        SensorLastMatchDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorMode_SensorLastMatch_CreatureId, data._creatureId, defaultObject._creatureId);
        loadSave(task, auxiliaries, Id_SensorMode_SensorLastMatch_Pos, data._pos, defaultObject._pos);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SensorLastMatchDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorDesc& data)
    {
        SensorDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Sensor_AutoTrigger, data._autoTrigger, defaultObject._autoTrigger);
        loadSave(task, auxiliaries, Id_Sensor_MinRange, data._minRange, defaultObject._minRange);
        loadSave(task, auxiliaries, Id_Sensor_MaxRange, data._maxRange, defaultObject._maxRange);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode, data._lastMatch);
    }
    SPLIT_SERIALIZATION(SensorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SquareSignalDesc& data)
    {
        SquareSignalDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_GeneratorMode_SquareSignal_Amplitude, data._amplitude, defaultObject._amplitude);
        loadSave(task, auxiliaries, Id_GeneratorMode_SquareSignal_Period, data._period, defaultObject._period);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SquareSignalDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SawtoothSignalDesc& data)
    {
        SawtoothSignalDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_GeneratorMode_SawtoothSignal_Amplitude, data._amplitude, defaultObject._amplitude);
        loadSave(task, auxiliaries, Id_GeneratorMode_SawtoothSignal_Period, data._period, defaultObject._period);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SawtoothSignalDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeneratorDesc& data)
    {
        GeneratorDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Generator_Additive, data._additive, defaultObject._additive);
        loadSave(task, auxiliaries, Id_Generator_NumPulses, data._numPulses, defaultObject._numPulses);
        loadSave(task, auxiliaries, Id_Generator_ValueOffset, data._valueOffset, defaultObject._valueOffset);
        loadSave(task, auxiliaries, Id_Generator_TimeOffset, data._timeOffset, defaultObject._timeOffset);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(GeneratorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackFreeCellDesc& data)
    {
        AttackFreeCellDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_AttackerMode_FreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackFreeCellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackCreatureDesc& data)
    {
        AttackCreatureDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackCreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerDesc& data)
    {
        AttackerDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(AttackerDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorDesc& data)
    {
        InjectorDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Injector_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(InjectorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoBendingDesc& data)
    {
        AutoBendingDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_InitialAngle, data._initialAngle, defaultObject._initialAngle);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoBending_Forward, data._forward, defaultObject._forward);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AutoBendingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualBendingDesc& data)
    {
        ManualBendingDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualBending_InitialAngle, data._initialAngle, defaultObject._initialAngle);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualBending_LastAngleDelta, data._lastAngleDelta, defaultObject._lastAngleDelta);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ManualBendingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AngleBendingDesc& data)
    {
        AngleBendingDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_AngleBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        loadSave(
            task, auxiliaries, Id_MuscleMode_AngleBending_AttractionRepulsionRatio, data._attractionRepulsionRatio, defaultObject._attractionRepulsionRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_AngleBending_InitialAngle, data._initialAngle, defaultObject._initialAngle);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AngleBendingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoCrawlingDesc& data)
    {
        AutoCrawlingDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_MaxAngleDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_InitialDistance, data._initialDistance, defaultObject._initialDistance);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_LastActualDistance, data._lastActualDistance, defaultObject._lastActualDistance);
        loadSave(task, auxiliaries, Id_MuscleMode_AutoCrawling_Forward, data._forward, defaultObject._forward);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AutoCrawlingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualCrawlingDesc& data)
    {
        ManualCrawlingDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_MaxAngleDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_InitialDistance, data._initialDistance, defaultObject._initialDistance);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_LastActualDistance, data._lastActualDistance, defaultObject._lastActualDistance);
        loadSave(task, auxiliaries, Id_MuscleMode_ManualCrawling_LastDistanceDelta, data._lastDistanceDelta, defaultObject._lastDistanceDelta);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ManualCrawlingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DirectMovementDesc& data)
    {
        DirectMovementDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DirectMovementDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleDesc& data)
    {
        MuscleDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Muscle_LastMovementX, data._lastMovementX, defaultObject._lastMovementX);
        loadSave(task, auxiliaries, Id_Muscle_LastMovementY, data._lastMovementY, defaultObject._lastMovementY);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(MuscleDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderDesc& data)
    {
        DefenderDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Defender_Mode, data._mode, defaultObject._mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DefenderDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectStructureDesc& data)
    {
        ReconnectStructureDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectStructureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectFreeCellDesc& data)
    {
        ReconnectFreeCellDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReconnectorMode_FreeCell_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectFreeCellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectCreatureDesc& data)
    {
        ReconnectCreatureDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReconnectorMode_Creature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        loadSave(task, auxiliaries, Id_ReconnectorMode_Creature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        loadSave(task, auxiliaries, Id_ReconnectorMode_Creature_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_ReconnectorMode_Creature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectCreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorDesc& data)
    {
        ReconnectorDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(ReconnectorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorDesc& data)
    {
        DetonatorDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Detonator_State, data._state, defaultObject._state);
        loadSave(task, auxiliaries, Id_Detonator_Countdown, data._countdown, defaultObject._countdown);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetonatorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DigestorDesc& data)
    {
        DigestorDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Digestor_RawEnergyConductivity, data._rawEnergyConductivity, defaultObject._rawEnergyConductivity);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DigestorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDelayDesc& data)
    {
        SignalDelayDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalDelay_Delay, data._delay, defaultObject._delay);
        loadSave(task, auxiliaries, Id_SignalDelay_NumMemoryEntriesInitialized, data._numSignalEntriesInitialized, defaultObject._numSignalEntriesInitialized);
        loadSave(task, auxiliaries, Id_SignalDelay_RingBufferIndex, data._ringBufferIndex, defaultObject._ringBufferIndex);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalDelayDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRecorderDesc& data)
    {
        SignalRecorderDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalRecorder_ReadOnly, data._readOnly, defaultObject._readOnly);
        loadSave(task, auxiliaries, Id_SignalRecorder_State, data._state, defaultObject._state);
        loadSave(task, auxiliaries, Id_SignalRecorder_NumSavedSignalEntries, data._numWrittenSignalEntries, defaultObject._numWrittenSignalEntries);
        loadSave(task, auxiliaries, Id_SignalRecorder_NumReadSignalEntries, data._numReadSignalEntries, defaultObject._numReadSignalEntries);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRecorderDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalStorageDesc& data)
    {
        SignalStorageDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalStorage_ReadOnly, data._readOnly, defaultObject._readOnly);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalStorageDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalIntegratorDesc& data)
    {
        SignalIntegratorDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalIntegrator_NewSignalWeight, data._newSignalWeight, defaultObject._newSignalWeight);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalIntegratorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalEntryDesc& data)
    {
        SignalEntryDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalEntry_Channels, data._channels, defaultObject._channels);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalEntryDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MemoryDesc& data)
    {
        MemoryDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Memory_ChannelBitMask, data._channelBitMask, defaultObject._channelBitMask);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
        ar(data._signalEntries);
    }
    SPLIT_SERIALIZATION(MemoryDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SenderDesc& data)
    {
        SenderDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Sender_Range, data._range, defaultObject._range);
        loadSave(task, auxiliaries, Id_Sender_MaxTimesSent, data._maxTimesSent, defaultObject._maxTimesSent);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SenderDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReceiverDesc& data)
    {
        ReceiverDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Receiver_RestrictToColor, data._restrictToColor, defaultObject._restrictToColor);
        loadSave(task, auxiliaries, Id_Receiver_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReceiverDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CommunicatorDesc& data)
    {
        CommunicatorDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._mode);
    }
    SPLIT_SERIALIZATION(CommunicatorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, StructureDesc& data)
    {
        StructureDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Structure_Energy, data._energy, defaultObject._energy);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(StructureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, FreeCellDesc& data)
    {
        FreeCellDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_FreeCell_Energy, data._energy, defaultObject._energy);
        loadSave(task, auxiliaries, Id_FreeCell_Age, data._age, defaultObject._age);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(FreeCellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellDesc& data)
    {
        CellDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Cell_UsableEnergy, data._usableEnergy, defaultObject._usableEnergy);
        loadSave(task, auxiliaries, Id_Cell_RawEnergy, data._rawEnergy, defaultObject._rawEnergy);
        loadSave(task, auxiliaries, Id_Cell_ReservedEnergy, data._reservedEnergy, defaultObject._reservedEnergy);
        loadSave(task, auxiliaries, Id_Cell_AngleToFront, data._frontAngle, defaultObject._frontAngle);
        loadSave(task, auxiliaries, Id_Cell_Age, data._age, defaultObject._age);
        loadSave(task, auxiliaries, Id_Cell_CellState, data._cellState, defaultObject._cellState);
        loadSave(task, auxiliaries, Id_Cell_ActivationTime, data._activationTime, defaultObject._activationTime);
        loadSave(task, auxiliaries, Id_Cell_NodeIndex, data._nodeIndex, defaultObject._nodeIndex);
        loadSave(task, auxiliaries, Id_Cell_ParentNodeIndex, data._parentNodeIndex, defaultObject._parentNodeIndex);
        loadSave(task, auxiliaries, Id_Cell_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        loadSave(task, auxiliaries, Id_Cell_FrontAngleId, data._frontAngleId, defaultObject._frontAngleId);
        loadSave(task, auxiliaries, Id_Cell_IsFrontAngleRefCell, data._headCell, defaultObject._headCell);
        loadSave(task, auxiliaries, Id_Cell_CreatureId, data._creatureId, defaultObject._creatureId);
        loadSave(task, auxiliaries, Id_Cell_Event, data._event, defaultObject._event);
        loadSave(task, auxiliaries, Id_Cell_EventCounter, data._eventCounter, defaultObject._eventCounter);
        loadSave(task, auxiliaries, Id_Cell_EventPos, data._eventPos, defaultObject._eventPos);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._cellType, data._constructor, data._signal, data._neuralNetwork);
    }
    SPLIT_SERIALIZATION(CellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ObjectDesc& data)
    {
        ObjectDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Object_Id, data._id, defaultObject._id);
        loadSave(task, auxiliaries, Id_Object_Pos, data._pos, defaultObject._pos);
        loadSave(task, auxiliaries, Id_Object_Vel, data._vel, defaultObject._vel);
        loadSave(task, auxiliaries, Id_Object_Stiffness, data._stiffness, defaultObject._stiffness);
        loadSave(task, auxiliaries, Id_Object_Color, data._color, defaultObject._color);
        loadSave(task, auxiliaries, Id_Object_Fixed, data._fixed, defaultObject._fixed);
        loadSave(task, auxiliaries, Id_Object_Sticky, data._sticky, defaultObject._sticky);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data._connections, data._type);
    }
    SPLIT_SERIALIZATION(ObjectDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CreatureDesc& data)
    {
        CreatureDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Creature_Id, data._id, defaultObject._id);
        loadSave(task, auxiliaries, Id_Creature_AncestorId, data._ancestorId, defaultObject._ancestorId);
        loadSave(task, auxiliaries, Id_Creature_Generation, data._generation, defaultObject._generation);
        loadSave(task, auxiliaries, Id_Creature_NumCells, data._numObjects, defaultObject._numObjects);
        loadSave(task, auxiliaries, Id_Creature_FrontAngleId, data._frontAngleId, defaultObject._frontAngleId);
        loadSave(task, auxiliaries, Id_Creature_GenomeId, data._genomeId, defaultObject._genomeId);
        loadSave(task, auxiliaries, Id_Creature_HaveMutationsApplied, data._mutationState, defaultObject._mutationState);

        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(CreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, EnergyDesc& data)
    {
        EnergyDesc defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Particle_Id, data._id, defaultObject._id);
        loadSave(task, auxiliaries, Id_Particle_Pos, data._pos, defaultObject._pos);
        loadSave(task, auxiliaries, Id_Particle_Vel, data._vel, defaultObject._vel);
        loadSave(task, auxiliaries, Id_Particle_Energy, data._energy, defaultObject._energy);
        loadSave(task, auxiliaries, Id_Particle_Color, data._color, defaultObject._color);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(EnergyDesc)

    template <class Archive>
    void serialize(Archive& ar, Desc& description)
    {
        ar(description._objects, description._energies, description._creatures, description._genomes);
    }
}

bool SerializerService::serializeSimulationToFiles(std::filesystem::path const& filename, DeserializedSimulation const& data) const
{
    try {
        log(Priority::Important, "save simulation to " + filename.string());

        if (filename.has_parent_path()) {
            std::filesystem::create_directories(filename.parent_path());
        }

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

bool SerializerService::serializeGenomeToFile(std::filesystem::path const& filename, GenomeDesc const& genome) const
{
    try {
        log(Priority::Important, "save genome to " + filename.string());
        //wrap constructor cell around genome
        Desc description;
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

bool SerializerService::deserializeGenomeFromFile(GenomeDesc& genome, std::filesystem::path const& filename) const
{
    try {
        log(Priority::Important, "load genome from " + filename.string());
        Desc description;
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

        Desc description;
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

        Desc description;
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

bool SerializerService::serializeContentToFile(std::filesystem::path const& filename, Desc const& content) const
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

bool SerializerService::deserializeContentFromFile(Desc& content, std::filesystem::path const& filename) const
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

void SerializerService::serializeDescription(Desc const& description, std::ostream& stream) const
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(Const::ProgramVersion);
    archive(description);
}

bool SerializerService::deserializeDescription(Desc& description, std::filesystem::path const& filename) const
{
    zstr::ifstream stream(filename.string(), std::ios::binary);
    if (!stream) {
        return false;
    }
    deserializeDescription(description, stream);
    return true;
}

void SerializerService::deserializeDescription(Desc& description, std::istream& stream) const
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
            return &dataPoints.numObjects;
        } else if (colIndex == 2) {
            return &dataPoints.numSelfReplicators;
        } else if (colIndex == 3) {
            return &dataPoints.numViruses;
        } else if (colIndex == 4) {
            return &dataPoints.numFreeCells;
        } else if (colIndex == 5) {
            return &dataPoints.numEnergyParticles;
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

bool SerializerService::wrapGenome(Desc& output, GenomeDesc const& input) const
{
    output.clear();
    output._genomes.emplace_back(input);
    output._creatures.emplace_back(CreatureDesc().genomeId(input._id));
    return true;
}

bool SerializerService::unwrapGenome(GenomeDesc& output, Desc& input) const
{
    if (input._genomes.size() != 1) {
        return false;
    }
    output = input._genomes.front();
    return true;
}
