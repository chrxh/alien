#include "SerializerService.h"

#include <algorithm>
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

    using AttributeMap = std::unordered_map<int, VariantData>;

    // RAII pattern
    template <class Archive>
    class SerializationScope
    {
    public:
        SerializationScope(SerializationTask task, Archive& ar)
            : _task(task)
            , _ar(ar)
        {
            if (_task == SerializationTask::Load) {
                _ar(_attributeMap);
            }
        }

        ~SerializationScope()
        {
            std::sort(_deferredDescOps.begin(), _deferredDescOps.end(), [](auto const& left, auto const& right) { return left.id < right.id; });

            // Process deferred operations
            if (_task == SerializationTask::Save) {

                // Save map first
                _ar(_attributeMap);

                // Save sorted ids
                std::vector<int> sortedIds;
                sortedIds.reserve(_deferredDescOps.size());
                for (const auto& op : _deferredDescOps) {
                    sortedIds.push_back(op.id);
                }
                _ar(sortedIds);

                // Then write size-prefixed Desc data in sorted id order
                for (auto const& op : _deferredDescOps) {
                    op.serializeFunc();
                }
            } else {

                // Read sorted ids
                std::vector<int> savedIds;
                _ar(savedIds);

                // For each id, check if we have a deferred read operation, otherwise skip bytes
                auto deferredOpIndex = 0;
                auto deferredOpSize = _deferredDescOps.size();
                for (int savedId : savedIds) {
                    // deferredOpIndex is an optimization to avoid
                    // `std::find_if(_deferredDescOps.begin(), _deferredDescOps.end(), [id](const auto& op) { return op.id == id; });`
                    // for each savedId (savedIds and _deferredDescOps are sorted)
                    while (deferredOpIndex < deferredOpSize && _deferredDescOps.at(deferredOpIndex).id < savedId) {
                        ++deferredOpIndex;
                    }

                    if (deferredOpIndex < deferredOpSize && _deferredDescOps.at(deferredOpIndex).id == savedId) {
                        // We want to read this Desc - execute the read
                        _deferredDescOps.at(deferredOpIndex).serializeFunc();
                    } else {
                        // Skip this Desc - read size and skip data
                        uint64_t dataSize = 0;
                        _ar(dataSize);
                        std::vector<uint8_t> buffer(dataSize);
                        _ar(cereal::binary_data(buffer.data(), dataSize));
                    }
                }
            }
        }

        SerializationScope(const SerializationScope&) = delete;
        SerializationScope& operator=(const SerializationScope&) = delete;

        SerializationScope(SerializationScope&&) = default;
        SerializationScope& operator=(SerializationScope&&) = default;

        // Implicit conversion to reference
        operator std::unordered_map<int, VariantData>&() & { return _attributeMap; }

        template <typename T>
        void addMember(int key, T& value, T const& defaultValue)
        {
            if (_task == SerializationTask::Load) {
                auto findResult = _attributeMap.find(key);
                if (findResult != _attributeMap.end()) {
                    auto const& variantData = findResult->second;
                    value = std::get<T>(variantData);
                } else {
                    value = defaultValue;
                }
            } else {
                _attributeMap.emplace(key, value);
            }
        }

        template <typename T>
        void addDesc(int key, T& value)
        {
            if (_task == SerializationTask::Save) {
                // Defer the save operation
                addDeferredDescOp(key, [this, &value]() {
                    // Serialize to buffer
                    std::ostringstream ss(std::ios::binary);
                    {
                        cereal::PortableBinaryOutputArchive bufferAr(ss);
                        bufferAr(value);
                    }
                    auto serializedData = std::move(ss).str();
                    uint64_t dataSize = serializedData.size();

                    // Write size-prefixed data
                    _ar(dataSize);
                    _ar(cereal::binary_data(serializedData.data(), dataSize));
                });
            } else {
                // Defer the load operation
                addDeferredDescOp(key, [this, &value]() {
                    // Read size-prefixed data
                    uint64_t dataSize = 0;
                    _ar(dataSize);

                    // Read serialized data into buffer
                    std::string serializedData(dataSize, '\0');
                    _ar(cereal::binary_data(serializedData.data(), dataSize));

                    // Deserialize from buffer
                    std::istringstream ss(std::move(serializedData), std::ios::binary);
                    cereal::PortableBinaryInputArchive bufferAr(ss);
                    bufferAr(value);
                });
            }
        }

        // Specialized overload for std::vector<NeuralNetWeight> - converts to/from std::vector<int8_t> for serialization
        void addMember(int key, std::vector<NeuralNetWeight>& value, std::vector<NeuralNetWeight> const& defaultValue)
        {
            if (_task == SerializationTask::Load) {
                auto findResult = _attributeMap.find(key);
                if (findResult != _attributeMap.end()) {
                    auto const& variantData = findResult->second;
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
                _attributeMap.emplace(key, int8Vec);
            }
        }

    private:
        void addDeferredDescOp(int id, std::function<void()> serializeFunc) { _deferredDescOps.push_back({id, std::move(serializeFunc)}); }

        struct DeferredOperation
        {
            int id;
            std::function<void()> serializeFunc;
        };

        SerializationTask _task;
        Archive& _ar;
        AttributeMap _attributeMap;
        std::vector<DeferredOperation> _deferredDescOps;
    };

    template <class Archive>
    SerializationScope<Archive> getSerializationScope(SerializationTask task, Archive& ar)
    {
        return SerializationScope<Archive>(task, ar);
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
    auto constexpr Id_Genome_AccumulatedMutations = 4;
    auto constexpr Id_Genome_PrevLineageId = 5;

    auto constexpr Id_NeuronMutation_EventProbability = 0;
    auto constexpr Id_NeuronMutation_WeightSigma = 1;
    auto constexpr Id_NeuronMutation_BiasSigma = 2;
    auto constexpr Id_NeuronMutation_ActivationFunctionProbability = 3;

    auto constexpr Id_ConnectionMutation_EventProbability = 0;
    auto constexpr Id_ConnectionMutation_Sigma = 1;

    auto constexpr Id_CellTypePropertiesMutation_EventProbability = 0;
    auto constexpr Id_CellTypePropertiesMutation_Sigma = 1;
    auto constexpr Id_CellTypePropertiesMutation_Probability = 2;

    auto constexpr Id_CellTypeModeMutation_EventProbability = 0;

    auto constexpr Id_CellTypeMutation_EventProbability = 0;

    auto constexpr Id_VoidMutation_EventProbability = 0;

    auto constexpr Id_ConstructorMutation_EventProbability = 0;
    auto constexpr Id_ConstructorMutation_Sigma = 1;
    auto constexpr Id_ConstructorMutation_Probability = 2;

    auto constexpr Id_Gene_Name = 0;
    auto constexpr Id_Gene_Shape = 1;
    auto constexpr Id_Gene_Stiffness = 5;
    auto constexpr Id_Gene_ConnectionDistance = 6;
    auto constexpr Id_Gene_HomogeneCellType = 7;

    auto constexpr Id_Node_ReferenceAngle = 0;
    auto constexpr Id_Node_Color = 1;

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
    auto constexpr Id_ConstructorGenome_ReservedEnergy = 5;
    auto constexpr Id_ConstructorGenome_Separation = 6;
    auto constexpr Id_ConstructorGenome_NumBranches = 7;
    auto constexpr Id_ConstructorGenome_NumConcatenations = 8;

    auto constexpr Id_SensorGenome_AutoTrigger = 0;
    auto constexpr Id_SensorGenome_MinRange = 1;
    auto constexpr Id_SensorGenome_MaxRange = 2;
    auto constexpr Id_SensorGenome_TagForAttackers = 3;

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
    auto constexpr Id_GeneratorGenome_MinValue = 4;
    auto constexpr Id_GeneratorGenome_MaxValue = 5;
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

    // Description member keys
    auto constexpr Id_Node_NeuralNetwork = 2;
    auto constexpr Id_Node_CellType = 3;
    auto constexpr Id_Node_Constructor = 4;

    auto constexpr Id_Gene_Nodes = 8;

    auto constexpr Id_MutationRates_NeuronMutation1 = 1;
    auto constexpr Id_MutationRates_NeuronMutation2 = 2;
    auto constexpr Id_MutationRates_ConnectionMutation1 = 3;
    auto constexpr Id_MutationRates_ConnectionMutation2 = 4;
    auto constexpr Id_MutationRates_CellTypePropertiesMutation1 = 5;
    auto constexpr Id_MutationRates_CellTypePropertiesMutation2 = 6;
    auto constexpr Id_MutationRates_CellTypeModeMutation = 7;
    auto constexpr Id_MutationRates_CellTypeMutation = 8;
    auto constexpr Id_MutationRates_VoidMutation = 9;
    auto constexpr Id_MutationRates_ConstructorMutation1 = 10;
    auto constexpr Id_MutationRates_ConstructorMutation2 = 11;

    auto constexpr Id_Genome_Genes = 6;
    auto constexpr Id_Genome_MutationRates = 7;

    auto constexpr Id_SensorGenome_Mode = 4;
    auto constexpr Id_GeneratorGenome_Mode = 3;
    auto constexpr Id_AttackerGenome_Mode = 0;
    auto constexpr Id_MuscleGenome_Mode = 0;
    auto constexpr Id_ReconnectorGenome_Mode = 0;
    auto constexpr Id_MemoryGenome_Mode = 1;
    auto constexpr Id_MemoryGenome_SignalEntries = 2;
    auto constexpr Id_CommunicatorGenome_Mode = 0;
}

namespace cereal
{
    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetGenomeDesc& data)
    {
        NeuralNetGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_NeuralNetGenome_Weights, data._weights, defaultObject._weights);
        scope.addMember(Id_NeuralNetGenome_Biases, data._biases, defaultObject._biases);
        scope.addMember(Id_NeuralNetGenome_ActivationFunctions, data._activationFunctions, defaultObject._activationFunctions);
        scope.addMember(Id_NeuralNetGenome_ConnectionWeights, data._connectionWeights, defaultObject._connectionWeights);
    }
    SPLIT_SERIALIZATION(NeuralNetGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, BaseGenomeDesc& data)
    {
        BaseGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(BaseGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DepotGenomeDesc& data)
    {
        DepotGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DepotGenome_storageLimit, data._storageLimit, defaultObject._storageLimit);
        scope.addMember(Id_DepotGenome_InitialStoredUsableEnergy, data._initialStoredUsableEnergy, defaultObject._initialStoredUsableEnergy);
    }
    SPLIT_SERIALIZATION(DepotGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorGenomeDesc& data)
    {
        ConstructorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ConstructorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        scope.addMember(Id_ConstructorGenome_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        scope.addMember(Id_ConstructorGenome_ConstructionActivationTime, data._constructionActivationTime, defaultObject._constructionActivationTime);
        scope.addMember(Id_ConstructorGenome_ConstructionAngle, data._constructionAngle, defaultObject._constructionAngle);
        scope.addMember(Id_ConstructorGenome_ProvideEnergy, data._provideEnergy, defaultObject._provideEnergy);
        scope.addMember(Id_ConstructorGenome_ReservedEnergy, data._reservedEnergy, defaultObject._reservedEnergy);
        scope.addMember(Id_ConstructorGenome_Separation, data._separation, defaultObject._separation);
        scope.addMember(Id_ConstructorGenome_NumBranches, data._numBranches, defaultObject._numBranches);
        scope.addMember(Id_ConstructorGenome_NumConcatenations, data._numConcatenations, defaultObject._numConcatenations);
    }
    SPLIT_SERIALIZATION(ConstructorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TelemetryGenomeDesc& data)
    {
        //TelemetryGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(TelemetryGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectEnergyGenomeDesc& data)
    {
        DetectEnergyGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorModeGenome_DetectEnergy_MinDensity, data._minDensity, defaultObject._minDensity);
    }
    SPLIT_SERIALIZATION(DetectEnergyGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectSolidGenomeDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(DetectSolidGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectFreeCellGenomeDesc& data)
    {
        DetectFreeCellGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorModeGenome_DetectFreeCell_MinDensity, data._minDensity, defaultObject._minDensity);
        scope.addMember(Id_SensorModeGenome_DetectFreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    SPLIT_SERIALIZATION(DetectFreeCellGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectCreatureGenomeDesc& data)
    {
        DetectCreatureGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorModeGenome_DetectCreature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        scope.addMember(Id_SensorModeGenome_DetectCreature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        scope.addMember(Id_SensorModeGenome_DetectCreature_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
        scope.addMember(Id_SensorModeGenome_DetectCreature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
    }
    SPLIT_SERIALIZATION(DetectCreatureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorGenomeDesc& data)
    {
        SensorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorGenome_AutoTrigger, data._autoTrigger, defaultObject._autoTrigger);
        scope.addMember(Id_SensorGenome_TagForAttackers, data._tagForAttackers, defaultObject._tagForAttackers);
        scope.addMember(Id_SensorGenome_MinRange, data._minRange, defaultObject._minRange);
        scope.addMember(Id_SensorGenome_MaxRange, data._maxRange, defaultObject._maxRange);
        scope.addDesc(Id_SensorGenome_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(SensorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SquareSignalGenomeDesc& data)
    {
        SquareSignalGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeneratorModeGenome_SquareSignal_Period, data._period, defaultObject._period);
    }
    SPLIT_SERIALIZATION(SquareSignalGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SawtoothSignalGenomeDesc& data)
    {
        SawtoothSignalGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeneratorModeGenome_SawtoothSignal_Period, data._period, defaultObject._period);
    }
    SPLIT_SERIALIZATION(SawtoothSignalGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeneratorGenomeDesc& data)
    {
        GeneratorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeneratorGenome_Additive, data._additive, defaultObject._additive);
        scope.addMember(Id_GeneratorGenome_MinValue, data._minValue, defaultObject._minValue);
        scope.addMember(Id_GeneratorGenome_MaxValue, data._maxValue, defaultObject._maxValue);
        scope.addMember(Id_GeneratorGenome_TimeOffset, data._timeOffset, defaultObject._timeOffset);
        scope.addDesc(Id_GeneratorGenome_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(GeneratorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackFreeCellGenomeDesc& data)
    {
        AttackFreeCellGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_AttackerModeGenome_FreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    SPLIT_SERIALIZATION(AttackFreeCellGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackCreatureGenomeDesc& data)
    {
        AttackCreatureGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(AttackCreatureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerGenomeDesc& data)
    {
        AttackerGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_AttackerGenome_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(AttackerGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorGenomeDesc& data)
    {
        InjectorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_InjectorGenome_GeneIndex, data._geneIndex, defaultObject._geneIndex);
    }
    SPLIT_SERIALIZATION(InjectorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoBendingGenomeDesc& data)
    {
        AutoBendingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_AutoBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        scope.addMember(Id_MuscleModeGenome_AutoBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
    }
    SPLIT_SERIALIZATION(AutoBendingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualBendingGenomeDesc& data)
    {
        ManualBendingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_ManualBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        scope.addMember(Id_MuscleModeGenome_ManualBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
    }
    SPLIT_SERIALIZATION(ManualBendingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AngleBendingGenomeDesc& data)
    {
        AngleBendingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_AngleBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        scope.addMember(Id_MuscleModeGenome_AngleBending_AttractionRepulsionRatio, data._attractionRepulsionRatio, defaultObject._attractionRepulsionRatio);
    }
    SPLIT_SERIALIZATION(AngleBendingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoCrawlingGenomeDesc& data)
    {
        AutoCrawlingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_AutoCrawling_MaxDistanceDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        scope.addMember(Id_MuscleModeGenome_AutoCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
    }
    SPLIT_SERIALIZATION(AutoCrawlingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualCrawlingGenomeDesc& data)
    {
        ManualCrawlingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_ManualCrawling_MaxDistanceDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        scope.addMember(Id_MuscleModeGenome_ManualCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
    }
    SPLIT_SERIALIZATION(ManualCrawlingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DirectMovementGenomeDesc& data)
    {
        DirectMovementGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(DirectMovementGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleGenomeDesc& data)
    {
        MuscleGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_MuscleGenome_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(MuscleGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderGenomeDesc& data)
    {
        DefenderGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DefenderGenome_Mode, data._mode, defaultObject._mode);
    }
    SPLIT_SERIALIZATION(DefenderGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectSolidGenomeDesc& data)
    {
        ReconnectSolidGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(ReconnectSolidGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectFreeCellGenomeDesc& data)
    {
        ReconnectFreeCellGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ReconnectorModeGenome_FreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    SPLIT_SERIALIZATION(ReconnectFreeCellGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectCreatureGenomeDesc& data)
    {
        ReconnectCreatureGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ReconnectorModeGenome_Creature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        scope.addMember(Id_ReconnectorModeGenome_Creature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        scope.addMember(Id_ReconnectorModeGenome_Creature_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
        scope.addMember(Id_ReconnectorModeGenome_Creature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
    }
    SPLIT_SERIALIZATION(ReconnectCreatureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorGenomeDesc& data)
    {
        ReconnectorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_ReconnectorGenome_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(ReconnectorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorGenomeDesc& data)
    {
        DetonatorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DetonatorGenome_Countdown, data._countdown, defaultObject._countdown);
    }
    SPLIT_SERIALIZATION(DetonatorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DigestorGenomeDesc& data)
    {
        DigestorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DigestorGenome_RawEnergyConductivity, data._rawEnergyConductivity, defaultObject._rawEnergyConductivity);
    }
    SPLIT_SERIALIZATION(DigestorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDelayGenomeDesc& data)
    {
        SignalDelayGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalDelayGenome_Delay, data._delay, defaultObject._delay);
    }
    SPLIT_SERIALIZATION(SignalDelayGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRecorderGenomeDesc& data)
    {
        SignalRecorderGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalRecorderGenome_ReadOnly, data._readOnly, defaultObject._readOnly);
        scope.addMember(Id_SignalRecorderGenome_NumSavedSignalEntries, data._numWrittenSignalEntries, defaultObject._numWrittenSignalEntries);
    }
    SPLIT_SERIALIZATION(SignalRecorderGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalStorageGenomeDesc& data)
    {
        SignalStorageGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalStorageGenome_ReadOnly, data._readOnly, defaultObject._readOnly);
    }
    SPLIT_SERIALIZATION(SignalStorageGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalIntegratorGenomeDesc& data)
    {
        SignalIntegratorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalIntegratorGenome_NewSignalWeight, data._newSignalWeight, defaultObject._newSignalWeight);
    }
    SPLIT_SERIALIZATION(SignalIntegratorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalEntryGenomeDesc& data)
    {
        SignalEntryGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalEntryGenome_Channels, data._channels, defaultObject._channels);
    }
    SPLIT_SERIALIZATION(SignalEntryGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MemoryGenomeDesc& data)
    {
        MemoryGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MemoryGenome_ChannelBitMask, data._channelBitMask, defaultObject._channelBitMask);
        scope.addDesc(Id_MemoryGenome_Mode, data._mode);
        scope.addDesc(Id_MemoryGenome_SignalEntries, data._signalEntries);
    }
    SPLIT_SERIALIZATION(MemoryGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SenderGenomeDesc& data)
    {
        SenderGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SenderGenome_Range, data._range, defaultObject._range);
        scope.addMember(Id_SenderGenome_MaxTimesSent, data._maxTimesSent, defaultObject._maxTimesSent);
    }
    SPLIT_SERIALIZATION(SenderGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReceiverGenomeDesc& data)
    {
        ReceiverGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ReceiverGenome_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
        scope.addMember(Id_ReceiverGenome_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
    }
    SPLIT_SERIALIZATION(ReceiverGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CommunicatorGenomeDesc& data)
    {
        CommunicatorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_CommunicatorGenome_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(CommunicatorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, VoidGenomeDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(VoidGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NodeDesc& data)
    {
        NodeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Node_ReferenceAngle, data._referenceAngle, defaultObject._referenceAngle);
        scope.addMember(Id_Node_Color, data._color, defaultObject._color);
        scope.addDesc(Id_Node_NeuralNetwork, data._neuralNetwork);
        scope.addDesc(Id_Node_CellType, data._cellType);
        scope.addDesc(Id_Node_Constructor, data._constructor);
    }
    SPLIT_SERIALIZATION(NodeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeneDesc& data)
    {
        GeneDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Gene_Name, data._name, defaultObject._name);
        scope.addMember(Id_Gene_Shape, data._shape, defaultObject._shape);
        scope.addMember(Id_Gene_Stiffness, data._stiffness, defaultObject._stiffness);
        scope.addMember(Id_Gene_ConnectionDistance, data._connectionDistance, defaultObject._connectionDistance);
        scope.addMember(Id_Gene_HomogeneCellType, data._homogeneCellType, defaultObject._homogeneCellType);
        scope.addDesc(Id_Gene_Nodes, data._nodes);
    }
    SPLIT_SERIALIZATION(GeneDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuronMutationDesc& data)
    {
        NeuronMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_NeuronMutation_EventProbability, data._eventProbability, defaultObject._eventProbability);
        scope.addMember(Id_NeuronMutation_WeightSigma, data._weightSigma, defaultObject._weightSigma);
        scope.addMember(Id_NeuronMutation_BiasSigma, data._biasSigma, defaultObject._biasSigma);
        scope.addMember(Id_NeuronMutation_ActivationFunctionProbability, data._activationFunctionProbability, defaultObject._activationFunctionProbability);
    }
    SPLIT_SERIALIZATION(NeuronMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConnectionMutationDesc& data)
    {
        ConnectionMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ConnectionMutation_EventProbability, data._eventProbability, defaultObject._eventProbability);
        scope.addMember(Id_ConnectionMutation_Sigma, data._sigma, defaultObject._sigma);
    }
    SPLIT_SERIALIZATION(ConnectionMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellTypePropertiesMutationDesc& data)
    {
        CellTypePropertiesMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_CellTypePropertiesMutation_EventProbability, data._eventProbability, defaultObject._eventProbability);
        scope.addMember(Id_CellTypePropertiesMutation_Sigma, data._sigma, defaultObject._sigma);
        scope.addMember(Id_CellTypePropertiesMutation_Probability, data._probability, defaultObject._probability);
    }
    SPLIT_SERIALIZATION(CellTypePropertiesMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellTypeModeMutationDesc& data)
    {
        CellTypeModeMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_CellTypeModeMutation_EventProbability, data._eventProbability, defaultObject._eventProbability);
    }
    SPLIT_SERIALIZATION(CellTypeModeMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellTypeMutationDesc& data)
    {
        CellTypeMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_CellTypeMutation_EventProbability, data._eventProbability, defaultObject._eventProbability);
    }
    SPLIT_SERIALIZATION(CellTypeMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, VoidMutationDesc& data)
    {
        VoidMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_VoidMutation_EventProbability, data._eventProbability, defaultObject._eventProbability);
    }
    SPLIT_SERIALIZATION(VoidMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorMutationDesc& data)
    {
        ConstructorMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ConstructorMutation_EventProbability, data._eventProbability, defaultObject._eventProbability);
        scope.addMember(Id_ConstructorMutation_Sigma, data._sigma, defaultObject._sigma);
        scope.addMember(Id_ConstructorMutation_Probability, data._probability, defaultObject._probability);
    }
    SPLIT_SERIALIZATION(ConstructorMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MutationRatesDesc& data)
    {
        MutationRatesDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_MutationRates_NeuronMutation1, data._neuronMutations[0]);
        scope.addDesc(Id_MutationRates_NeuronMutation2, data._neuronMutations[1]);
        scope.addDesc(Id_MutationRates_ConnectionMutation1, data._connectionMutations[0]);
        scope.addDesc(Id_MutationRates_ConnectionMutation2, data._connectionMutations[1]);
        scope.addDesc(Id_MutationRates_CellTypePropertiesMutation1, data._cellTypePropertiesMutations[0]);
        scope.addDesc(Id_MutationRates_CellTypePropertiesMutation2, data._cellTypePropertiesMutations[1]);
        scope.addDesc(Id_MutationRates_CellTypeModeMutation, data._cellTypeModeMutation);
        scope.addDesc(Id_MutationRates_CellTypeMutation, data._cellTypeMutation);
        scope.addDesc(Id_MutationRates_VoidMutation, data._voidMutation);
        scope.addDesc(Id_MutationRates_ConstructorMutation1, data._constructorMutations[0]);
        scope.addDesc(Id_MutationRates_ConstructorMutation2, data._constructorMutations[1]);
    }
    SPLIT_SERIALIZATION(MutationRatesDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GenomeDesc& data)
    {
        GenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Genome_Id, data._id, defaultObject._id);
        scope.addMember(Id_Genome_Name, data._name, defaultObject._name);
        scope.addMember(Id_Genome_LineageId, data._lineageId, defaultObject._lineageId);
        scope.addMember(Id_Genome_AccumulatedMutations, data._accumulatedMutations, defaultObject._accumulatedMutations);
        scope.addMember(Id_Genome_PrevLineageId, data._prevLineageId, defaultObject._prevLineageId);
        scope.addMember(Id_Genome_FrontAngle, data._frontAngle, defaultObject._frontAngle);
        scope.addDesc(Id_Genome_Genes, data._genes);
        scope.addDesc(Id_Genome_MutationRates, data._mutationRates);
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
    auto constexpr Id_Creature_HeadUpdateId = 5;
    auto constexpr Id_Creature_GenomeId = 6;
    auto constexpr Id_Creature_MutationState = 7;

    auto constexpr Id_Solid_Energy = 0;

    auto constexpr Id_Fluid_Energy = 0;
    auto constexpr Id_Fluid_Glow = 1;

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
    auto constexpr Id_Cell_HeadUpdateId = 11;
    auto constexpr Id_Cell_HeadCell = 12;
    auto constexpr Id_Cell_CreatureId = 13;
    auto constexpr Id_Cell_Event = 14;
    auto constexpr Id_Cell_EventCounter = 15;
    auto constexpr Id_Cell_EventPos = 16;
    auto constexpr Id_Cell_LastUpdate = 18;
    auto constexpr Id_Cell_ConcatenationIndex = 19;
    auto constexpr Id_Cell_BranchIndex = 20;

    auto constexpr Id_Object_Id = 0;
    auto constexpr Id_Object_Pos = 2;
    auto constexpr Id_Object_Vel = 3;
    auto constexpr Id_Object_Stiffness = 4;
    auto constexpr Id_Object_Color = 5;
    auto constexpr Id_Object_Fixed = 6;
    auto constexpr Id_Object_Sticky = 17;

    auto constexpr Id_Signal_Channels = 0;
    auto constexpr Id_Signal_NumTimesSent = 1;

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
    auto constexpr Id_Constructor_LastConstructedCellId = 5;
    auto constexpr Id_Constructor_ConstructionAngle = 7;
    auto constexpr Id_Constructor_ProvideEnergy = 8;
    auto constexpr Id_Constructor_CurrentOffspring = 9;
    auto constexpr Id_Constructor_ReservedEnergy = 10;
    auto constexpr Id_Constructor_Separation = 11;
    auto constexpr Id_Constructor_NumBranches = 12;
    auto constexpr Id_Constructor_NumConcatenations = 13;

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
    auto constexpr Id_Generator_TimeOffset = 3;
    auto constexpr Id_Generator_MinValue = 5;
    auto constexpr Id_Generator_MaxValue = 6;

    auto constexpr Id_GeneratorMode_SquareSignal_Period = 1;
    auto constexpr Id_GeneratorMode_SawtoothSignal_Period = 1;

    auto constexpr Id_AttackerMode_FreeCell_RestrictToColor = 0;

    auto constexpr Id_Sensor_MinRange = 0;
    auto constexpr Id_Sensor_MaxRange = 1;
    auto constexpr Id_Sensor_AutoTrigger = 2;
    auto constexpr Id_Sensor_TagForAttackers = 3;

    auto constexpr Id_SensorMode_DetectEnergy_MinDensity = 0;

    auto constexpr Id_SensorMode_DetectFreeCell_MinDensity = 0;
    auto constexpr Id_SensorMode_DetectFreeCell_RestrictToColor = 1;

    auto constexpr Id_SensorMode_SensorLastMatch_CreatureIdPart = 0;
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

    // Description member keys for objects data
    auto constexpr Id_Cell_CellType = 21;
    auto constexpr Id_Cell_Constructor = 22;
    auto constexpr Id_Cell_Signal = 23;
    auto constexpr Id_Cell_NeuralNetwork = 24;

    auto constexpr Id_Object_Connections = 7;
    auto constexpr Id_Object_Type = 8;

    auto constexpr Id_Sensor_Mode = 4;
    auto constexpr Id_Sensor_LastMatch = 5;
    auto constexpr Id_Generator_Mode = 4;
    auto constexpr Id_Attacker_Mode = 0;
    auto constexpr Id_Muscle_Mode = 3;
    auto constexpr Id_Reconnector_Mode = 0;
    auto constexpr Id_Memory_Mode = 1;
    auto constexpr Id_Memory_SignalEntries = 2;
    auto constexpr Id_Communicator_Mode = 0;

    auto constexpr Id_Desc_Objects = 0;
    auto constexpr Id_Desc_Energies = 1;
    auto constexpr Id_Desc_Creatures = 2;
    auto constexpr Id_Desc_Genomes = 3;
}

namespace cereal
{


    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConnectionDesc& data)
    {
        ConnectionDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Connection_ObjectId, data._objectId, defaultObject._objectId);
        scope.addMember(Id_Connection_Distance, data._distance, defaultObject._distance);
        scope.addMember(Id_Connection_AngleFromPrevious, data._angleFromPrevious, defaultObject._angleFromPrevious);
    }
    SPLIT_SERIALIZATION(ConnectionDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDesc& data)
    {
        SignalDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Signal_Channels, data._channels, defaultObject._channels);
        scope.addMember(Id_Signal_NumTimesSent, data._numTimesSent, defaultObject._numTimesSent);
    }
    SPLIT_SERIALIZATION(SignalDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetDesc& data)
    {
        NeuralNetDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_NeuralNet_Weights, data._weights, defaultObject._weights);
        scope.addMember(Id_NeuralNet_Biases, data._biases, defaultObject._biases);
        scope.addMember(Id_NeuralNet_ActivationFunctions, data._activationFunctions, defaultObject._activationFunctions);
        scope.addMember(Id_NeuralNet_ConnectionWeights, data._connectionWeights, defaultObject._connectionWeights);
    }
    SPLIT_SERIALIZATION(NeuralNetDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, BaseDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(BaseDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DepotDesc& data)
    {
        DepotDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Depot_storageLimit, data._storageLimit, defaultObject._storageLimit);
        scope.addMember(Id_Depot_StoredUsableEnergy, data._storedUsableEnergy, defaultObject._storedUsableEnergy);
    }
    SPLIT_SERIALIZATION(DepotDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorDesc& data)
    {
        ConstructorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Constructor_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        scope.addMember(Id_Constructor_ConstructionActivationTime, data._constructionActivationTime, defaultObject._constructionActivationTime);
        scope.addMember(Id_Constructor_ConstructionAngle, data._constructionAngle, defaultObject._constructionAngle);
        scope.addMember(Id_Constructor_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        scope.addMember(Id_Constructor_LastConstructedCellId, data._lastConstructedCellId, defaultObject._lastConstructedCellId);
        scope.addMember(Id_Constructor_CurrentOffspring, data._currentOffspring, defaultObject._currentOffspring);
        scope.addMember(Id_Constructor_ProvideEnergy, data._provideEnergy, defaultObject._provideEnergy);
        scope.addMember(Id_Constructor_ReservedEnergy, data._reservedEnergy, defaultObject._reservedEnergy);
        scope.addMember(Id_Constructor_Separation, data._separation, defaultObject._separation);
        scope.addMember(Id_Constructor_NumBranches, data._numBranches, defaultObject._numBranches);
        scope.addMember(Id_Constructor_NumConcatenations, data._numConcatenations, defaultObject._numConcatenations);
    }
    SPLIT_SERIALIZATION(ConstructorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TelemetryDesc& data)
    {
        //TelemetryDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(TelemetryDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectEnergyDesc& data)
    {
        DetectEnergyDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorMode_DetectEnergy_MinDensity, data._minDensity, defaultObject._minDensity);
    }
    SPLIT_SERIALIZATION(DetectEnergyDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectSolidDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(DetectSolidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectFreeCellDesc& data)
    {
        DetectFreeCellDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorMode_DetectFreeCell_MinDensity, data._minDensity, defaultObject._minDensity);
        scope.addMember(Id_SensorMode_DetectFreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    SPLIT_SERIALIZATION(DetectFreeCellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectCreatureDesc& data)
    {
        DetectCreatureDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorMode_DetectCreature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        scope.addMember(Id_SensorMode_DetectCreature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        scope.addMember(Id_SensorMode_DetectCreature_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
        scope.addMember(Id_SensorMode_DetectCreature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
    }
    SPLIT_SERIALIZATION(DetectCreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorLastMatchDesc& data)
    {
        SensorLastMatchDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorMode_SensorLastMatch_CreatureIdPart, data._creatureIdPart, defaultObject._creatureIdPart);
        scope.addMember(Id_SensorMode_SensorLastMatch_Pos, data._pos, defaultObject._pos);
    }
    SPLIT_SERIALIZATION(SensorLastMatchDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorDesc& data)
    {
        SensorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Sensor_AutoTrigger, data._autoTrigger, defaultObject._autoTrigger);
        scope.addMember(Id_Sensor_TagForAttackers, data._tagForAttackers, defaultObject._tagForAttackers);
        scope.addMember(Id_Sensor_MinRange, data._minRange, defaultObject._minRange);
        scope.addMember(Id_Sensor_MaxRange, data._maxRange, defaultObject._maxRange);
        scope.addDesc(Id_Sensor_Mode, data._mode);
        scope.addDesc(Id_Sensor_LastMatch, data._lastMatch);
    }
    SPLIT_SERIALIZATION(SensorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SquareSignalDesc& data)
    {
        SquareSignalDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeneratorMode_SquareSignal_Period, data._period, defaultObject._period);
    }
    SPLIT_SERIALIZATION(SquareSignalDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SawtoothSignalDesc& data)
    {
        SawtoothSignalDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeneratorMode_SawtoothSignal_Period, data._period, defaultObject._period);
    }
    SPLIT_SERIALIZATION(SawtoothSignalDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeneratorDesc& data)
    {
        GeneratorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Generator_Additive, data._additive, defaultObject._additive);
        scope.addMember(Id_Generator_NumPulses, data._numPulses, defaultObject._numPulses);
        scope.addMember(Id_Generator_MinValue, data._minValue, defaultObject._minValue);
        scope.addMember(Id_Generator_MaxValue, data._maxValue, defaultObject._maxValue);
        scope.addMember(Id_Generator_TimeOffset, data._timeOffset, defaultObject._timeOffset);
        scope.addDesc(Id_Generator_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(GeneratorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackFreeCellDesc& data)
    {
        AttackFreeCellDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_AttackerMode_FreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    SPLIT_SERIALIZATION(AttackFreeCellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackCreatureDesc& data)
    {
        AttackCreatureDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(AttackCreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerDesc& data)
    {
        AttackerDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_Attacker_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(AttackerDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorDesc& data)
    {
        InjectorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Injector_GeneIndex, data._geneIndex, defaultObject._geneIndex);
    }
    SPLIT_SERIALIZATION(InjectorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoBendingDesc& data)
    {
        AutoBendingDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleMode_AutoBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        scope.addMember(Id_MuscleMode_AutoBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        scope.addMember(Id_MuscleMode_AutoBending_InitialAngle, data._initialAngle, defaultObject._initialAngle);
        scope.addMember(Id_MuscleMode_AutoBending_Forward, data._forward, defaultObject._forward);
    }
    SPLIT_SERIALIZATION(AutoBendingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualBendingDesc& data)
    {
        ManualBendingDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleMode_ManualBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        scope.addMember(Id_MuscleMode_ManualBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        scope.addMember(Id_MuscleMode_ManualBending_InitialAngle, data._initialAngle, defaultObject._initialAngle);
        scope.addMember(Id_MuscleMode_ManualBending_LastAngleDelta, data._lastAngleDelta, defaultObject._lastAngleDelta);
    }
    SPLIT_SERIALIZATION(ManualBendingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AngleBendingDesc& data)
    {
        AngleBendingDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleMode_AngleBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        scope.addMember(Id_MuscleMode_AngleBending_AttractionRepulsionRatio, data._attractionRepulsionRatio, defaultObject._attractionRepulsionRatio);
        scope.addMember(Id_MuscleMode_AngleBending_InitialAngle, data._initialAngle, defaultObject._initialAngle);
    }
    SPLIT_SERIALIZATION(AngleBendingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoCrawlingDesc& data)
    {
        AutoCrawlingDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleMode_AutoCrawling_MaxAngleDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        scope.addMember(Id_MuscleMode_AutoCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        scope.addMember(Id_MuscleMode_AutoCrawling_InitialDistance, data._initialDistance, defaultObject._initialDistance);
        scope.addMember(Id_MuscleMode_AutoCrawling_LastActualDistance, data._lastActualDistance, defaultObject._lastActualDistance);
        scope.addMember(Id_MuscleMode_AutoCrawling_Forward, data._forward, defaultObject._forward);
    }
    SPLIT_SERIALIZATION(AutoCrawlingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualCrawlingDesc& data)
    {
        ManualCrawlingDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleMode_ManualCrawling_MaxAngleDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        scope.addMember(Id_MuscleMode_ManualCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
        scope.addMember(Id_MuscleMode_ManualCrawling_InitialDistance, data._initialDistance, defaultObject._initialDistance);
        scope.addMember(Id_MuscleMode_ManualCrawling_LastActualDistance, data._lastActualDistance, defaultObject._lastActualDistance);
        scope.addMember(Id_MuscleMode_ManualCrawling_LastDistanceDelta, data._lastDistanceDelta, defaultObject._lastDistanceDelta);
    }
    SPLIT_SERIALIZATION(ManualCrawlingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DirectMovementDesc& data)
    {
        DirectMovementDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(DirectMovementDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleDesc& data)
    {
        MuscleDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Muscle_LastMovementX, data._lastMovementX, defaultObject._lastMovementX);
        scope.addMember(Id_Muscle_LastMovementY, data._lastMovementY, defaultObject._lastMovementY);
        scope.addDesc(Id_Muscle_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(MuscleDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderDesc& data)
    {
        DefenderDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Defender_Mode, data._mode, defaultObject._mode);
    }
    SPLIT_SERIALIZATION(DefenderDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectSolidDesc& data)
    {
        ReconnectSolidDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(ReconnectSolidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectFreeCellDesc& data)
    {
        ReconnectFreeCellDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ReconnectorMode_FreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    SPLIT_SERIALIZATION(ReconnectFreeCellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectCreatureDesc& data)
    {
        ReconnectCreatureDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ReconnectorMode_Creature_MinNumCells, data._minNumCells, defaultObject._minNumCells);
        scope.addMember(Id_ReconnectorMode_Creature_MaxNumCells, data._maxNumCells, defaultObject._maxNumCells);
        scope.addMember(Id_ReconnectorMode_Creature_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
        scope.addMember(Id_ReconnectorMode_Creature_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
    }
    SPLIT_SERIALIZATION(ReconnectCreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorDesc& data)
    {
        ReconnectorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_Reconnector_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(ReconnectorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorDesc& data)
    {
        DetonatorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Detonator_State, data._state, defaultObject._state);
        scope.addMember(Id_Detonator_Countdown, data._countdown, defaultObject._countdown);
    }
    SPLIT_SERIALIZATION(DetonatorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DigestorDesc& data)
    {
        DigestorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Digestor_RawEnergyConductivity, data._rawEnergyConductivity, defaultObject._rawEnergyConductivity);
    }
    SPLIT_SERIALIZATION(DigestorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDelayDesc& data)
    {
        SignalDelayDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalDelay_Delay, data._delay, defaultObject._delay);
        scope.addMember(Id_SignalDelay_NumMemoryEntriesInitialized, data._numSignalEntriesInitialized, defaultObject._numSignalEntriesInitialized);
        scope.addMember(Id_SignalDelay_RingBufferIndex, data._ringBufferIndex, defaultObject._ringBufferIndex);
    }
    SPLIT_SERIALIZATION(SignalDelayDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRecorderDesc& data)
    {
        SignalRecorderDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalRecorder_ReadOnly, data._readOnly, defaultObject._readOnly);
        scope.addMember(Id_SignalRecorder_State, data._state, defaultObject._state);
        scope.addMember(Id_SignalRecorder_NumSavedSignalEntries, data._numWrittenSignalEntries, defaultObject._numWrittenSignalEntries);
        scope.addMember(Id_SignalRecorder_NumReadSignalEntries, data._numReadSignalEntries, defaultObject._numReadSignalEntries);
    }
    SPLIT_SERIALIZATION(SignalRecorderDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalStorageDesc& data)
    {
        SignalStorageDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalStorage_ReadOnly, data._readOnly, defaultObject._readOnly);
    }
    SPLIT_SERIALIZATION(SignalStorageDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalIntegratorDesc& data)
    {
        SignalIntegratorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalIntegrator_NewSignalWeight, data._newSignalWeight, defaultObject._newSignalWeight);
    }
    SPLIT_SERIALIZATION(SignalIntegratorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalEntryDesc& data)
    {
        SignalEntryDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalEntry_Channels, data._channels, defaultObject._channels);
    }
    SPLIT_SERIALIZATION(SignalEntryDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MemoryDesc& data)
    {
        MemoryDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Memory_ChannelBitMask, data._channelBitMask, defaultObject._channelBitMask);
        scope.addDesc(Id_Memory_Mode, data._mode);
        scope.addDesc(Id_Memory_SignalEntries, data._signalEntries);
    }
    SPLIT_SERIALIZATION(MemoryDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SenderDesc& data)
    {
        SenderDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Sender_Range, data._range, defaultObject._range);
        scope.addMember(Id_Sender_MaxTimesSent, data._maxTimesSent, defaultObject._maxTimesSent);
    }
    SPLIT_SERIALIZATION(SenderDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReceiverDesc& data)
    {
        ReceiverDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Receiver_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
        scope.addMember(Id_Receiver_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
    }
    SPLIT_SERIALIZATION(ReceiverDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CommunicatorDesc& data)
    {
        CommunicatorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_Communicator_Mode, data._mode);
    }
    SPLIT_SERIALIZATION(CommunicatorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, VoidDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    SPLIT_SERIALIZATION(VoidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SolidDesc& data)
    {
        SolidDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Solid_Energy, data._energy, defaultObject._energy);
    }
    SPLIT_SERIALIZATION(SolidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, FluidDesc& data)
    {
        FluidDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Fluid_Energy, data._energy, defaultObject._energy);
        scope.addMember(Id_Fluid_Glow, data._glow, defaultObject._glow);
    }
    SPLIT_SERIALIZATION(FluidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, FreeCellDesc& data)
    {
        FreeCellDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_FreeCell_Energy, data._energy, defaultObject._energy);
        scope.addMember(Id_FreeCell_Age, data._age, defaultObject._age);
    }
    SPLIT_SERIALIZATION(FreeCellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellDesc& data)
    {
        CellDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Cell_UsableEnergy, data._usableEnergy, defaultObject._usableEnergy);
        scope.addMember(Id_Cell_RawEnergy, data._rawEnergy, defaultObject._rawEnergy);
        scope.addMember(Id_Cell_AngleToFront, data._frontAngle, defaultObject._frontAngle);
        scope.addMember(Id_Cell_Age, data._age, defaultObject._age);
        scope.addMember(Id_Cell_CellState, data._cellState, defaultObject._cellState);
        scope.addMember(Id_Cell_ActivationTime, data._activationTime, defaultObject._activationTime);
        scope.addMember(Id_Cell_NodeIndex, data._nodeIndex, defaultObject._nodeIndex);
        scope.addMember(Id_Cell_ParentNodeIndex, data._parentNodeIndex, defaultObject._parentNodeIndex);
        scope.addMember(Id_Cell_ConcatenationIndex, data._concatenationIndex, defaultObject._concatenationIndex);
        scope.addMember(Id_Cell_BranchIndex, data._branchIndex, defaultObject._branchIndex);
        scope.addMember(Id_Cell_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        scope.addMember(Id_Cell_HeadUpdateId, data._headUpdateId, defaultObject._headUpdateId);
        scope.addMember(Id_Cell_HeadCell, data._headCell, defaultObject._headCell);
        scope.addMember(Id_Cell_CreatureId, data._creatureId, defaultObject._creatureId);
        scope.addMember(Id_Cell_Event, data._event, defaultObject._event);
        scope.addMember(Id_Cell_EventCounter, data._eventCounter, defaultObject._eventCounter);
        scope.addMember(Id_Cell_EventPos, data._eventPos, defaultObject._eventPos);
        scope.addMember(Id_Cell_LastUpdate, data._lastUpdate, defaultObject._lastUpdate);
        scope.addDesc(Id_Cell_CellType, data._cellType);
        scope.addDesc(Id_Cell_Constructor, data._constructor);
        scope.addDesc(Id_Cell_Signal, data._signal);
        scope.addDesc(Id_Cell_NeuralNetwork, data._neuralNetwork);
    }
    SPLIT_SERIALIZATION(CellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ObjectDesc& data)
    {
        ObjectDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Object_Id, data._id, defaultObject._id);
        scope.addMember(Id_Object_Pos, data._pos, defaultObject._pos);
        scope.addMember(Id_Object_Vel, data._vel, defaultObject._vel);
        scope.addMember(Id_Object_Stiffness, data._stiffness, defaultObject._stiffness);
        scope.addMember(Id_Object_Color, data._color, defaultObject._color);
        scope.addMember(Id_Object_Fixed, data._fixed, defaultObject._fixed);
        scope.addMember(Id_Object_Sticky, data._sticky, defaultObject._sticky);
        scope.addDesc(Id_Object_Connections, data._connections);
        scope.addDesc(Id_Object_Type, data._type);
    }
    SPLIT_SERIALIZATION(ObjectDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CreatureDesc& data)
    {
        CreatureDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Creature_Id, data._id, defaultObject._id);
        scope.addMember(Id_Creature_AncestorId, data._ancestorId, defaultObject._ancestorId);
        scope.addMember(Id_Creature_Generation, data._generation, defaultObject._generation);
        scope.addMember(Id_Creature_NumCells, data._numCells, defaultObject._numCells);
        scope.addMember(Id_Creature_HeadUpdateId, data._headUpdateId, defaultObject._headUpdateId);
        scope.addMember(Id_Creature_GenomeId, data._genomeId, defaultObject._genomeId);
        scope.addMember(Id_Creature_MutationState, data._mutationState, defaultObject._mutationState);
    }
    SPLIT_SERIALIZATION(CreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, EnergyDesc& data)
    {
        EnergyDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Particle_Id, data._id, defaultObject._id);
        scope.addMember(Id_Particle_Pos, data._pos, defaultObject._pos);
        scope.addMember(Id_Particle_Vel, data._vel, defaultObject._vel);
        scope.addMember(Id_Particle_Energy, data._energy, defaultObject._energy);
        scope.addMember(Id_Particle_Color, data._color, defaultObject._color);
    }
    SPLIT_SERIALIZATION(EnergyDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, Desc& description)
    {
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_Desc_Objects, description._objects);
        scope.addDesc(Id_Desc_Energies, description._energies);
        scope.addDesc(Id_Desc_Creatures, description._creatures);
        scope.addDesc(Id_Desc_Genomes, description._genomes);
    }
    SPLIT_SERIALIZATION(Desc)
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
        if (startIndex + MAX_COLORS < serializedData.size()) {
            dataPoint.summedValues = std::stod(serializedData.at(startIndex + MAX_COLORS));
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
