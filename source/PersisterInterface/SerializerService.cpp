#include "SerializerService.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <ranges>
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
#include <EngineInterface/ParametersValidationService.h>
#include <EngineInterface/SimulationParameters.h>

#include "SettingsParserService.h"

#include "ZstdStream.h"

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
        std::vector<std::vector<float>>,
        IntVector2D>;

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
    auto constexpr Id_Genome_ResistanceToInjection = 6;
    auto constexpr Id_Genome_ApplyMetaMutations = 7;

    auto constexpr Id_NeuronMutation_NodeProbability = 0;
    auto constexpr Id_NeuronMutation_WeightChangeSigma = 1;
    auto constexpr Id_NeuronMutation_BiasChangeSigma = 2;
    auto constexpr Id_NeuronMutation_ActfnChangeProbability = 3;

    auto constexpr Id_ConnectionMutation_NodeProbability = 0;
    auto constexpr Id_ConnectionMutation_ValueChangeSigma = 1;

    auto constexpr Id_CellTypePropertiesMutation_NodeProbability = 0;
    auto constexpr Id_CellTypePropertiesMutation_ValueChangeSigma = 1;
    auto constexpr Id_CellTypePropertiesMutation_EnumChangeProbability = 2;

    auto constexpr Id_GeometryMutation_GeneProbability = 0;
    auto constexpr Id_GeometryMutation_ValueChangeSigma = 1;
    auto constexpr Id_GeometryMutation_EnumChangeProbability = 2;

    auto constexpr Id_CellTypeModeMutation_NodeProbability = 0;

    auto constexpr Id_CellTypeMutation_NodeProbability = 0;

    auto constexpr Id_VoidMutation_NodeProbability = 0;

    auto constexpr Id_ExtendGeneMutation_GeneProbability = 0;

    auto constexpr Id_AddNodeMutation_NodeProbability = 0;

    auto constexpr Id_TrimGeneMutation_GeneProbability = 0;

    auto constexpr Id_DeleteNodeMutation_NodeProbability = 0;

    auto constexpr Id_DuplicateGeneMutation_GeneProbability = 0;

    auto constexpr Id_DeleteGeneMutation_GeneProbability = 0;

    auto constexpr Id_CopyNodeSectionMutation_GeneProbability = 0;

    auto constexpr Id_MoveNodeSectionMutation_GeneProbability = 0;

    auto constexpr Id_ConstructorMutation_NodeProbability = 0;
    auto constexpr Id_ConstructorMutation_ValueChangeSigma = 1;
    auto constexpr Id_ConstructorMutation_EnumChangeProbability = 2;
    auto constexpr Id_ConstructorMutation_ConstructorToggleProbability = 3;

    auto constexpr Id_Gene_Name = 0;
    auto constexpr Id_Gene_Shape = 1;
    auto constexpr Id_Gene_Stiffness = 5;
    auto constexpr Id_Gene_ConnectionDistance = 6;
    auto constexpr Id_Gene_HomogeneousCellType = 7;

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
    auto constexpr Id_SenderGenome_Oneway = 3;

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
    auto constexpr Id_MutationRates_ExtendGeneMutation = 12;
    auto constexpr Id_MutationRates_AddNodeMutation = 13;
    auto constexpr Id_MutationRates_TrimGeneMutation = 14;
    auto constexpr Id_MutationRates_DeleteNodeMutation = 15;
    auto constexpr Id_MutationRates_DuplicateGeneMutation = 16;
    auto constexpr Id_MutationRates_DeleteGeneMutation = 17;
    auto constexpr Id_MutationRates_CopyNodeSectionMutation = 18;
    auto constexpr Id_MutationRates_MoveNodeSectionMutation = 19;
    auto constexpr Id_MutationRates_GeometryMutation1 = 20;
    auto constexpr Id_MutationRates_GeometryMutation2 = 21;

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
        scope.addMember(Id_SenderGenome_Oneway, data._oneway, defaultObject._oneway);
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
        scope.addMember(Id_Gene_HomogeneousCellType, data._homogeneousCellType, defaultObject._homogeneousCellType);
        scope.addDesc(Id_Gene_Nodes, data._nodes);
    }
    SPLIT_SERIALIZATION(GeneDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuronMutationDesc& data)
    {
        NeuronMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_NeuronMutation_NodeProbability, data._nodeProbability, defaultObject._nodeProbability);
        scope.addMember(Id_NeuronMutation_WeightChangeSigma, data._weightChangeSigma, defaultObject._weightChangeSigma);
        scope.addMember(Id_NeuronMutation_BiasChangeSigma, data._biasChangeSigma, defaultObject._biasChangeSigma);
        scope.addMember(Id_NeuronMutation_ActfnChangeProbability, data._actfnChangeProbability, defaultObject._actfnChangeProbability);
    }
    SPLIT_SERIALIZATION(NeuronMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConnectionMutationDesc& data)
    {
        ConnectionMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ConnectionMutation_NodeProbability, data._nodeProbability, defaultObject._nodeProbability);
        scope.addMember(Id_ConnectionMutation_ValueChangeSigma, data._valueChangeSigma, defaultObject._valueChangeSigma);
    }
    SPLIT_SERIALIZATION(ConnectionMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellTypePropertiesMutationDesc& data)
    {
        CellTypePropertiesMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_CellTypePropertiesMutation_NodeProbability, data._nodeProbability, defaultObject._nodeProbability);
        scope.addMember(Id_CellTypePropertiesMutation_ValueChangeSigma, data._valueChangeSigma, defaultObject._valueChangeSigma);
        scope.addMember(Id_CellTypePropertiesMutation_EnumChangeProbability, data._enumChangeProbability, defaultObject._enumChangeProbability);
    }
    SPLIT_SERIALIZATION(CellTypePropertiesMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GeometryMutationDesc& data)
    {
        GeometryMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeometryMutation_GeneProbability, data._geneProbability, defaultObject._geneProbability);
        scope.addMember(Id_GeometryMutation_ValueChangeSigma, data._valueChangeSigma, defaultObject._valueChangeSigma);
        scope.addMember(Id_GeometryMutation_EnumChangeProbability, data._enumChangeProbability, defaultObject._enumChangeProbability);
    }
    SPLIT_SERIALIZATION(GeometryMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellTypeModeMutationDesc& data)
    {
        CellTypeModeMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_CellTypeModeMutation_NodeProbability, data._nodeProbability, defaultObject._nodeProbability);
    }
    SPLIT_SERIALIZATION(CellTypeModeMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellTypeMutationDesc& data)
    {
        CellTypeMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_CellTypeMutation_NodeProbability, data._nodeProbability, defaultObject._nodeProbability);
    }
    SPLIT_SERIALIZATION(CellTypeMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, VoidMutationDesc& data)
    {
        VoidMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_VoidMutation_NodeProbability, data._nodeProbability, defaultObject._nodeProbability);
    }
    SPLIT_SERIALIZATION(VoidMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ExtendGeneMutationDesc& data)
    {
        ExtendGeneMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ExtendGeneMutation_GeneProbability, data._geneProbability, defaultObject._geneProbability);
    }
    SPLIT_SERIALIZATION(ExtendGeneMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AddNodeMutationDesc& data)
    {
        AddNodeMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_AddNodeMutation_NodeProbability, data._nodeProbability, defaultObject._nodeProbability);
    }
    SPLIT_SERIALIZATION(AddNodeMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TrimGeneMutationDesc& data)
    {
        TrimGeneMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_TrimGeneMutation_GeneProbability, data._geneProbability, defaultObject._geneProbability);
    }
    SPLIT_SERIALIZATION(TrimGeneMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DeleteNodeMutationDesc& data)
    {
        DeleteNodeMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DeleteNodeMutation_NodeProbability, data._nodeProbability, defaultObject._nodeProbability);
    }
    SPLIT_SERIALIZATION(DeleteNodeMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DuplicateGeneMutationDesc& data)
    {
        DuplicateGeneMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DuplicateGeneMutation_GeneProbability, data._geneProbability, defaultObject._geneProbability);
    }
    SPLIT_SERIALIZATION(DuplicateGeneMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DeleteGeneMutationDesc& data)
    {
        DeleteGeneMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DeleteGeneMutation_GeneProbability, data._geneProbability, defaultObject._geneProbability);
    }
    SPLIT_SERIALIZATION(DeleteGeneMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CopyNodeSectionMutationDesc& data)
    {
        CopyNodeSectionMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_CopyNodeSectionMutation_GeneProbability, data._geneProbability, defaultObject._geneProbability);
    }
    SPLIT_SERIALIZATION(CopyNodeSectionMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MoveNodeSectionMutationDesc& data)
    {
        MoveNodeSectionMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MoveNodeSectionMutation_GeneProbability, data._geneProbability, defaultObject._geneProbability);
    }
    SPLIT_SERIALIZATION(MoveNodeSectionMutationDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorMutationDesc& data)
    {
        ConstructorMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ConstructorMutation_NodeProbability, data._nodeProbability, defaultObject._nodeProbability);
        scope.addMember(Id_ConstructorMutation_ValueChangeSigma, data._valueChangeSigma, defaultObject._valueChangeSigma);
        scope.addMember(Id_ConstructorMutation_EnumChangeProbability, data._enumChangeProbability, defaultObject._enumChangeProbability);
        scope.addMember(Id_ConstructorMutation_ConstructorToggleProbability, data._constructorToggleProbability, defaultObject._constructorToggleProbability);
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
        scope.addDesc(Id_MutationRates_GeometryMutation1, data._geometryMutations[0]);
        scope.addDesc(Id_MutationRates_GeometryMutation2, data._geometryMutations[1]);
        scope.addDesc(Id_MutationRates_CellTypeModeMutation, data._cellTypeModeMutation);
        scope.addDesc(Id_MutationRates_CellTypeMutation, data._cellTypeMutation);
        scope.addDesc(Id_MutationRates_VoidMutation, data._voidMutation);
        scope.addDesc(Id_MutationRates_ExtendGeneMutation, data._extendGeneMutation);
        scope.addDesc(Id_MutationRates_AddNodeMutation, data._addNodeMutation);
        scope.addDesc(Id_MutationRates_TrimGeneMutation, data._trimGeneMutation);
        scope.addDesc(Id_MutationRates_DeleteNodeMutation, data._deleteNodeMutation);
        scope.addDesc(Id_MutationRates_DuplicateGeneMutation, data._duplicateGeneMutation);
        scope.addDesc(Id_MutationRates_DeleteGeneMutation, data._deleteGeneMutation);
        scope.addDesc(Id_MutationRates_CopyNodeSectionMutation, data._copyNodeSectionMutation);
        scope.addDesc(Id_MutationRates_MoveNodeSectionMutation, data._moveNodeSectionMutation);
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
        scope.addMember(Id_Genome_FrontAngle, data._frontAngle, defaultObject._frontAngle);
        scope.addMember(Id_Genome_ResistanceToInjection, data._resistanceToInjection, defaultObject._resistanceToInjection);
        scope.addMember(Id_Genome_ApplyMetaMutations, data._applyMetaMutations, defaultObject._applyMetaMutations);
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
    auto constexpr Id_Creature_LineageId = 3;
    auto constexpr Id_Creature_AccumulatedMutations = 9;
    auto constexpr Id_Creature_AccumulatedMutationsInLineage = 10;

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
    auto constexpr Id_Cell_SignalChanges = 25;
    auto constexpr Id_Cell_LastUpdate = 18;
    auto constexpr Id_Cell_ConcatenationIndex = 19;
    auto constexpr Id_Cell_BranchIndex = 20;

    auto constexpr Id_Object_Id = 0;
    auto constexpr Id_Object_Pos = 2;
    auto constexpr Id_Object_Vel = 3;
    auto constexpr Id_Object_Stiffness = 4;
    auto constexpr Id_Object_Color = 5;
    auto constexpr Id_Object_Static = 6;
    auto constexpr Id_Object_Sticky = 17;

    auto constexpr Id_Signal_Channels = 0;

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
    auto constexpr Id_Sender_Oneway = 3;

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
        scope.addMember(Id_Sender_Oneway, data._oneway, defaultObject._oneway);
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
        scope.addMember(Id_Cell_SignalChanges, data._signalChanges, defaultObject._signalChanges);
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
        scope.addMember(Id_Object_Static, data._isStatic, defaultObject._isStatic);
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
        scope.addMember(Id_Creature_LineageId, data._lineageId, defaultObject._lineageId);
        scope.addMember(Id_Creature_AccumulatedMutations, data._accumulatedMutations, defaultObject._accumulatedMutations);
        scope.addMember(Id_Creature_AccumulatedMutationsInLineage, data._accumulatedMutationsInLineage, defaultObject._accumulatedMutationsInLineage);
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

        zstd::ofstream stream(filename.string(), std::ios::binary, zstd::DefaultCompressionLevel, zstd::recommendedWorkerCount());
        if (!stream) {
            return false;
        }
        serializeSimulation(data, stream);
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeSimulationFromFiles(DeserializedSimulation& data, std::filesystem::path const& filename) const
{
    try {
        log(Priority::Important, "load simulation from " + filename.string());

        zstd::ifstream stream(filename.string(), std::ios::binary);
        if (!stream) {
            return false;
        }
        deserializeSimulation(data, stream);

        ParametersValidationService::get().validateAndCorrect({data.auxiliaryData.worldSize}, data.auxiliaryData.simulationParameters);
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deleteSimulation(std::filesystem::path const& filename) const
{
    try {
        log(Priority::Important, "delete simulation " + filename.string());
        return std::filesystem::remove(filename);
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeSimulationToStrings(SerializedSimulation& output, DeserializedSimulation const& input) const
{
    try {
        std::stringstream stdStream;
        zstd::ostream stream(stdStream, zstd::DefaultCompressionLevel, zstd::recommendedWorkerCount());
        if (!stream) {
            return false;
        }
        serializeSimulation(input, stream);
        stream.flush();
        output.mainData = stdStream.str();
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeSimulationFromStrings(DeserializedSimulation& output, SerializedSimulation const& input) const
{
    try {
        std::stringstream stdStream(input.mainData);
        zstd::istream stream(stdStream);
        if (!stream) {
            return false;
        }
        deserializeSimulation(output, stream);

        ParametersValidationService::get().validateAndCorrect({output.auxiliaryData.worldSize}, output.auxiliaryData.simulationParameters);
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeGenomeToFile(std::filesystem::path const& filename, GenomeDesc const& genome) const
{
    try {
        log(Priority::Important, "save genome to " + filename.string());
        // Wrap constructor cell around genome
        Desc description;
        if (!wrapGenome(description, genome)) {
            return false;
        }

        zstd::ofstream stream(filename.string(), std::ios::binary);
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
        zstd::ostream stream(stdStream);
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
        zstd::istream stream(stdStream);
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
        serializeSettings(parameters, stream);
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
        deserializeSettings(parameters, stream);
        stream.close();
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeContentToFile(std::filesystem::path const& filename, Desc const& content) const
{
    try {
        zstd::ofstream fileStream(filename.string(), std::ios::binary);
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
    zstd::ifstream stream(filename.string(), std::ios::binary);
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

void SerializerService::serializeSettings(SimulationParameters const& parameters, std::ostream& stream) const
{
    boost::property_tree::json_parser::write_json(stream, SettingsParserService::get().encodeSimulationParameters(parameters));
}

void SerializerService::deserializeSettings(SimulationParameters& parameters, std::istream& stream) const
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    parameters = SettingsParserService::get().decodeSimulationParameters(tree);
}

/************************************************************************/
/* Statistics history                                                   */
/************************************************************************/
namespace
{
    auto constexpr Id_StatisticsHistory_Overall = 0;
    auto constexpr Id_StatisticsHistory_Lineages = 1;

    auto constexpr Id_Timeline_Timestep = 1;
    auto constexpr Id_Timeline_SystemClock = 2;

    auto constexpr Id_OverallTimeline_UniqueColorTimelines = 16;
    auto constexpr Id_OverallTimeline_ColorBitsetGroups = 17;

    auto constexpr Id_ColorColumns_NumCreatures = 0;
    auto constexpr Id_ColorColumns_NumGenomes = 1;
    auto constexpr Id_ColorColumns_SumCreatureCells = 2;
    auto constexpr Id_ColorColumns_SumCreatureGenerations = 3;
    auto constexpr Id_ColorColumns_SumGenomeNodes = 4;
    auto constexpr Id_ColorColumns_SumMutationRates = 5;
    auto constexpr Id_ColorColumns_SumCreatureEnergy = 6;
    auto constexpr Id_ColorColumns_NumCreatedCreatures = 7;
    auto constexpr Id_ColorColumns_TotalMutations = 8;
    auto constexpr Id_ColorColumns_TotalAttackedEnergy = 9;
    auto constexpr Id_ColorColumns_TotalMuscleActivity = 10;

    auto constexpr Id_LineageTimeline_ColorBitset = 3;
    auto constexpr Id_LineageTimeline_RepresentativeCellId = 4;
    auto constexpr Id_LineageTimeline_NumCreatures = 5;
    auto constexpr Id_LineageTimeline_NumGenomes = 6;
    auto constexpr Id_LineageTimeline_SumCreatureCells = 7;
    auto constexpr Id_LineageTimeline_SumCreatureGenerations = 8;
    auto constexpr Id_LineageTimeline_SumGenomeNodes = 9;
    auto constexpr Id_LineageTimeline_SumMutationRates = 10;
    auto constexpr Id_LineageTimeline_SumCreatureEnergy = 11;
    auto constexpr Id_LineageTimeline_NumCreatedCreatures = 12;
    auto constexpr Id_LineageTimeline_TotalMutations = 13;
    auto constexpr Id_LineageTimeline_TotalAttackedEnergy = 14;
    auto constexpr Id_LineageTimeline_TotalMuscleActivity = 15;

    // Metric columns are plot statistics and are stored as float to halve the serialized size; the exact values stay double in memory
    struct ColorTimeline
    {
        std::vector<float> numCreatures;
        std::vector<float> numGenomes;
        std::vector<float> sumCreatureCells;
        std::vector<float> sumCreatureGenerations;
        std::vector<float> sumGenomeNodes;
        std::vector<float> sumMutationRates;
        std::vector<float> sumCreatureEnergy;
        std::vector<float> numCreatedCreatures;
        std::vector<float> totalMutations;
        std::vector<float> totalAttackedEnergy;
        std::vector<float> totalMuscleActivity;
    };

    struct OverallTimeline
    {
        std::vector<double> timestep;
        std::vector<double> systemClock;
        std::unordered_map<uint32_t, ColorTimeline> colorTimelines;
    };

    struct LineageTimeline
    {
        std::vector<double> timestep;
        std::vector<double> systemClock;
        std::vector<uint32_t> colorBitset;
        std::vector<uint64_t> representativeCellId;
        std::vector<float> numCreatures;
        std::vector<float> numGenomes;
        std::vector<float> sumCreatureCells;
        std::vector<float> sumCreatureGenerations;
        std::vector<float> sumGenomeNodes;
        std::vector<float> sumMutationRates;
        std::vector<float> sumCreatureEnergy;
        std::vector<float> numCreatedCreatures;
        std::vector<float> totalMutations;
        std::vector<float> totalAttackedEnergy;
        std::vector<float> totalMuscleActivity;
    };

    struct StatisticsTimelines
    {
        OverallTimeline overallTimeline;
        std::unordered_map<uint32_t, LineageTimeline> lineageTimelines;
    };

    struct LineageColumnDesc
    {
        int id;
        std::vector<float> LineageTimeline::* column;
        double LineageDataPoint::* field;
    };
    std::vector<LineageColumnDesc> const LineageColumnDescs = {
        {Id_LineageTimeline_NumCreatures, &LineageTimeline::numCreatures, &LineageDataPoint::numCreatures},
        {Id_LineageTimeline_NumGenomes, &LineageTimeline::numGenomes, &LineageDataPoint::numGenomes},
        {Id_LineageTimeline_SumCreatureCells, &LineageTimeline::sumCreatureCells, &LineageDataPoint::sumCreatureCells},
        {Id_LineageTimeline_SumCreatureGenerations, &LineageTimeline::sumCreatureGenerations, &LineageDataPoint::sumCreatureGenerations},
        {Id_LineageTimeline_SumGenomeNodes, &LineageTimeline::sumGenomeNodes, &LineageDataPoint::sumGenomeNodes},
        {Id_LineageTimeline_SumMutationRates, &LineageTimeline::sumMutationRates, &LineageDataPoint::sumMutationRates},
        {Id_LineageTimeline_SumCreatureEnergy, &LineageTimeline::sumCreatureEnergy, &LineageDataPoint::sumCreatureEnergy},
        {Id_LineageTimeline_NumCreatedCreatures, &LineageTimeline::numCreatedCreatures, &LineageDataPoint::numCreatedCreatures},
        {Id_LineageTimeline_TotalMutations, &LineageTimeline::totalMutations, &LineageDataPoint::totalMutations},
        {Id_LineageTimeline_TotalAttackedEnergy, &LineageTimeline::totalAttackedEnergy, &LineageDataPoint::totalAttackedEnergy},
        {Id_LineageTimeline_TotalMuscleActivity, &LineageTimeline::totalMuscleActivity, &LineageDataPoint::totalMuscleActivity},
    };

    struct ColorColumnDesc
    {
        int id;
        std::vector<float> ColorTimeline::* column;
        double ColorOverallDataPoint::* field;
    };
    std::vector<ColorColumnDesc> const ColorColumnDescs = {
        {Id_ColorColumns_NumCreatures, &ColorTimeline::numCreatures, &ColorOverallDataPoint::numCreatures},
        {Id_ColorColumns_NumGenomes, &ColorTimeline::numGenomes, &ColorOverallDataPoint::numGenomes},
        {Id_ColorColumns_SumCreatureCells, &ColorTimeline::sumCreatureCells, &ColorOverallDataPoint::sumCreatureCells},
        {Id_ColorColumns_SumCreatureGenerations, &ColorTimeline::sumCreatureGenerations, &ColorOverallDataPoint::sumCreatureGenerations},
        {Id_ColorColumns_SumGenomeNodes, &ColorTimeline::sumGenomeNodes, &ColorOverallDataPoint::sumGenomeNodes},
        {Id_ColorColumns_SumMutationRates, &ColorTimeline::sumMutationRates, &ColorOverallDataPoint::sumMutationRates},
        {Id_ColorColumns_SumCreatureEnergy, &ColorTimeline::sumCreatureEnergy, &ColorOverallDataPoint::sumCreatureEnergy},
        {Id_ColorColumns_NumCreatedCreatures, &ColorTimeline::numCreatedCreatures, &ColorOverallDataPoint::numCreatedCreatures},
        {Id_ColorColumns_TotalMutations, &ColorTimeline::totalMutations, &ColorOverallDataPoint::totalMutations},
        {Id_ColorColumns_TotalAttackedEnergy, &ColorTimeline::totalAttackedEnergy, &ColorOverallDataPoint::totalAttackedEnergy},
        {Id_ColorColumns_TotalMuscleActivity, &ColorTimeline::totalMuscleActivity, &ColorOverallDataPoint::totalMuscleActivity},
    };

    bool colorTimelinesEqual(ColorTimeline const& left, ColorTimeline const& right)
    {
        for (auto const& [id, column, field] : ColorColumnDescs) {
            if (left.*column != right.*column) {
                return false;
            }
        }
        return true;
    }

    size_t hashColorTimeline(ColorTimeline const& timeline)
    {
        size_t seed = 0;
        auto combine = [&seed](size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); };
        for (auto const& [id, column, field] : ColorColumnDescs) {
            auto const& values = timeline.*column;
            combine(values.size());
            for (auto const value : values) {
                combine(std::hash<float>{}(value));
            }
        }
        return seed;
    }

    // Many color combinations share the same (often all-zero) ColorTimeline. Storing each distinct timeline once
    // together with the color combinations that map to it saves a lot of space (there are up to 2^colors - 1 combinations).
    struct DeduplicatedColorTimelines
    {
        std::vector<ColorTimeline> uniqueTimelines;
        std::vector<std::vector<uint32_t>> colorBitsetGroups;  // Parallel to uniqueTimelines
    };

    DeduplicatedColorTimelines deduplicateColorTimelines(std::unordered_map<uint32_t, ColorTimeline> const& colorTimelines)
    {
        DeduplicatedColorTimelines result;
        std::unordered_map<size_t, std::vector<size_t>> hashToUniqueIndices;

        // Sort color combinations for deterministic output
        std::vector<uint32_t> sortedColorBitsets;
        sortedColorBitsets.reserve(colorTimelines.size());
        for (auto const& [colorBitset, timeline] : colorTimelines) {
            sortedColorBitsets.emplace_back(colorBitset);
        }
        std::sort(sortedColorBitsets.begin(), sortedColorBitsets.end());

        for (auto const colorBitset : sortedColorBitsets) {
            auto const& timeline = colorTimelines.at(colorBitset);
            auto& candidateIndices = hashToUniqueIndices[hashColorTimeline(timeline)];

            std::optional<size_t> matchIndex;
            for (auto const index : candidateIndices) {
                if (colorTimelinesEqual(result.uniqueTimelines.at(index), timeline)) {
                    matchIndex = index;
                    break;
                }
            }
            if (!matchIndex) {
                matchIndex = result.uniqueTimelines.size();
                result.uniqueTimelines.emplace_back(timeline);
                result.colorBitsetGroups.emplace_back();
                candidateIndices.emplace_back(*matchIndex);
            }
            result.colorBitsetGroups.at(*matchIndex).emplace_back(colorBitset);
        }
        return result;
    }

    std::unordered_map<uint32_t, ColorTimeline> expandColorTimelines(DeduplicatedColorTimelines const& deduplicated)
    {
        std::unordered_map<uint32_t, ColorTimeline> result;
        for (auto&& [timeline, colorBitsets] : std::views::zip(deduplicated.uniqueTimelines, deduplicated.colorBitsetGroups)) {
            for (auto const colorBitset : colorBitsets) {
                result.emplace(colorBitset, timeline);
            }
        }
        return result;
    }

    template <typename Timeline, typename Sample>
    void extractTimingColumns(Timeline& timeline, std::vector<Sample> const& samples)
    {
        timeline.timestep.reserve(samples.size());
        timeline.systemClock.reserve(samples.size());
        for (auto const& sample : samples) {
            timeline.timestep.emplace_back(sample.timestep);
            timeline.systemClock.emplace_back(sample.systemClock);
        }
    }

    template <typename Sample, typename Timeline>
    std::vector<Sample> createSamplesWithTiming(Timeline const& timeline)
    {
        std::vector<Sample> result(timeline.timestep.size());
        for (auto&& [sample, value] : std::views::zip(result, timeline.timestep)) {
            sample.timestep = value;
        }
        for (auto&& [sample, value] : std::views::zip(result, timeline.systemClock)) {
            sample.systemClock = value;
        }
        return result;
    }

    template <typename Timeline, typename Sample, typename Columns>
    void extractDataColumns(Timeline& timeline, std::vector<Sample> const& samples, Columns const& columns)
    {
        for (auto const& [id, column, field] : columns) {
            auto& values = timeline.*column;
            values.reserve(samples.size());
            for (auto const& sample : samples) {
                values.emplace_back(static_cast<float>(sample.data.*field));
            }
        }
    }

    template <typename Sample, typename Timeline, typename Columns>
    void applyDataColumns(std::vector<Sample>& samples, Timeline const& timeline, Columns const& columns)
    {
        for (auto const& [id, column, field] : columns) {
            for (auto&& [sample, value] : std::views::zip(samples, timeline.*column)) {
                sample.data.*field = value;
            }
        }
    }

    OverallTimeline convertToTimeline(std::vector<ColorSamples> const& samples)
    {
        OverallTimeline result;
        extractTimingColumns(result, samples);

        for (auto const& sample : samples) {
            for (auto const& [colorBitset, dataPoint] : sample.data) {
                result.colorTimelines.try_emplace(colorBitset);
            }
        }
        for (auto& [colorBitset, columns] : result.colorTimelines) {
            for (auto const& [id, column, field] : ColorColumnDescs) {
                auto& values = columns.*column;
                values.reserve(samples.size());
                for (auto const& sample : samples) {
                    auto it = sample.data.find(colorBitset);
                    values.emplace_back(static_cast<float>(it != sample.data.end() ? it->second.*field : 0.0));
                }
            }
        }
        return result;
    }

    std::vector<ColorSamples> convertToSamples(OverallTimeline const& timeline)
    {
        auto result = createSamplesWithTiming<ColorSamples>(timeline);

        for (auto const& [colorBitset, columns] : timeline.colorTimelines) {
            for (auto const& [id, column, field] : ColorColumnDescs) {
                for (auto&& [sample, value] : std::views::zip(result, columns.*column)) {
                    sample.data[colorBitset].*field = value;
                }
            }
        }
        return result;
    }

    LineageTimeline convertToTimeline(std::vector<LineageSample> const& samples)
    {
        LineageTimeline result;
        extractTimingColumns(result, samples);
        extractDataColumns(result, samples, LineageColumnDescs);
        result.colorBitset.reserve(samples.size());
        result.representativeCellId.reserve(samples.size());
        for (auto const& sample : samples) {
            result.colorBitset.emplace_back(sample.data.colorBitset);
            result.representativeCellId.emplace_back(sample.data.representativeCellId);
        }
        return result;
    }

    auto constexpr MaxSavedLineages = size_t{250};

    std::unordered_map<uint32_t, double> countCreaturesByLineage(Desc const& mainData)
    {
        std::unordered_map<uint32_t, double> result;
        for (auto const& creature : mainData._creatures) {
            ++result[static_cast<uint32_t>(creature._lineageId)];
        }
        return result;
    }

    // Size of a lineage taken from the objects being saved. The last history sample is only a fallback for lineages
    // that are missing there (e.g. when the statistics are saved on their own): it is averaged over a sampling
    // interval and therefore deviates from the actual creature count.
    double getNumCreatures(
        uint32_t lineageId,
        std::unordered_map<uint32_t, std::vector<LineageSample>> const& lineages,
        std::unordered_map<uint32_t, double> const& numCreaturesByLineage)
    {
        if (auto it = numCreaturesByLineage.find(lineageId); it != numCreaturesByLineage.end()) {
            return it->second;
        }
        auto const& samples = lineages.at(lineageId);
        return samples.empty() ? 0.0 : samples.back().data.numCreatures;
    }

    std::vector<uint32_t> selectLineagesToSave(
        std::unordered_map<uint32_t, std::vector<LineageSample>> const& lineages,
        std::unordered_map<uint32_t, double> const& numCreaturesByLineage)
    {
        std::vector<uint32_t> result;
        result.reserve(lineages.size());
        for (auto const lineageId : std::views::keys(lineages)) {
            result.emplace_back(lineageId);
        }
        if (result.size() > MaxSavedLineages) {
            // The lineage id decides between equally large lineages to keep the output deterministic
            auto isLarger = [&lineages, &numCreaturesByLineage](uint32_t left, uint32_t right) {
                auto leftNumCreatures = getNumCreatures(left, lineages, numCreaturesByLineage);
                auto rightNumCreatures = getNumCreatures(right, lineages, numCreaturesByLineage);
                return leftNumCreatures != rightNumCreatures ? leftNumCreatures > rightNumCreatures : left < right;
            };
            std::ranges::partial_sort(result, result.begin() + MaxSavedLineages, isLarger);
            result.resize(MaxSavedLineages);
        }
        return result;
    }

    std::vector<LineageSample> convertToSamples(LineageTimeline const& timeline)
    {
        auto result = createSamplesWithTiming<LineageSample>(timeline);
        applyDataColumns(result, timeline, LineageColumnDescs);
        for (auto&& [sample, value] : std::views::zip(result, timeline.colorBitset)) {
            sample.data.colorBitset = value;
        }
        for (auto&& [sample, value] : std::views::zip(result, timeline.representativeCellId)) {
            sample.data.representativeCellId = value;
        }
        return result;
    }
}

namespace cereal
{
    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ColorTimeline& data)
    {
        auto scope = getSerializationScope(task, ar);
        for (auto const& [id, column, field] : ColorColumnDescs) {
            scope.addDesc(id, data.*column);
        }
    }
    SPLIT_SERIALIZATION(ColorTimeline)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, OverallTimeline& data)
    {
        DeduplicatedColorTimelines deduplicated;
        if (task == SerializationTask::Save) {
            deduplicated = deduplicateColorTimelines(data.colorTimelines);
        }
        {
            auto scope = getSerializationScope(task, ar);
            scope.addDesc(Id_Timeline_Timestep, data.timestep);
            scope.addDesc(Id_Timeline_SystemClock, data.systemClock);
            scope.addDesc(Id_OverallTimeline_UniqueColorTimelines, deduplicated.uniqueTimelines);
            scope.addDesc(Id_OverallTimeline_ColorBitsetGroups, deduplicated.colorBitsetGroups);
        }
        if (task == SerializationTask::Load) {
            data.colorTimelines = expandColorTimelines(deduplicated);
        }
    }
    SPLIT_SERIALIZATION(OverallTimeline)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, LineageTimeline& data)
    {
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_Timeline_Timestep, data.timestep);
        scope.addDesc(Id_Timeline_SystemClock, data.systemClock);
        scope.addDesc(Id_LineageTimeline_ColorBitset, data.colorBitset);
        scope.addDesc(Id_LineageTimeline_RepresentativeCellId, data.representativeCellId);
        for (auto const& [id, column, field] : LineageColumnDescs) {
            scope.addDesc(id, data.*column);
        }
    }
    SPLIT_SERIALIZATION(LineageTimeline)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, StatisticsTimelines& data)
    {
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_StatisticsHistory_Overall, data.overallTimeline);
        scope.addDesc(Id_StatisticsHistory_Lineages, data.lineageTimelines);
    }
    SPLIT_SERIALIZATION(StatisticsTimelines)
}

/************************************************************************/
/* Simulation                                                           */
/************************************************************************/
namespace
{
    auto constexpr Id_Simulation_Timestep = 100;
    auto constexpr Id_Simulation_RealTime = 101;
    auto constexpr Id_Simulation_Zoom = 102;
    auto constexpr Id_Simulation_Center = 103;
    auto constexpr Id_Simulation_WorldSize = 104;
    auto constexpr Id_Simulation_Statistics = 105;
    auto constexpr Id_Simulation_SimulationParameters = 106;

    StatisticsTimelines convertToTimelines(StatisticsHistoryData const& statistics, Desc const& mainData)
    {
        StatisticsTimelines result;
        result.overallTimeline = convertToTimeline(statistics.colors);
        for (auto const lineageId : selectLineagesToSave(statistics.lineages, countCreaturesByLineage(mainData))) {
            result.lineageTimelines.emplace(lineageId, convertToTimeline(statistics.lineages.at(lineageId)));
        }
        return result;
    }

    StatisticsHistoryData convertToStatisticsHistory(StatisticsTimelines const& timelines)
    {
        StatisticsHistoryData result;
        result.colors = convertToSamples(timelines.overallTimeline);
        for (auto const& [lineageId, timeline] : timelines.lineageTimelines) {
            result.lineages.emplace(lineageId, convertToSamples(timeline));
        }
        return result;
    }
}

namespace cereal
{
    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DeserializedSimulation& data)
    {
        StatisticsTimelines timelines;
        std::string encodedSimulationParameters;
        if (task == SerializationTask::Save) {
            timelines = convertToTimelines(data.statistics, data.mainData);
            encodedSimulationParameters = SettingsParserService::get().encodeSimulationParametersToString(data.auxiliaryData.simulationParameters);
        }
        {
            SettingsForSerialization defaultSettings;
            auto& settings = data.auxiliaryData;
            auto scope = getSerializationScope(task, ar);

            scope.addMember(Id_Simulation_Timestep, settings.timestep, defaultSettings.timestep);
            auto realTimeInMs = static_cast<uint64_t>(settings.realTime.count());
            scope.addMember(Id_Simulation_RealTime, realTimeInMs, static_cast<uint64_t>(defaultSettings.realTime.count()));
            settings.realTime = std::chrono::milliseconds(realTimeInMs);
            scope.addMember(Id_Simulation_Zoom, settings.zoom, defaultSettings.zoom);
            scope.addMember(Id_Simulation_Center, settings.center, defaultSettings.center);
            scope.addMember(Id_Simulation_WorldSize, settings.worldSize, defaultSettings.worldSize);
            scope.addMember(Id_Simulation_SimulationParameters, encodedSimulationParameters, std::string());

            scope.addDesc(Id_Desc_Objects, data.mainData._objects);
            scope.addDesc(Id_Desc_Energies, data.mainData._energies);
            scope.addDesc(Id_Desc_Creatures, data.mainData._creatures);
            scope.addDesc(Id_Desc_Genomes, data.mainData._genomes);
            scope.addDesc(Id_Simulation_Statistics, timelines);
        }
        if (task == SerializationTask::Load) {
            data.statistics = convertToStatisticsHistory(timelines);
            if (!encodedSimulationParameters.empty()) {
                data.auxiliaryData.simulationParameters = SettingsParserService::get().decodeSimulationParametersFromString(encodedSimulationParameters);
            }
        }
    }
    SPLIT_SERIALIZATION(DeserializedSimulation)
}

void SerializerService::serializeSimulation(DeserializedSimulation const& data, std::ostream& stream) const
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(Const::ProgramVersion);
    archive(data);
}

void SerializerService::deserializeSimulation(DeserializedSimulation& data, std::istream& stream) const
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
    archive(data);
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
