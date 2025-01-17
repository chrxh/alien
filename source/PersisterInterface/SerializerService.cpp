#include "SerializerService.h"

#include <sstream>
#include <stdexcept>
#include <filesystem>

#include <optional>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/variant.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/adaptors.hpp>
#include <zstr.hpp>

#include "Base/LoggingService.h"
#include "Base/Resources.h"
#include "Base/VersionParserService.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GenomeConstants.h"
#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/GenomeDescriptionService.h"

#include "SettingsParserService.h"

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
    auto constexpr Id_GenomeHeader_Shape = 0;
    auto constexpr Id_GenomeHeader_SeparateConstruction = 1;
    auto constexpr Id_GenomeHeader_AngleAlignment = 2;
    auto constexpr Id_GenomeHeader_Stiffness = 3;
    auto constexpr Id_GenomeHeader_ConnectionDistance = 4;
    auto constexpr Id_GenomeHeader_NumRepetitions = 5;
    auto constexpr Id_GenomeHeader_ConcatenationAngle1 = 6;
    auto constexpr Id_GenomeHeader_ConcatenationAngle2 = 7;
    auto constexpr Id_GenomeHeader_NumBranches = 8;

    auto constexpr Id_SignalRoutingGenome_Active = 0;
    auto constexpr Id_SignalRoutingGenome_BaseAngle = 1;
    auto constexpr Id_SignalRoutingGenome_OpeneningAngle = 2;

    auto constexpr Id_CellGenome_ReferenceAngle = 0;
    auto constexpr Id_CellGenome_Energy = 1;
    auto constexpr Id_CellGenome_Color = 2;
    auto constexpr Id_CellGenome_NumRequiredAdditionalConnections = 3;

    auto constexpr Id_NeuralNetworkGenome_Weights = 0;
    auto constexpr Id_NeuralNetworkGenome_Biases = 1;
    auto constexpr Id_NeuralNetworkGenome_ActivationFunctions = 2;

    auto constexpr Id_TransmitterGenome_Mode = 0;

    auto constexpr Id_ConstructorGenome_Mode = 0;
    auto constexpr Id_ConstructorGenome_ConstructionActivationTime = 1;
    auto constexpr Id_ConstructorGenome_ConstructionAngle1 = 2;
    auto constexpr Id_ConstructorGenome_ConstructionAngle2 = 3;

    auto constexpr Id_DefenderGenome_Mode = 0;

    auto constexpr Id_MuscleGenome_Mode = 0;

    auto constexpr Id_InjectorGenome_Mode = 0;

    auto constexpr Id_AttackerGenome_Mode = 0;

    auto constexpr Id_OscillatorGenome_PulseMode = 0;
    auto constexpr Id_OscillatorGenome_AlternationMode = 1;

    auto constexpr Id_SensorGenome_MinDensity = 0;
    auto constexpr Id_SensorGenome_RestrictToColor = 1;
    auto constexpr Id_SensorGenome_RestrictToMutants = 2;
    auto constexpr Id_SensorGenome_MinRange = 3;
    auto constexpr Id_SensorGenome_MaxRange = 4;

    auto constexpr Id_ReconnectorGenome_RestrictToColor = 0;
    auto constexpr Id_ReconnectorGenome_RestrictToMutants = 1;

    auto constexpr Id_DetonatorGenome_Countdown = 0;

    auto constexpr Id_Particle_Color = 0;

    auto constexpr Id_Cell_Id = 0;
    auto constexpr Id_Cell_Energy = 1;
    auto constexpr Id_Cell_Pos = 2;
    auto constexpr Id_Cell_Vel = 3;
    auto constexpr Id_Cell_Stiffness = 4;
    auto constexpr Id_Cell_Color = 5;
    auto constexpr Id_Cell_Barrier = 6;
    auto constexpr Id_Cell_Age = 7;
    auto constexpr Id_Cell_LivingState = 8;
    auto constexpr Id_Cell_ActivationTime = 9;
    auto constexpr Id_Cell_CreatureId = 10;
    auto constexpr Id_Cell_MutationId = 11;
    auto constexpr Id_Cell_CellTypeUsed = 12;
    auto constexpr Id_Cell_AncestorMutationId = 13;
    auto constexpr Id_Cell_GenomeComplexity = 14;
    auto constexpr Id_Cell_DetectedByCreatureId = 15;
    auto constexpr Id_Cell_GenomeNodeIndex = 20;

    auto constexpr Id_Signal_Channels = 0;
    auto constexpr Id_Signal_Origin = 1;
    auto constexpr Id_Signal_TargetX = 2;
    auto constexpr Id_Signal_TargetY = 3;

    auto constexpr Id_SignalRouting_Active = 0;
    auto constexpr Id_SignalRouting_BaseAngle = 1;
    auto constexpr Id_SignalRouting_OpeneningAngle = 2;

    auto constexpr Id_Connection_CellId = 0;
    auto constexpr Id_Connection_Distance = 1;
    auto constexpr Id_Connection_AngleFromPrevious = 2;

    auto constexpr Id_Metadata_Name = 0;
    auto constexpr Id_Metadata_Description = 1;

    auto constexpr Id_NeuralNetwork_Weights = 0;
    auto constexpr Id_NeuralNetwork_Biases = 1;
    auto constexpr Id_NeuralNetwork_ActivationFunctions = 2;

    auto constexpr Id_Constructor_ActivationMode = 0;
    auto constexpr Id_Constructor_ConstructionActivationTime = 1;
    auto constexpr Id_Constructor_GenomeCurrentNodeIndex = 2;
    auto constexpr Id_Constructor_GenomeGeneration = 3;
    auto constexpr Id_Constructor_ConstructionAngle1 = 4;
    auto constexpr Id_Constructor_ConstructionAngle2 = 5;
    auto constexpr Id_Constructor_OffspringCreatureId = 6;
    auto constexpr Id_Constructor_OffspringMutationId = 7;
    auto constexpr Id_Constructor_GenomeCurrentRepetition = 8;
    auto constexpr Id_Constructor_LastConstructedCellId = 9;
    auto constexpr Id_Constructor_NumInheritedGenomeNodes = 10;
    auto constexpr Id_Constructor_CurrentBranch = 11;

    auto constexpr Id_Defender_Mode = 0;

    auto constexpr Id_Muscle_Mode = 0;
    auto constexpr Id_Muscle_LastBendingDirection = 1;
    auto constexpr Id_Muscle_LastBendingSourceIndex = 2;
    auto constexpr Id_Muscle_ConsecutiveBendingAngle = 3;
    auto constexpr Id_Muscle_LastMovementX = 4;
    auto constexpr Id_Muscle_LastMovementY = 5;

    auto constexpr Id_Injector_Mode = 0;
    auto constexpr Id_Injector_Counter = 1;

    auto constexpr Id_Attacker_Mode = 0;

    auto constexpr Id_Oscillator_PulseMode = 0;
    auto constexpr Id_Oscillator_AlternationMode = 1;

    auto constexpr Id_Sensor_MinDensity = 0;
    auto constexpr Id_Sensor_MemoryChannel1 = 1;
    auto constexpr Id_Sensor_MemoryChannel2 = 2;
    auto constexpr Id_Sensor_MemoryChannel3 = 3;
    auto constexpr Id_Sensor_RestrictToColor = 4;
    auto constexpr Id_Sensor_RestrictToMutants = 5;
    auto constexpr Id_Sensor_TargetX = 6;
    auto constexpr Id_Sensor_TargetY = 7;
    auto constexpr Id_Sensor_MinRange = 8;
    auto constexpr Id_Sensor_MaxRange = 9;

    auto constexpr Id_Transmitter_Mode = 0;

    auto constexpr Id_Reconnector_RestrictToColor = 0;
    auto constexpr Id_Reconnector_RestrictToMutants = 1;

    auto constexpr Id_Detonator_State = 0;
    auto constexpr Id_Detonator_Countdown = 1;

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
        std::optional<uint8_t>,
        std::optional<int8_t>,
        std::optional<int>,
        std::optional<float>,
        std::vector<bool>,
        std::vector<uint8_t>,
        std::vector<int8_t>,
        std::vector<int>,
        std::vector<float>,
        std::vector<std::vector<uint8_t>>,
        std::vector<std::vector<int8_t>>,
        std::vector<std::vector<int>>,
        std::vector<std::vector<float>>
    >;

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

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetworkGenomeDescription& data)
    {
        NeuralNetworkGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_NeuralNetworkGenome_Weights, data.weights, defaultObject.weights);
        loadSave(task, auxiliaries, Id_NeuralNetworkGenome_Biases, data.biases, defaultObject.biases);
        loadSave(task, auxiliaries, Id_NeuralNetworkGenome_ActivationFunctions, data.activationFunctions, defaultObject.activationFunctions);
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
        loadSave(task, auxiliaries, Id_TransmitterGenome_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DepotGenomeDescription)

    template <class Archive>
    void serialize(Archive& ar, MakeGenomeCopy& data)
    {}

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorGenomeDescription& data)
    {
        ConstructorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ConstructorGenome_Mode, data.mode, defaultObject.mode);
        loadSave(task, auxiliaries, Id_ConstructorGenome_ConstructionActivationTime, data.constructionActivationTime, defaultObject.constructionActivationTime);
        loadSave(task, auxiliaries, Id_ConstructorGenome_ConstructionAngle1, data.constructionAngle1, defaultObject.constructionAngle1);
        loadSave(task, auxiliaries, Id_ConstructorGenome_ConstructionAngle2, data.constructionAngle2, defaultObject.constructionAngle2);
        processLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
            ar(genomeData);
            if (std::holds_alternative<MakeGenomeCopy>(genomeData)) {
                data.genome = MakeGenomeCopy();
            } else {
                data.genome = GenomeDescriptionService::get().convertDescriptionToBytes(std::get<GenomeDescription>(genomeData));
            }
        } else {
            std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
            if (std::holds_alternative<MakeGenomeCopy>(data.genome)) {
                genomeData = MakeGenomeCopy();
            } else {
                genomeData = GenomeDescriptionService::get().convertBytesToDescription(std::get<std::vector<uint8_t>>(data.genome));
            }
            ar(genomeData);
        }
    }
    SPLIT_SERIALIZATION(ConstructorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorGenomeDescription& data)
    {
        SensorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SensorGenome_MinDensity, data.minDensity, defaultObject.minDensity);
        loadSave(task, auxiliaries, Id_SensorGenome_RestrictToColor, data.restrictToColor, defaultObject.restrictToColor);
        loadSave(task, auxiliaries, Id_SensorGenome_RestrictToMutants, data.restrictToMutants, defaultObject.restrictToMutants);
        loadSave(task, auxiliaries, Id_SensorGenome_MinRange, data.minRange, defaultObject.minRange);
        loadSave(task, auxiliaries, Id_SensorGenome_MaxRange, data.maxRange, defaultObject.maxRange);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SensorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, OscillatorGenomeDescription& data)
    {
        OscillatorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_OscillatorGenome_PulseMode, data.pulseMode, defaultObject.pulseMode);
        loadSave(task, auxiliaries, Id_OscillatorGenome_AlternationMode, data.alternationMode, defaultObject.alternationMode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(OscillatorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerGenomeDescription& data)
    {
        AttackerGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_AttackerGenome_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackerGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorGenomeDescription& data)
    {
        InjectorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_InjectorGenome_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
            ar(genomeData);
            if (std::holds_alternative<MakeGenomeCopy>(genomeData)) {
                data.genome = MakeGenomeCopy();
            } else {
                data.genome = GenomeDescriptionService::get().convertDescriptionToBytes(std::get<GenomeDescription>(genomeData));
            }
        } else {
            std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
            if (std::holds_alternative<MakeGenomeCopy>(data.genome)) {
                genomeData = MakeGenomeCopy();
            } else {
                genomeData = GenomeDescriptionService::get().convertBytesToDescription(std::get<std::vector<uint8_t>>(data.genome));
            }
            ar(genomeData);
        }
    }
    SPLIT_SERIALIZATION(InjectorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleGenomeDescription& data)
    {
        MuscleGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_MuscleGenome_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(MuscleGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderGenomeDescription& data)
    {
        DefenderGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DefenderGenome_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DefenderGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorGenomeDescription& data)
    {
        ReconnectorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_ReconnectorGenome_RestrictToColor, data.restrictToColor, defaultObject.restrictToColor);
        loadSave(task, auxiliaries, Id_ReconnectorGenome_RestrictToMutants, data.restrictToMutants, defaultObject.restrictToMutants);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorGenomeDescription& data)
    {
        DetonatorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_DetonatorGenome_Countdown, data.countdown, defaultObject.countdown);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetonatorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRoutingRestrictionGenomeDescription& data)
    {
        SignalRoutingRestrictionGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalRoutingGenome_Active, data.active, defaultObject.active);
        loadSave(task, auxiliaries, Id_SignalRoutingGenome_BaseAngle, data.baseAngle, defaultObject.baseAngle);
        loadSave(task, auxiliaries, Id_SignalRoutingGenome_OpeneningAngle, data.openingAngle, defaultObject.openingAngle);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRoutingRestrictionGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellGenomeDescription& data)
    {
        CellGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_CellGenome_ReferenceAngle, data.referenceAngle, defaultObject.referenceAngle);
        loadSave(task, auxiliaries, Id_CellGenome_Energy, data.energy, defaultObject.energy);
        loadSave(task, auxiliaries, Id_CellGenome_Color, data.color, defaultObject.color);
        loadSave(task, auxiliaries, Id_CellGenome_NumRequiredAdditionalConnections, data.numRequiredAdditionalConnections, defaultObject.numRequiredAdditionalConnections);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data.signalRoutingRestriction, data.neuralNetwork, data.cellTypeData);
    }
    SPLIT_SERIALIZATION(CellGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GenomeHeaderDescription& data)
    {
        GenomeHeaderDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_GenomeHeader_Shape, data.shape, defaultObject.shape);
        loadSave(task, auxiliaries, Id_GenomeHeader_NumBranches, data.numBranches, defaultObject.numBranches);
        loadSave(task, auxiliaries, Id_GenomeHeader_SeparateConstruction, data.separateConstruction, defaultObject.separateConstruction);
        loadSave(task, auxiliaries, Id_GenomeHeader_AngleAlignment, data.angleAlignment, defaultObject.angleAlignment);
        loadSave(task, auxiliaries, Id_GenomeHeader_Stiffness, data.stiffness, defaultObject.stiffness);
        loadSave(task, auxiliaries, Id_GenomeHeader_ConnectionDistance, data.connectionDistance, defaultObject.connectionDistance);
        loadSave(task, auxiliaries, Id_GenomeHeader_NumRepetitions, data.numRepetitions, defaultObject.numRepetitions);
        loadSave(task, auxiliaries, Id_GenomeHeader_ConcatenationAngle1, data.concatenationAngle1, defaultObject.concatenationAngle1);
        loadSave(task, auxiliaries, Id_GenomeHeader_ConcatenationAngle2, data.concatenationAngle2, defaultObject.concatenationAngle2);

        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(GenomeHeaderDescription)

    template <class Archive>
    void serialize(Archive& ar, GenomeDescription& data)
    {
        ar(data.header, data.cells);
    }

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellMetadataDescription& data)
    {
        CellMetadataDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Metadata_Name, data.name, defaultObject.name);
        loadSave(task, auxiliaries, Id_Metadata_Description, data.description, defaultObject.description);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(CellMetadataDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConnectionDescription& data)
    {
        ConnectionDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Connection_CellId, data.cellId, defaultObject.cellId);
        loadSave(task, auxiliaries, Id_Connection_Distance, data.distance, defaultObject.distance);
        loadSave(task, auxiliaries, Id_Connection_AngleFromPrevious, data.angleFromPrevious, defaultObject.angleFromPrevious);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ConnectionDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDescription& data)
    {
        SignalDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Signal_Channels, data.channels, defaultObject.channels);
        loadSave(task, auxiliaries, Id_Signal_Origin, data.origin, defaultObject.origin);
        loadSave(task, auxiliaries, Id_Signal_TargetX, data.targetX, defaultObject.targetX);
        loadSave(task, auxiliaries, Id_Signal_TargetY, data.targetY, defaultObject.targetY);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRoutingRestrictionDescription& data)
    {
        SignalRoutingRestrictionDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_SignalRouting_Active, data.active, defaultObject.active);
        loadSave(task, auxiliaries, Id_SignalRouting_BaseAngle, data.baseAngle, defaultObject.baseAngle);
        loadSave(task, auxiliaries, Id_SignalRouting_OpeneningAngle, data.openingAngle, defaultObject.openingAngle);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SignalRoutingRestrictionDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetworkDescription& data)
    {
        NeuralNetworkDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_NeuralNetwork_Weights, data.weights, defaultObject.weights);
        loadSave(task, auxiliaries, Id_NeuralNetwork_Biases, data.biases, defaultObject.biases);
        loadSave(task, auxiliaries, Id_NeuralNetwork_ActivationFunctions, data.activationFunctions, defaultObject.activationFunctions);
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
        loadSave(task, auxiliaries, Id_Transmitter_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DepotDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorDescription& data)
    {
        ConstructorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Constructor_ActivationMode, data.activationMode, defaultObject.activationMode);
        loadSave(task, auxiliaries, Id_Constructor_ConstructionActivationTime, data.constructionActivationTime, defaultObject.constructionActivationTime);
        loadSave(task, auxiliaries, Id_Constructor_LastConstructedCellId, data.lastConstructedCellId, defaultObject.lastConstructedCellId);
        loadSave(task, auxiliaries, Id_Constructor_GenomeCurrentNodeIndex, data.genomeCurrentNodeIndex, defaultObject.genomeCurrentNodeIndex);
        loadSave(task, auxiliaries, Id_Constructor_GenomeCurrentRepetition, data.genomeCurrentRepetition, defaultObject.genomeCurrentRepetition);
        loadSave(task, auxiliaries, Id_Constructor_CurrentBranch, data.currentBranch, defaultObject.currentBranch);
        loadSave(task, auxiliaries, Id_Constructor_OffspringCreatureId, data.offspringCreatureId, defaultObject.offspringCreatureId);
        loadSave(task, auxiliaries, Id_Constructor_OffspringMutationId, data.offspringMutationId, defaultObject.offspringMutationId);
        loadSave(task, auxiliaries, Id_Constructor_GenomeGeneration, data.genomeGeneration, defaultObject.genomeGeneration);
        loadSave(task, auxiliaries, Id_Constructor_ConstructionAngle1, data.constructionAngle1, defaultObject.constructionAngle1);
        loadSave(task, auxiliaries, Id_Constructor_ConstructionAngle2, data.constructionAngle2, defaultObject.constructionAngle2);
        loadSave(task, auxiliaries, Id_Constructor_NumInheritedGenomeNodes, data.numInheritedGenomeNodes, defaultObject.numInheritedGenomeNodes);
        processLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            GenomeDescription genomeDesc;
            ar(genomeDesc);
            data.genome = GenomeDescriptionService::get().convertDescriptionToBytes(genomeDesc);
        } else {
            GenomeDescription genomeDesc = GenomeDescriptionService::get().convertBytesToDescription(data.genome);
            ar(genomeDesc);
        }
    }
    SPLIT_SERIALIZATION(ConstructorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorDescription& data)
    {
        SensorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Sensor_MinDensity, data.minDensity, defaultObject.minDensity);
        loadSave(task, auxiliaries, Id_Sensor_RestrictToColor, data.restrictToColor, defaultObject.restrictToColor);
        loadSave(task, auxiliaries, Id_Sensor_RestrictToMutants, data.restrictToMutants, defaultObject.restrictToMutants);
        loadSave(task, auxiliaries, Id_Sensor_MemoryChannel1, data.memoryChannel1, defaultObject.memoryChannel1);
        loadSave(task, auxiliaries, Id_Sensor_MemoryChannel2, data.memoryChannel2, defaultObject.memoryChannel2);
        loadSave(task, auxiliaries, Id_Sensor_MemoryChannel3, data.memoryChannel3, defaultObject.memoryChannel3);
        loadSave(task, auxiliaries, Id_Sensor_TargetX, data.memoryTargetX, defaultObject.memoryTargetX);
        loadSave(task, auxiliaries, Id_Sensor_TargetY, data.memoryTargetY, defaultObject.memoryTargetY);
        loadSave(task, auxiliaries, Id_Sensor_MinRange, data.minRange, defaultObject.minRange);
        loadSave(task, auxiliaries, Id_Sensor_MaxRange, data.maxRange, defaultObject.maxRange);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SensorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, OscillatorDescription& data)
    {
        OscillatorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Oscillator_PulseMode, data.pulseMode, defaultObject.pulseMode);
        loadSave(task, auxiliaries, Id_Oscillator_AlternationMode, data.alternationMode, defaultObject.alternationMode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(OscillatorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerDescription& data)
    {
        AttackerDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Attacker_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackerDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorDescription& data)
    {
        InjectorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Injector_Mode, data.mode, defaultObject.mode);
        loadSave(task, auxiliaries, Id_Injector_Counter, data.counter, defaultObject.counter);
        processLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            GenomeDescription genomeDesc;
            ar(genomeDesc);
            data.genome = GenomeDescriptionService::get().convertDescriptionToBytes(genomeDesc);
        } else {
            GenomeDescription genomeDesc = GenomeDescriptionService::get().convertBytesToDescription(data.genome);
            ar(genomeDesc);
        }
    }
    SPLIT_SERIALIZATION(InjectorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleDescription& data)
    {
        MuscleDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Muscle_Mode, data.mode, defaultObject.mode);
        loadSave(task, auxiliaries, Id_Muscle_LastBendingDirection, data.lastBendingDirection, defaultObject.lastBendingDirection);
        loadSave(task, auxiliaries, Id_Muscle_LastBendingSourceIndex, data.lastBendingSourceIndex, defaultObject.lastBendingSourceIndex);
        loadSave(task, auxiliaries, Id_Muscle_ConsecutiveBendingAngle, data.consecutiveBendingAngle, defaultObject.consecutiveBendingAngle);
        loadSave(task, auxiliaries, Id_Muscle_LastMovementX, data.lastMovementX, defaultObject.lastMovementX);
        loadSave(task, auxiliaries, Id_Muscle_LastMovementY, data.lastMovementY, defaultObject.lastMovementY);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(MuscleDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderDescription& data)
    {
        DefenderDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Defender_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DefenderDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorDescription& data)
    {
        ReconnectorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Reconnector_RestrictToColor, data.restrictToColor, defaultObject.restrictToColor);
        loadSave(task, auxiliaries, Id_Reconnector_RestrictToMutants, data.restrictToMutants, defaultObject.restrictToMutants);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorDescription& data)
    {
        DetonatorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Detonator_State, data.state, defaultObject.state);
        loadSave(task, auxiliaries, Id_Detonator_Countdown, data.countdown, defaultObject.countdown);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetonatorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellDescription& data)
    {
        CellDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Cell_Id, data.id, defaultObject.id);
        loadSave(task, auxiliaries, Id_Cell_Energy, data.energy, defaultObject.energy);
        loadSave(task, auxiliaries, Id_Cell_Pos, data.pos, defaultObject.pos);
        loadSave(task, auxiliaries, Id_Cell_Vel, data.vel, defaultObject.vel);
        loadSave(task, auxiliaries, Id_Cell_Stiffness, data.stiffness, defaultObject.stiffness);
        loadSave(task, auxiliaries, Id_Cell_Color, data.color, defaultObject.color);
        loadSave(task, auxiliaries, Id_Cell_Barrier, data.barrier, defaultObject.barrier);
        loadSave(task, auxiliaries, Id_Cell_Age, data.age, defaultObject.age);
        loadSave(task, auxiliaries, Id_Cell_LivingState, data.livingState, defaultObject.livingState);
        loadSave(task, auxiliaries, Id_Cell_CreatureId, data.creatureId, defaultObject.creatureId);
        loadSave(task, auxiliaries, Id_Cell_MutationId, data.mutationId, defaultObject.mutationId);
        loadSave(task, auxiliaries, Id_Cell_AncestorMutationId, data.ancestorMutationId, defaultObject.ancestorMutationId);
        loadSave(task, auxiliaries, Id_Cell_ActivationTime, data.activationTime, defaultObject.activationTime);
        loadSave(task, auxiliaries, Id_Cell_GenomeComplexity, data.genomeComplexity, defaultObject.genomeComplexity);
        loadSave(task, auxiliaries, Id_Cell_DetectedByCreatureId, data.detectedByCreatureId, defaultObject.detectedByCreatureId);
        loadSave(task, auxiliaries, Id_Cell_CellTypeUsed, data.cellTypeUsed, defaultObject.cellTypeUsed);
        loadSave(task, auxiliaries, Id_Cell_GenomeNodeIndex, data.genomeNodeIndex, defaultObject.genomeNodeIndex);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data.connections, data.cellTypeData, data.signal, data.signalRoutingRestriction, data.neuralNetwork, data.metadata);
    }
    SPLIT_SERIALIZATION(CellDescription)

    template <class Archive>
    void serialize(Archive& ar, ClusterDescription& data)
    {
        ar(data.cells);
    }

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ParticleDescription& data)
    {
        ParticleDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave(task, auxiliaries, Id_Particle_Color, data.color, defaultObject.color);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data.id, data.pos, data.vel, data.energy);
    }
    SPLIT_SERIALIZATION(ParticleDescription)

    template <class Archive>
    void serialize(Archive& ar, ClusteredDataDescription& data)
    {
        ar(data.clusters, data.particles);
    }
}

bool SerializerService::serializeSimulationToFiles(std::filesystem::path const& filename, DeserializedSimulation const& data)
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
            serializeDataDescription(data.mainData, stream);
        }
        {
            std::ofstream stream(settingsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeAuxiliaryData(data.auxiliaryData, stream);
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

bool SerializerService::deserializeSimulationFromFiles(DeserializedSimulation& data, std::filesystem::path const& filename)
{
    try {
        log(Priority::Important, "load simulation from " + filename.string());
        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));
        std::filesystem::path statisticsFilename(filename);
        statisticsFilename.replace_extension(std::filesystem::path(".statistics.csv"));

        if (!deserializeDataDescription(data.mainData, filename)) {
            return false;
        }
        {
            std::ifstream stream(settingsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeAuxiliaryData(data.auxiliaryData, stream);
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

bool SerializerService::deleteSimulation(std::filesystem::path const& filename)
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

bool SerializerService::serializeSimulationToStrings(SerializedSimulation& output, DeserializedSimulation const& input)
{
    try {
        {
            std::stringstream stdStream;
            zstr::ostream stream(stdStream, std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeDataDescription(input.mainData, stream);
            stream.flush();
            output.mainData = stdStream.str();
        }
        {
            std::stringstream stream;
            serializeAuxiliaryData(input.auxiliaryData, stream);
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

bool SerializerService::deserializeSimulationFromStrings(DeserializedSimulation& output, SerializedSimulation const& input)
{
    try {
        {
            std::stringstream stdStream(input.mainData);
            zstr::istream stream(stdStream, std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeDataDescription(output.mainData, stream);
        }
        {
            std::stringstream stream(input.auxiliaryData);
            deserializeAuxiliaryData(output.auxiliaryData, stream);
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

bool SerializerService::serializeGenomeToFile(std::filesystem::path const& filename, std::vector<uint8_t> const& genome)
{
    try {
        log(Priority::Important, "save genome to " + filename.string());
        //wrap constructor cell around genome
        ClusteredDataDescription data;
        if (!wrapGenome(data, genome)) {
            return false;
        }

        zstr::ofstream stream(filename.string(), std::ios::binary);
        if (!stream) {
            return false;
        }
        serializeDataDescription(data, stream);

        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeGenomeFromFile(std::vector<uint8_t>& genome, std::filesystem::path const& filename)
{
    try {
        log(Priority::Important, "load genome from " + filename.string());
        ClusteredDataDescription data;
        if (!deserializeDataDescription(data, filename)) {
            return false;
        }
        if (!unwrapGenome(genome, data)) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeGenomeToString(std::string& output, std::vector<uint8_t> const& input)
{
    try {
        std::stringstream stdStream;
        zstr::ostream stream(stdStream, std::ios::binary);
        if (!stream) {
            return false;
        }

        ClusteredDataDescription data;
        if (!wrapGenome(data, input)) {
            return false;
        }

        serializeDataDescription(data, stream);
        stream.flush();
        output = stdStream.str();
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeGenomeFromString(std::vector<uint8_t>& output, std::string const& input)
{
    try {
        std::stringstream stdStream(input);
        zstr::istream stream(stdStream, std::ios::binary);
        if (!stream) {
            return false;
        }

        ClusteredDataDescription data;
        deserializeDataDescription(data, stream);

        if (!unwrapGenome(output, data)) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::serializeSimulationParametersToFile(std::filesystem::path const& filename, SimulationParameters const& parameters)
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

bool SerializerService::deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::filesystem::path const& filename)
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

bool SerializerService::serializeStatisticsToFile(std::filesystem::path const& filename, StatisticsHistoryData const& statistics)
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

bool SerializerService::serializeContentToFile(std::filesystem::path const& filename, ClusteredDataDescription const& content)
{
    try {
        zstr::ofstream fileStream(filename.string(), std::ios::binary);
        if (!fileStream) {
            return false;
        }
        serializeDataDescription(content, fileStream);

        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeContentFromFile(ClusteredDataDescription& content, std::filesystem::path const& filename)
{
    try {
        if (!deserializeDataDescription(content, filename)) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

void SerializerService::serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream)
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(Const::ProgramVersion);
    archive(data);
}

bool SerializerService::deserializeDataDescription(ClusteredDataDescription& data, std::filesystem::path const& filename)
{
    zstr::ifstream stream(filename.string(), std::ios::binary);
    if (!stream) {
        return false;
    }
    deserializeDataDescription(data, stream);
    return true;
}

void SerializerService::deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream)
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

void SerializerService::serializeAuxiliaryData(SettingsForSerialization const& auxiliaryData, std::ostream& stream)
{
    boost::property_tree::json_parser::write_json(stream, SettingsParserService::get().encodeAuxiliaryData(auxiliaryData));
}

void SerializerService::deserializeAuxiliaryData(SettingsForSerialization& auxiliaryData, std::istream& stream)
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    auxiliaryData = SettingsParserService::get().decodeAuxiliaryData(tree);
}

void SerializerService::serializeSimulationParameters(SimulationParameters const& parameters, std::ostream& stream)
{
    boost::property_tree::json_parser::write_json(stream, SettingsParserService::get().encodeSimulationParameters(parameters));
}

void SerializerService::deserializeSimulationParameters(SimulationParameters& parameters, std::istream& stream)
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
        {"Transmitter activities", true},
        {"Defender activities", true},
        {"Injection activities", true},
        {"Completed injections", true},
        {"Oscillator pulses", true},
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
            return &dataPoints.numTransmitterActivities;
        } else if (colIndex == 13) {
            return &dataPoints.numInjectionActivities;
        } else if (colIndex == 14) {
            return &dataPoints.numCompletedInjections;
        } else if (colIndex == 15) {
            return &dataPoints.numOscillatorPulses;
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
            return &dataPoints.averageGenomeComplexity;
        } else if (colIndex == 24) {
            return &dataPoints.maxGenomeComplexityOfColonies;
        } else if (colIndex == 25) {
            return &dataPoints.varianceGenomeComplexity;
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

void SerializerService::serializeStatistics(StatisticsHistoryData const& statistics, std::ostream& stream)
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
    for (auto const& [colName, colorDependent]: ColumnDescriptions) {
        if (index != 0) {
            stream << ", ";
        }
        if (!colorDependent) {
            stream << colName;
        }
        else {
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

void SerializerService::deserializeStatistics(StatisticsHistoryData& statistics, std::istream& stream)
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

bool SerializerService::wrapGenome(ClusteredDataDescription& output, std::vector<uint8_t> const& input)
{
    output.clear();
    output.addCluster(ClusterDescription().addCell(CellDescription().setCellTypeData(ConstructorDescription().setGenome(input))));
    return true;
}


bool SerializerService::unwrapGenome(std::vector<uint8_t>& output, ClusteredDataDescription const& input)
{
    if (input.clusters.size() != 1) {
        return false;
    }
    auto cluster = input.clusters.front();
    if (cluster.cells.size() != 1) {
        return false;
    }
    auto cell = cluster.cells.front();
    if (cell.getCellType() != CellType_Constructor) {
        return false;
    }
    output = std::get<ConstructorDescription>(cell.cellTypeData).genome;
    return true;
}
