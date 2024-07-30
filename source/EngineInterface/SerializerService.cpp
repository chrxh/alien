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
#include "Base/VersionChecker.h"

#include "Descriptions.h"
#include "SimulationParameters.h"
#include "AuxiliaryDataParserService.h"
#include "GenomeConstants.h"
#include "GenomeDescriptions.h"
#include "GenomeDescriptionService.h"

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
    auto constexpr Id_GenomeHeader_SeparateConstruction = 2;
    auto constexpr Id_GenomeHeader_AngleAlignment = 3;
    auto constexpr Id_GenomeHeader_Stiffness = 4;
    auto constexpr Id_GenomeHeader_ConnectionDistance = 5;
    auto constexpr Id_GenomeHeader_NumRepetitions = 6;
    auto constexpr Id_GenomeHeader_ConcatenationAngle1 = 7;
    auto constexpr Id_GenomeHeader_ConcatenationAngle2 = 8;
    auto constexpr Id_GenomeHeader_NumBranches = 9;

    auto constexpr Id_CellGenome_ReferenceAngle = 1;
    auto constexpr Id_CellGenome_Energy = 7;
    auto constexpr Id_CellGenome_Color = 2;
    auto constexpr Id_CellGenome_NumRequiredAdditionalConnections = 9;
    auto constexpr Id_CellGenome_ExecutionOrderNumber = 4;
    auto constexpr Id_CellGenome_InputExecutionOrderNumber = 8;
    auto constexpr Id_CellGenome_OutputBlocked = 6;

    auto constexpr Id_NeuronGenome_ActivationFunctions = 0;

    auto constexpr Id_TransmitterGenome_Mode = 0;

    auto constexpr Id_ConstructorGenome_Mode = 0;
    auto constexpr Id_ConstructorGenome_SingleConstruction = 1;
    auto constexpr Id_ConstructorGenome_SeparateConstruction = 2;
    auto constexpr Id_ConstructorGenome_AngleAlignment = 4;
    auto constexpr Id_ConstructorGenome_Stiffness = 5;
    auto constexpr Id_ConstructorGenome_ConstructionActivationTime = 6;
    auto constexpr Id_ConstructorGenome_GenomeHeader = 8;
    auto constexpr Id_ConstructorGenome_ConstructionAngle1 = 9;
    auto constexpr Id_ConstructorGenome_ConstructionAngle2 = 10;

    auto constexpr Id_DefenderGenome_Mode = 0;

    auto constexpr Id_MuscleGenome_Mode = 0;

    auto constexpr Id_InjectorGenome_Mode = 0;

    auto constexpr Id_AttackerGenome_Mode = 0;

    auto constexpr Id_NerveGenome_PulseMode = 0;
    auto constexpr Id_NerveGenome_AlternationMode = 1;

    auto constexpr Id_SensorGenome_FixedAngle = 0;
    auto constexpr Id_SensorGenome_MinDensity = 1;
    auto constexpr Id_SensorGenome_Color_Deprecated = 2;
    auto constexpr Id_SensorGenome_RestrictToColor = 3;
    auto constexpr Id_SensorGenome_RestrictToOtherMutants_Deprecated = 4;
    auto constexpr Id_SensorGenome_RestrictToMutants = 5;


    auto constexpr Id_ReconnectorGenome_Color_Deprecated = 0;
    auto constexpr Id_ReconnectorGenome_RestrictToColor = 1;
    auto constexpr Id_ReconnectorGenome_RestrictToMutants = 2;

    auto constexpr Id_DetonatorGenome_Countdown = 0;


    auto constexpr Id_Particle_Color = 0;

    auto constexpr Id_Cell_Stiffness = 0;
    auto constexpr Id_Cell_Color = 1;
    auto constexpr Id_Cell_ExecutionOrderNumber = 2;
    auto constexpr Id_Cell_Barrier = 3;
    auto constexpr Id_Cell_Age = 4;
    auto constexpr Id_Cell_LivingState = 5;
    auto constexpr Id_Cell_OutputBlocked = 7;
    auto constexpr Id_Cell_ActivationTime = 8;
    auto constexpr Id_Cell_InputExecutionOrderNumber = 9;
    auto constexpr Id_Cell_CreatureId = 11;
    auto constexpr Id_Cell_MutationId = 12;
    auto constexpr Id_Cell_GenomeComplexity_Deprecated = 13;
    auto constexpr Id_Cell_DetectedByCreatureId = 14;
    auto constexpr Id_Cell_CellFunctionUsed = 15;
    auto constexpr Id_Cell_AncestorMutationId = 16;
    auto constexpr Id_Cell_GenomeComplexity = 17;

    auto constexpr Id_Neuron_ActivationFunctions = 0;

    auto constexpr Id_Constructor_ActivationMode = 0;
    auto constexpr Id_Constructor_SingleConstruction = 1;
    auto constexpr Id_Constructor_SeparateConstruction = 2;
    auto constexpr Id_Constructor_AngleAlignment = 4;
    auto constexpr Id_Constructor_Stiffness = 5;
    auto constexpr Id_Constructor_ConstructionActivationTime = 6;
    auto constexpr Id_Constructor_GenomeCurrentNodeIndex = 7;
    auto constexpr Id_Constructor_GenomeGeneration = 9;
    auto constexpr Id_Constructor_GenomeHeader = 10;
    auto constexpr Id_Constructor_ConstructionAngle1 = 11;
    auto constexpr Id_Constructor_ConstructionAngle2 = 12;
    auto constexpr Id_Constructor_OffspringCreatureId = 13;
    auto constexpr Id_Constructor_OffspringMutationId = 14;
    auto constexpr Id_Constructor_IsConstructionBuilt = 15; //unused
    auto constexpr Id_Constructor_GenomeCurrentRepetition = 16;
    auto constexpr Id_Constructor_LastConstructedCellId = 17;
    auto constexpr Id_Constructor_StateFlags = 18;
    auto constexpr Id_Constructor_NumInheritedGenomeNodes = 19;
    auto constexpr Id_Constructor_CurrentBranch = 20;

    auto constexpr Id_Defender_Mode = 0;

    auto constexpr Id_Muscle_Mode = 0;
    auto constexpr Id_Muscle_LastBendingDirection = 1;
    auto constexpr Id_Muscle_LastBendingSourceIndex = 4;
    auto constexpr Id_Muscle_ConsecutiveBendingAngle = 3;

    auto constexpr Id_Injector_Mode = 0;
    auto constexpr Id_Injector_Counter = 1;
    auto constexpr Id_Injector_GenomeHeader = 2;

    auto constexpr Id_Attacker_Mode = 0;

    auto constexpr Id_Nerve_PulseMode = 0;
    auto constexpr Id_Nerve_AlternationMode = 1;

    auto constexpr Id_Sensor_FixedAngle = 0;
    auto constexpr Id_Sensor_MinDensity = 1;
    auto constexpr Id_Sensor_Color_Deprecated = 2;
    //auto constexpr Id_Sensor_TargetedCreatureId = 3;
    auto constexpr Id_Sensor_MemoryChannel1 = 4;
    auto constexpr Id_Sensor_MemoryChannel2 = 5;
    auto constexpr Id_Sensor_MemoryChannel3 = 6;
    auto constexpr Id_Sensor_RestrictToColor = 7;
    auto constexpr Id_Sensor_RestrictToOtherMutants_Deprecated = 8;
    auto constexpr Id_Sensor_RestrictToMutants = 9;

    auto constexpr Id_Transmitter_Mode = 0;

    auto constexpr Id_Reconnector_Color_Deprecated = 0;
    auto constexpr Id_Reconnector_RestrictToColor = 1;
    auto constexpr Id_Reconnector_RestrictToMutants = 2;

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
    using VariantData = std::variant<int, float, uint64_t, bool, std::optional<float>, std::optional<int>, std::vector<int>, uint32_t, uint8_t>;

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
    void loadSave(SerializationTask task, Archive& ar, NeuronGenomeDescription& data)
    {
        NeuronGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<std::vector<int>>(task, auxiliaries, Id_NeuronGenome_ActivationFunctions, data.activationFunctions, defaultObject.activationFunctions);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data.weights, data.biases);
    }
    SPLIT_SERIALIZATION(NeuronGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TransmitterGenomeDescription& data)
    {
        TransmitterGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_TransmitterGenome_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(TransmitterGenomeDescription)

    template <class Archive>
    void serialize(Archive& ar, MakeGenomeCopy& data)
    {}

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorGenomeDescription& data)
    {
        ConstructorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_ConstructorGenome_Mode, data.mode, defaultObject.mode);
        loadSave<int>(task, auxiliaries, Id_ConstructorGenome_ConstructionActivationTime, data.constructionActivationTime, defaultObject.constructionActivationTime);
        loadSave<float>(task, auxiliaries, Id_ConstructorGenome_ConstructionAngle1, data.constructionAngle1, defaultObject.constructionAngle1);
        loadSave<float>(task, auxiliaries, Id_ConstructorGenome_ConstructionAngle2, data.constructionAngle2, defaultObject.constructionAngle2);
        if (task == SerializationTask::Save) {
            auxiliaries[Id_ConstructorGenome_GenomeHeader] = true;
        }
        processLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            auto hasGenomeHeader = auxiliaries.contains(Id_ConstructorGenome_GenomeHeader);
            if (hasGenomeHeader) {
                std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
                ar(genomeData);
                if (std::holds_alternative<MakeGenomeCopy>(genomeData)) {
                    data.genome = MakeGenomeCopy();
                } else {
                    data.genome = GenomeDescriptionService::convertDescriptionToBytes(std::get<GenomeDescription>(genomeData));
                }
            } else {
                std::variant<MakeGenomeCopy, std::vector<CellGenomeDescription>> genomeData;
                ar(genomeData);
                if (std::holds_alternative<MakeGenomeCopy>(genomeData)) {
                    data.genome = MakeGenomeCopy();
                }

                //compatibility with older versions
                //>>>
                else {
                    GenomeDescription genomeDesc;
                    genomeDesc.cells = std::get<std::vector<CellGenomeDescription>>(genomeData);
                    genomeDesc.header.numBranches =
                        std::get<bool>(auxiliaries.at(Id_ConstructorGenome_SingleConstruction)) ? 1 : 2;
                    genomeDesc.header.separateConstruction = std::get<bool>(auxiliaries.at(Id_ConstructorGenome_SeparateConstruction));
                    genomeDesc.header.angleAlignment = std::get<int>(auxiliaries.at(Id_ConstructorGenome_AngleAlignment));
                    genomeDesc.header.stiffness = std::get<float>(auxiliaries.at(Id_ConstructorGenome_Stiffness));
                    data.genome = GenomeDescriptionService::convertDescriptionToBytes(genomeDesc);
                    if (!genomeDesc.cells.empty()) {
                        data.constructionAngle1 = genomeDesc.cells.front().referenceAngle;
                        data.constructionAngle2 = genomeDesc.cells.back().referenceAngle;
                    }
                }
                //<<<
            }
        } else {
            std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
            if (std::holds_alternative<MakeGenomeCopy>(data.genome)) {
                genomeData = MakeGenomeCopy();
            } else {
                genomeData = GenomeDescriptionService::convertBytesToDescription(std::get<std::vector<uint8_t>>(data.genome));
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
        loadSave<std::optional<float>>(task, auxiliaries, Id_SensorGenome_FixedAngle, data.fixedAngle, defaultObject.fixedAngle);
        loadSave<float>(task, auxiliaries, Id_SensorGenome_MinDensity, data.minDensity, defaultObject.minDensity);
        loadSave<std::optional<int>>(task, auxiliaries, Id_SensorGenome_RestrictToColor, data.restrictToColor, defaultObject.restrictToColor);
        loadSave(task, auxiliaries, Id_SensorGenome_RestrictToMutants, data.restrictToMutants, defaultObject.restrictToMutants);
        processLoadSaveMap(task, ar, auxiliaries);

        //compatibility with older versions
        //>>>
        if (task == SerializationTask::Load) {
            if (auxiliaries.contains(Id_SensorGenome_Color_Deprecated)) {
                data.restrictToColor = std::get<int>(auxiliaries.at(Id_SensorGenome_Color_Deprecated));
            }
            if (auxiliaries.contains(Id_SensorGenome_RestrictToOtherMutants_Deprecated)) {
                auto restrictToOtherMutants = std::get<bool>(auxiliaries.at(Id_SensorGenome_RestrictToOtherMutants_Deprecated));
                data.restrictToMutants = restrictToOtherMutants ? SensorRestrictToMutants_RestrictToNutrient : SensorRestrictToMutants_NoRestriction;
            }
        }
        //<<<
    }
    SPLIT_SERIALIZATION(SensorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NerveGenomeDescription& data)
    {
        NerveGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_NerveGenome_PulseMode, data.pulseMode, defaultObject.pulseMode);
        loadSave<int>(task, auxiliaries, Id_NerveGenome_AlternationMode, data.alternationMode, defaultObject.alternationMode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(NerveGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerGenomeDescription& data)
    {
        AttackerGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_AttackerGenome_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackerGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorGenomeDescription& data)
    {
        InjectorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_InjectorGenome_Mode, data.mode, defaultObject.mode);
        if (task == SerializationTask::Save) {
            auxiliaries[Id_Constructor_GenomeHeader] = true;
        }
        processLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            auto hasGenomeHeader = auxiliaries.contains(Id_Constructor_GenomeHeader);
            if (hasGenomeHeader) {
                std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
                ar(genomeData);
                if (std::holds_alternative<MakeGenomeCopy>(genomeData)) {
                    data.genome = MakeGenomeCopy();
                } else {
                    data.genome = GenomeDescriptionService::convertDescriptionToBytes(std::get<GenomeDescription>(genomeData));
                }
            } else {
                std::variant<MakeGenomeCopy, std::vector<CellGenomeDescription>> genomeData;
                ar(genomeData);
                if (std::holds_alternative<MakeGenomeCopy>(genomeData)) {
                    data.genome = MakeGenomeCopy();
                } else {
                    GenomeDescription genomeDesc;
                    genomeDesc.cells = std::get<std::vector<CellGenomeDescription>>(genomeData);
                    data.genome = GenomeDescriptionService::convertDescriptionToBytes(genomeDesc);
                }
            }
        } else {
            std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
            if (std::holds_alternative<MakeGenomeCopy>(data.genome)) {
                genomeData = MakeGenomeCopy();
            } else {
                genomeData = GenomeDescriptionService::convertBytesToDescription(std::get<std::vector<uint8_t>>(data.genome));
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
        loadSave<int>(task, auxiliaries, Id_MuscleGenome_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(MuscleGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderGenomeDescription& data)
    {
        DefenderGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_DefenderGenome_Mode, data.mode, defaultObject.mode);
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

        //compatibility with older versions
        //>>>
        if (task == SerializationTask::Load) {
            if (auxiliaries.contains(Id_ReconnectorGenome_Color_Deprecated)) {
                data.restrictToColor = std::get<int>(auxiliaries.at(Id_ReconnectorGenome_Color_Deprecated));
            }
        }
        //<<<
    }
    SPLIT_SERIALIZATION(ReconnectorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorGenomeDescription& data)
    {
        DetonatorGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_DetonatorGenome_Countdown, data.countdown, defaultObject.countdown);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetonatorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellGenomeDescription& data)
    {
        CellGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<float>(task, auxiliaries, Id_CellGenome_ReferenceAngle, data.referenceAngle, defaultObject.referenceAngle);
        loadSave<float>(task, auxiliaries, Id_CellGenome_Energy, data.energy, defaultObject.energy);
        loadSave<int>(task, auxiliaries, Id_CellGenome_Color, data.color, defaultObject.color);
        loadSave<std::optional<int>>(task, auxiliaries, Id_CellGenome_NumRequiredAdditionalConnections, data.numRequiredAdditionalConnections, defaultObject.numRequiredAdditionalConnections);
        loadSave<int>(task, auxiliaries, Id_CellGenome_ExecutionOrderNumber, data.executionOrderNumber, defaultObject.executionOrderNumber);
        loadSave<std::optional<int>>(task, auxiliaries, Id_CellGenome_InputExecutionOrderNumber, data.inputExecutionOrderNumber, defaultObject.inputExecutionOrderNumber);
        loadSave<bool>(task, auxiliaries, Id_CellGenome_OutputBlocked, data.outputBlocked, defaultObject.outputBlocked);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data.cellFunction);
    }
    SPLIT_SERIALIZATION(CellGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GenomeHeaderDescription& data)
    {
        GenomeHeaderDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_GenomeHeader_Shape, data.shape, defaultObject.shape);
        loadSave<int>(task, auxiliaries, Id_GenomeHeader_NumBranches, data.numBranches, defaultObject.numBranches);
        loadSave<bool>(task, auxiliaries, Id_GenomeHeader_SeparateConstruction, data.separateConstruction, defaultObject.separateConstruction);
        loadSave<int>(task, auxiliaries, Id_GenomeHeader_AngleAlignment, data.angleAlignment, defaultObject.angleAlignment);
        loadSave<float>(task, auxiliaries, Id_GenomeHeader_Stiffness, data.stiffness, defaultObject.stiffness);
        loadSave<float>(task, auxiliaries, Id_GenomeHeader_ConnectionDistance, data.connectionDistance, defaultObject.connectionDistance);
        loadSave<int>(task, auxiliaries, Id_GenomeHeader_NumRepetitions, data.numRepetitions, defaultObject.numRepetitions);
        loadSave<float>(task, auxiliaries, Id_GenomeHeader_ConcatenationAngle1, data.concatenationAngle1, defaultObject.concatenationAngle1);
        loadSave<float>(task, auxiliaries, Id_GenomeHeader_ConcatenationAngle2, data.concatenationAngle2, defaultObject.concatenationAngle2);

        //compatibility with older versions
        //>>>
        if (task == SerializationTask::Load) {
            if (auxiliaries.contains(Id_Constructor_SingleConstruction)) {
                data.numBranches = std::get<bool>(auxiliaries.at(Id_Constructor_SingleConstruction)) ? 1 : 2;
            }
        }
        //<<<

        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(GenomeHeaderDescription)

    template <class Archive>
    void serialize(Archive& ar, GenomeDescription& data)
    {
        ar(data.header, data.cells);
    }

    template <class Archive>
    void serialize(Archive& ar, CellMetadataDescription& data)
    {
        ar(data.name, data.description);
    }
    template <class Archive>
    void serialize(Archive& ar, ConnectionDescription& data)
    {
        ar(data.cellId, data.distance, data.angleFromPrevious);
    }
    template <class Archive>
    void serialize(Archive& ar, ActivityDescription& data)
    {
        ar(data.channels);
    }

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuronDescription& data)
    {
        NeuronDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<std::vector<int>>(task, auxiliaries, Id_Neuron_ActivationFunctions, data.activationFunctions, defaultObject.activationFunctions);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data.weights, data.biases);
    }
    SPLIT_SERIALIZATION(NeuronDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TransmitterDescription& data)
    {
        TransmitterDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Transmitter_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(TransmitterDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorDescription& data)
    {
        ConstructorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Constructor_ActivationMode, data.activationMode, defaultObject.activationMode);
        loadSave<int>(task, auxiliaries, Id_Constructor_ConstructionActivationTime, data.constructionActivationTime, defaultObject.constructionActivationTime);
        loadSave<uint64_t>(task, auxiliaries, Id_Constructor_LastConstructedCellId, data.lastConstructedCellId, defaultObject.lastConstructedCellId);
        loadSave<int>(task, auxiliaries, Id_Constructor_GenomeCurrentNodeIndex, data.genomeCurrentNodeIndex, defaultObject.genomeCurrentNodeIndex);
        loadSave<int>(task, auxiliaries, Id_Constructor_GenomeCurrentRepetition, data.genomeCurrentRepetition, defaultObject.genomeCurrentRepetition);
        loadSave<int>(task, auxiliaries, Id_Constructor_CurrentBranch, data.currentBranch, defaultObject.currentBranch);
        loadSave<int>(task, auxiliaries, Id_Constructor_OffspringCreatureId, data.offspringCreatureId, defaultObject.offspringCreatureId);
        loadSave<int>(task, auxiliaries, Id_Constructor_OffspringMutationId, data.offspringMutationId, defaultObject.offspringMutationId);
        loadSave<int>(task, auxiliaries, Id_Constructor_GenomeGeneration, data.genomeGeneration, defaultObject.genomeGeneration);
        loadSave<float>(task, auxiliaries, Id_Constructor_ConstructionAngle1, data.constructionAngle1, defaultObject.constructionAngle1);
        loadSave<float>(task, auxiliaries, Id_Constructor_ConstructionAngle2, data.constructionAngle2, defaultObject.constructionAngle2);
        loadSave<int>(task, auxiliaries, Id_Constructor_NumInheritedGenomeNodes, data.numInheritedGenomeNodes, defaultObject.numInheritedGenomeNodes);
        if (task == SerializationTask::Save) {
            auxiliaries[Id_Constructor_GenomeHeader] = true;
        }
        processLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            auto hasGenomeHeader = auxiliaries.contains(Id_Constructor_GenomeHeader);
            auto useNewGenomeIndex = auxiliaries.contains(Id_Constructor_IsConstructionBuilt) || auxiliaries.contains(Id_Constructor_StateFlags)
                || auxiliaries.contains(Id_Constructor_CurrentBranch);

            if (hasGenomeHeader && useNewGenomeIndex) {
                GenomeDescription genomeDesc;
                ar(genomeDesc);
                data.genome = GenomeDescriptionService::convertDescriptionToBytes(genomeDesc);
            }

            //compatibility with older versions
            //>>>
            if (!hasGenomeHeader) {
                GenomeDescription genomeDesc;
                ar(genomeDesc.cells);
                genomeDesc.header.numBranches =
                    std::get<bool>(auxiliaries.at(Id_Constructor_SingleConstruction)) ? 1 : 2;
                genomeDesc.header.separateConstruction = std::get<bool>(auxiliaries.at(Id_Constructor_SeparateConstruction));
                genomeDesc.header.angleAlignment = std::get<int>(auxiliaries.at(Id_Constructor_AngleAlignment));
                genomeDesc.header.stiffness = std::get<float>(auxiliaries.at(Id_Constructor_Stiffness));
                data.genome = GenomeDescriptionService::convertDescriptionToBytes(genomeDesc);

                data.genomeCurrentNodeIndex = 0;

                if (!genomeDesc.cells.empty()) {
                    data.constructionAngle1 = genomeDesc.cells.front().referenceAngle;
                    data.constructionAngle2 = genomeDesc.cells.back().referenceAngle;
                }
            }

            if (hasGenomeHeader && !useNewGenomeIndex) {
                GenomeDescription genomeDesc;
                ar(genomeDesc);
                data.genome = GenomeDescriptionService::convertDescriptionToBytes(genomeDesc);

                auto oldVersionSpec =
                    GenomeEncodingSpecification().numRepetitions(false).concatenationAngle1(false).concatenationAngle2(false);
                auto oldGenome = GenomeDescriptionService::convertDescriptionToBytes(genomeDesc, oldVersionSpec);
                data.genomeCurrentNodeIndex = GenomeDescriptionService::convertNodeAddressToNodeIndex(oldGenome, data.genomeCurrentNodeIndex, oldVersionSpec);
                if (data.genomeCurrentNodeIndex >= toInt(genomeDesc.cells.size())) {
                    data.genomeCurrentNodeIndex = 0;
                }
            }
            //<<<

        } else {
            GenomeDescription genomeDesc = GenomeDescriptionService::convertBytesToDescription(data.genome);
            ar(genomeDesc);
        }
    }
    SPLIT_SERIALIZATION(ConstructorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SensorDescription& data)
    {
        SensorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<std::optional<float>>(task, auxiliaries, Id_Sensor_FixedAngle, data.fixedAngle, defaultObject.fixedAngle);
        loadSave<float>(task, auxiliaries, Id_Sensor_MinDensity, data.minDensity, defaultObject.minDensity);
        loadSave<std::optional<int>>(task, auxiliaries, Id_Sensor_RestrictToColor, data.restrictToColor, defaultObject.restrictToColor);
        loadSave(task, auxiliaries, Id_Sensor_RestrictToMutants, data.restrictToMutants, defaultObject.restrictToMutants);
        loadSave<float>(task, auxiliaries, Id_Sensor_MemoryChannel1, data.memoryChannel1, defaultObject.memoryChannel1);
        loadSave<float>(task, auxiliaries, Id_Sensor_MemoryChannel2, data.memoryChannel2, defaultObject.memoryChannel2);
        loadSave<float>(task, auxiliaries, Id_Sensor_MemoryChannel3, data.memoryChannel3, defaultObject.memoryChannel3);
        processLoadSaveMap(task, ar, auxiliaries);

        //compatibility with older versions
        //>>>
        if (task == SerializationTask::Load) {
            if (auxiliaries.contains(Id_Sensor_Color_Deprecated)) {
                data.restrictToColor = std::get<int>(auxiliaries.at(Id_Sensor_Color_Deprecated));
            }
            if (auxiliaries.contains(Id_Sensor_RestrictToOtherMutants_Deprecated)) {
                auto restrictToOtherMutants = std::get<bool>(auxiliaries.at(Id_Sensor_RestrictToOtherMutants_Deprecated));
                data.restrictToMutants = restrictToOtherMutants ? SensorRestrictToMutants_RestrictToNutrient : SensorRestrictToMutants_NoRestriction;
            }
        }
        //<<<
    }
    SPLIT_SERIALIZATION(SensorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NerveDescription& data)
    {
        NerveDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Nerve_PulseMode, data.pulseMode, defaultObject.pulseMode);
        loadSave<int>(task, auxiliaries, Id_Nerve_AlternationMode, data.alternationMode, defaultObject.alternationMode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(NerveDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerDescription& data)
    {
        AttackerDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Attacker_Mode, data.mode, defaultObject.mode);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(AttackerDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorDescription& data)
    {
        InjectorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Injector_Mode, data.mode, defaultObject.mode);
        loadSave<int>(task, auxiliaries, Id_Injector_Counter, data.counter, defaultObject.counter);
        if (task == SerializationTask::Save) {
            auxiliaries[Id_Injector_GenomeHeader] = true;
        }
        processLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            auto hasGenomeHeader = auxiliaries.contains(Id_Injector_GenomeHeader);
            if (hasGenomeHeader) {
                GenomeDescription genomeDesc;
                ar(genomeDesc);
                data.genome = GenomeDescriptionService::convertDescriptionToBytes(genomeDesc);
            } else {
                GenomeDescription genomeDesc;
                ar(genomeDesc.cells);
                data.genome = GenomeDescriptionService::convertDescriptionToBytes(genomeDesc);
            }
        } else {
            GenomeDescription genomeDesc = GenomeDescriptionService::convertBytesToDescription(data.genome);
            ar(genomeDesc);
        }
    }
    SPLIT_SERIALIZATION(InjectorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleDescription& data)
    {
        MuscleDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Muscle_Mode, data.mode, defaultObject.mode);
        loadSave<int>(task, auxiliaries, Id_Muscle_LastBendingDirection, data.lastBendingDirection, defaultObject.lastBendingDirection);
        loadSave<int>(task, auxiliaries, Id_Muscle_LastBendingSourceIndex, data.lastBendingSourceIndex, defaultObject.lastBendingSourceIndex);
        loadSave<float>(task, auxiliaries, Id_Muscle_ConsecutiveBendingAngle, data.consecutiveBendingAngle, defaultObject.consecutiveBendingAngle);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(MuscleDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderDescription& data)
    {
        DefenderDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Defender_Mode, data.mode, defaultObject.mode);
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

        //compatibility with older versions
        //>>>
        if (task == SerializationTask::Load) {
            if (auxiliaries.contains(Id_Reconnector_Color_Deprecated)) {
                data.restrictToColor = std::get<int>(auxiliaries.at(Id_Reconnector_Color_Deprecated));
            }
        }
        //<<<
    }
    SPLIT_SERIALIZATION(ReconnectorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorDescription& data)
    {
        DetonatorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Detonator_State, data.state, defaultObject.state);
        loadSave<int>(task, auxiliaries, Id_Detonator_Countdown, data.countdown, defaultObject.countdown);
        processLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DetonatorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CellDescription& data)
    {
        CellDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<float>(task, auxiliaries, Id_Cell_Stiffness, data.stiffness, defaultObject.stiffness);
        loadSave<int>(task, auxiliaries, Id_Cell_Color, data.color, defaultObject.color);
        loadSave<int>(task, auxiliaries, Id_Cell_ExecutionOrderNumber, data.executionOrderNumber, defaultObject.executionOrderNumber);
        loadSave<bool>(task, auxiliaries, Id_Cell_Barrier, data.barrier, defaultObject.barrier);
        loadSave<int>(task, auxiliaries, Id_Cell_Age, data.age, defaultObject.age);
        loadSave<int>(task, auxiliaries, Id_Cell_LivingState, data.livingState, defaultObject.livingState);
        loadSave<int>(task, auxiliaries, Id_Cell_CreatureId, data.creatureId, defaultObject.creatureId);
        loadSave<int>(task, auxiliaries, Id_Cell_MutationId, data.mutationId, defaultObject.mutationId);
        loadSave<uint8_t>(task, auxiliaries, Id_Cell_AncestorMutationId, data.ancestorMutationId, defaultObject.ancestorMutationId);
        loadSave<std::optional<int>>(
            task, auxiliaries, Id_Cell_InputExecutionOrderNumber, data.inputExecutionOrderNumber, defaultObject.inputExecutionOrderNumber);
        loadSave<bool>(task, auxiliaries, Id_Cell_OutputBlocked, data.outputBlocked, defaultObject.outputBlocked);
        loadSave<int>(task, auxiliaries, Id_Cell_ActivationTime, data.activationTime, defaultObject.activationTime);
        loadSave<float>(task, auxiliaries, Id_Cell_GenomeComplexity, data.genomeComplexity, defaultObject.genomeComplexity);
        loadSave<uint8_t>(task, auxiliaries, Id_Cell_DetectedByCreatureId, data.detectedByCreatureId, defaultObject.detectedByCreatureId);
        loadSave<uint8_t>(task, auxiliaries, Id_Cell_CellFunctionUsed, data.cellFunctionUsed, defaultObject.cellFunctionUsed);
        processLoadSaveMap(task, ar, auxiliaries);

        ar(data.id, data.connections, data.pos, data.vel, data.energy, data.maxConnections, data.cellFunction, data.activity, data.metadata);

        //compatibility with older versions
        //>>>
        if (task == SerializationTask::Load) {
            if (auxiliaries.contains(Id_Cell_GenomeComplexity_Deprecated)) {
                data.genomeComplexity = toFloat(std::get<int>(auxiliaries.at(Id_Cell_GenomeComplexity_Deprecated)));
            }
        }
        //<<<
    }
    SPLIT_SERIALIZATION(CellDescription)

    template <class Archive>
    void serialize(Archive& ar, ClusterDescription& data)
    {
        uint64_t id = 0;
        ar(id); //legacy: not used anymore
        ar(data.cells);
    }

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ParticleDescription& data)
    {
        ParticleDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Particle_Color, data.color, defaultObject.color);
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

bool SerializerService::serializeSimulationToFiles(std::string const& filename, DeserializedSimulation const& data)
{
    try {
        log(Priority::Important, "save simulation to " + filename);
        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));
        std::filesystem::path statisticsFilename(filename);
        statisticsFilename.replace_extension(std::filesystem::path(".statistics.csv"));

        {
            zstr::ofstream stream(filename, std::ios::binary);
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

bool SerializerService::deserializeSimulationFromFiles(DeserializedSimulation& data, std::string const& filename)
{
    try {
        log(Priority::Important, "load simulation from " + filename);
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

bool SerializerService::serializeGenomeToFile(std::string const& filename, std::vector<uint8_t> const& genome)
{
    try {
        log(Priority::Important, "save genome to " + filename);
        //wrap constructor cell around genome
        ClusteredDataDescription data;
        if (!wrapGenome(data, genome)) {
            return false;
        }

        zstr::ofstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        serializeDataDescription(data, stream);

        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeGenomeFromFile(std::vector<uint8_t>& genome, std::string const& filename)
{
    try {
        log(Priority::Important, "load genome from " + filename);
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

bool SerializerService::serializeSimulationParametersToFile(std::string const& filename, SimulationParameters const& parameters)
{
    try {
        log(Priority::Important, "save simulation parameters to " + filename);
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

bool SerializerService::deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::string const& filename)
{
    try {
        log(Priority::Important, "load simulation parameters from " + filename);
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

bool SerializerService::serializeStatisticsToFile(std::string const& filename, StatisticsHistoryData const& statistics)
{
    try {
        log(Priority::Important, "save statistics history to " + filename);
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

bool SerializerService::serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content)
{
    try {
        zstr::ofstream fileStream(filename, std::ios::binary);
        if (!fileStream) {
            return false;
        }
        serializeDataDescription(content, fileStream);

        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeContentFromFile(ClusteredDataDescription& content, std::string const& filename)
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

bool SerializerService::deserializeDataDescription(ClusteredDataDescription& data, std::string const& filename)
{
    zstr::ifstream stream(filename, std::ios::binary);
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

    if (!VersionChecker::isVersionValid(version)) {
        throw std::runtime_error("No version detected.");
    }
    if (VersionChecker::isVersionOutdated(version)) {
        throw std::runtime_error("Version not supported.");
    }
    archive(data);
}

void SerializerService::serializeAuxiliaryData(AuxiliaryData const& auxiliaryData, std::ostream& stream)
{
    boost::property_tree::json_parser::write_json(stream, AuxiliaryDataParserService::encodeAuxiliaryData(auxiliaryData));
}

void SerializerService::deserializeAuxiliaryData(AuxiliaryData& auxiliaryData, std::istream& stream)
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    auxiliaryData = AuxiliaryDataParserService::decodeAuxiliaryData(tree);
}

void SerializerService::serializeSimulationParameters(SimulationParameters const& parameters, std::ostream& stream)
{
    boost::property_tree::json_parser::write_json(stream, AuxiliaryDataParserService::encodeSimulationParameters(parameters));
}

void SerializerService::deserializeSimulationParameters(SimulationParameters& parameters, std::istream& stream)
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    parameters = AuxiliaryDataParserService::decodeSimulationParameters(tree);
}

namespace
{
    std::string toString(double value)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(9) << value;
        return ss.str();        
    }

    void loadSave(SerializationTask task, std::vector<std::string>& serializedData, int startIndex, double& value)
    {
        if (task == SerializationTask::Load) {
            if (startIndex < serializedData.size()) {
                value = std::stod(serializedData.at(startIndex));
            }
        } else {
            serializedData.emplace_back(toString(value));
        }
    }

    void loadSave(SerializationTask task, std::vector<std::string>& serializedData, int startIndex, DataPoint& dataPoint)
    {
        if (task == SerializationTask::Load) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                auto index = startIndex + i;
                if (index < serializedData.size()) {
                    dataPoint.values[i] = std::stod(serializedData.at(index));
                }
            }
            if (startIndex + 7 < serializedData.size()) {
                dataPoint.summedValues = std::stod(serializedData.at(startIndex + 7));
            }
        } else {
            for (int i = 0; i < MAX_COLORS; ++i) {
                serializedData.emplace_back(toString(dataPoint.values[i]));
            }
            serializedData.emplace_back(toString(dataPoint.summedValues));
        }
    }

    void loadSave(SerializationTask task, std::vector<std::string>& serializedData, DataPointCollection& dataPoints)
    {
        loadSave(task, serializedData, 0, dataPoints.time);
        loadSave(task, serializedData, 1 + 0 * 8, dataPoints.numCells);
        loadSave(task, serializedData, 1 + 1 * 8, dataPoints.numSelfReplicators);
        loadSave(task, serializedData, 1 + 2 * 8, dataPoints.numViruses);
        loadSave(task, serializedData, 1 + 3 * 8, dataPoints.numConnections);
        loadSave(task, serializedData, 1 + 4 * 8, dataPoints.numParticles);
        loadSave(task, serializedData, 1 + 5 * 8, dataPoints.averageGenomeCells);
        loadSave(task, serializedData, 1 + 6 * 8, dataPoints.totalEnergy);
        loadSave(task, serializedData, 1 + 7 * 8, dataPoints.numCreatedCells);
        loadSave(task, serializedData, 1 + 8 * 8, dataPoints.numAttacks);
        loadSave(task, serializedData, 1 + 9 * 8, dataPoints.numMuscleActivities);
        loadSave(task, serializedData, 1 + 10 * 8, dataPoints.numDefenderActivities);
        loadSave(task, serializedData, 1 + 11 * 8, dataPoints.numTransmitterActivities);
        loadSave(task, serializedData, 1 + 12 * 8, dataPoints.numInjectionActivities);
        loadSave(task, serializedData, 1 + 13 * 8, dataPoints.numCompletedInjections);
        loadSave(task, serializedData, 1 + 14 * 8, dataPoints.numNervePulses);
        loadSave(task, serializedData, 1 + 15 * 8, dataPoints.numNeuronActivities);
        loadSave(task, serializedData, 1 + 16 * 8, dataPoints.numSensorActivities);
        loadSave(task, serializedData, 1 + 17 * 8, dataPoints.numSensorMatches);
        loadSave(task, serializedData, 1 + 18 * 8, dataPoints.numReconnectorCreated);
        loadSave(task, serializedData, 1 + 19 * 8, dataPoints.numReconnectorRemoved);
        loadSave(task, serializedData, 1 + 20 * 8, dataPoints.numDetonations);
        loadSave(task, serializedData, 1 + 21 * 8, dataPoints.numColonies);
    }
}

void SerializerService::serializeStatistics(StatisticsHistoryData const& statistics, std::ostream& stream)
{
    //header row
    stream << "Time step";
    auto writeLabelAllColors = [&stream](auto const& name) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            stream << ", " << name << " (color " << i << ")";
        }
        stream << ", " << name << " (accumulated)";
    };
    writeLabelAllColors("Cells");
    writeLabelAllColors("Self-replicators");
    writeLabelAllColors("Viruses");
    writeLabelAllColors("Cell connections");
    writeLabelAllColors("Energy particles");
    writeLabelAllColors("Average genome cells");
    writeLabelAllColors("Total energy");
    writeLabelAllColors("Created cells");
    writeLabelAllColors("Attacks");
    writeLabelAllColors("Muscle activities");
    writeLabelAllColors("Transmitter activities");
    writeLabelAllColors("Defender activities");
    writeLabelAllColors("Injection activities");
    writeLabelAllColors("Completed injections");
    writeLabelAllColors("Nerve pulses");
    writeLabelAllColors("Neuron activities");
    writeLabelAllColors("Sensor activities");
    writeLabelAllColors("Sensor matches");
    writeLabelAllColors("Reconnector creations");
    writeLabelAllColors("Reconnector deletions");
    writeLabelAllColors("Detonations");
    writeLabelAllColors("Colonies");
    stream << std::endl;

    //content
    for (auto dataPoints : statistics) {
        std::vector<std::string> entries;
        loadSave(SerializationTask::Save, entries, dataPoints);
        stream << boost::join(entries, ",") << std::endl;
    }
}

void SerializerService::deserializeStatistics(StatisticsHistoryData& statistics, std::istream& stream)
{
    statistics.clear();

    std::vector<std::vector<std::string>> data;
    std::string line;
    std::getline(stream, line); //skip header line
    while (std::getline(stream, line)) {
        std::vector<std::string> entries;
        boost::split(entries, line, boost::is_any_of(","));

        DataPointCollection dataPoints;
        loadSave(SerializationTask::Load, entries, dataPoints);

        statistics.emplace_back(dataPoints);
    }
}

bool SerializerService::wrapGenome(ClusteredDataDescription& output, std::vector<uint8_t> const& input)
{
    output.clear();
    output.addCluster(ClusterDescription().addCell(CellDescription().setCellFunction(ConstructorDescription().setGenome(input))));
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
    if (cell.getCellFunctionType() != CellFunction_Constructor) {
        return false;
    }
    output = std::get<ConstructorDescription>(*cell.cellFunction).genome;
    return true;
}
