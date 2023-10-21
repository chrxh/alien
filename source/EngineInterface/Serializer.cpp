#include "Serializer.h"

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

#include "Base/Resources.h"
#include "Base/VersionChecker.h"

#include "Descriptions.h"
#include "SimulationParameters.h"
#include "AuxiliaryDataParser.h"
#include "GenomeConstants.h"
#include "GenomeDescriptions.h"
#include "GenomeDescriptionConverter.h"

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
    auto constexpr Id_Particle_Color = 0;

    auto constexpr Id_Cell_Stiffness = 0;
    auto constexpr Id_Cell_Color = 1;
    auto constexpr Id_Cell_ExecutionOrderNumber = 2;
    auto constexpr Id_Cell_Barrier = 3;
    auto constexpr Id_Cell_Age = 4;
    auto constexpr Id_Cell_LivingState = 5;
    auto constexpr Id_Cell_CreatureId = 11;
    auto constexpr Id_Cell_MutationId = 12;
    auto constexpr Id_Cell_InputExecutionOrderNumber = 9;
    auto constexpr Id_Cell_OutputBlocked = 7;
    auto constexpr Id_Cell_ActivationTime = 8;
    auto constexpr Id_Cell_GenomeNumNodes = 13;
    
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
    auto constexpr Id_Constructor_IsConstructionBuilt = 15;
    auto constexpr Id_Constructor_GenomeCurrentCopy = 16;
    auto constexpr Id_Constructor_LastConstructedCellId = 17;

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
    auto constexpr Id_Sensor_Color = 2;
    auto constexpr Id_Sensor_TargetedCreatureId = 3;
    auto constexpr Id_Sensor_MemoryChannel1 = 4;
    auto constexpr Id_Sensor_MemoryChannel2 = 5;
    auto constexpr Id_Sensor_MemoryChannel3 = 6;

    auto constexpr Id_Transmitter_Mode = 0;

    auto constexpr Id_GenomeHeader_Shape = 0;
    auto constexpr Id_GenomeHeader_SingleConstruction = 1;
    auto constexpr Id_GenomeHeader_SeparateConstruction = 2;
    auto constexpr Id_GenomeHeader_AngleAlignment = 3;
    auto constexpr Id_GenomeHeader_Stiffness = 4;
    auto constexpr Id_GenomeHeader_ConnectionDistance = 5;
    auto constexpr Id_GenomeHeader_NumRepetitions = 6;
    auto constexpr Id_GenomeHeader_ConcatenationAngle1 = 7;
    auto constexpr Id_GenomeHeader_ConcatenationAngle2 = 8;

    auto constexpr Id_CellGenome_ReferenceAngle = 1;
    auto constexpr Id_CellGenome_Energy = 7;
    auto constexpr Id_CellGenome_Color = 2;
    auto constexpr Id_CellGenome_NumRequiredAdditionalConnections = 9;
    auto constexpr Id_CellGenome_ExecutionOrderNumber = 4;
    auto constexpr Id_CellGenome_InputExecutionOrderNumber = 8;
    auto constexpr Id_CellGenome_OutputBlocked = 6;

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
    auto constexpr Id_SensorGenome_Color = 2;

    auto constexpr Id_Reconnector_Color = 0;

}

namespace cereal
{
    enum class SerializationTask
    {
        Load,
        Save
    };
    using VariantData = std::variant<int, float, uint64_t, bool, std::optional<float>, std::optional<int>>;

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
    void setLoadSaveMap(SerializationTask task, Archive& ar, std::unordered_map<int, VariantData>& loadSaveMap)
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
        setLoadSaveMap(task, ar, auxiliaries);

        ar(data.weights, data.biases);
    }
    SPLIT_SERIALIZATION(NeuronGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TransmitterGenomeDescription& data)
    {
        TransmitterGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_TransmitterGenome_Mode, data.mode, defaultObject.mode);
        setLoadSaveMap(task, ar, auxiliaries);
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
        setLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            auto hasGenomeHeader = auxiliaries.contains(Id_ConstructorGenome_GenomeHeader);
            if (hasGenomeHeader) {
                std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
                ar(genomeData);
                if (std::holds_alternative<MakeGenomeCopy>(genomeData)) {
                    data.genome = MakeGenomeCopy();
                } else {
                    data.genome = GenomeDescriptionConverter::convertDescriptionToBytes(std::get<GenomeDescription>(genomeData));
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
                    genomeDesc.header.singleConstruction = std::get<bool>(auxiliaries.at(Id_ConstructorGenome_SingleConstruction));
                    genomeDesc.header.separateConstruction = std::get<bool>(auxiliaries.at(Id_ConstructorGenome_SeparateConstruction));
                    genomeDesc.header.angleAlignment = std::get<int>(auxiliaries.at(Id_ConstructorGenome_AngleAlignment));
                    genomeDesc.header.stiffness = std::get<float>(auxiliaries.at(Id_ConstructorGenome_Stiffness));
                    data.genome = GenomeDescriptionConverter::convertDescriptionToBytes(genomeDesc);
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
                genomeData = GenomeDescriptionConverter::convertBytesToDescription(std::get<std::vector<uint8_t>>(data.genome));
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
        loadSave<int>(task, auxiliaries, Id_SensorGenome_Color, data.color, defaultObject.color);
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SensorGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NerveGenomeDescription& data)
    {
        NerveGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_NerveGenome_PulseMode, data.pulseMode, defaultObject.pulseMode);
        loadSave<int>(task, auxiliaries, Id_NerveGenome_AlternationMode, data.alternationMode, defaultObject.alternationMode);
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(NerveGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerGenomeDescription& data)
    {
        AttackerGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_AttackerGenome_Mode, data.mode, defaultObject.mode);
        setLoadSaveMap(task, ar, auxiliaries);
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
        setLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            auto hasGenomeHeader = auxiliaries.contains(Id_Constructor_GenomeHeader);
            if (hasGenomeHeader) {
                std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
                ar(genomeData);
                if (std::holds_alternative<MakeGenomeCopy>(genomeData)) {
                    data.genome = MakeGenomeCopy();
                } else {
                    data.genome = GenomeDescriptionConverter::convertDescriptionToBytes(std::get<GenomeDescription>(genomeData));
                }
            } else {
                std::variant<MakeGenomeCopy, std::vector<CellGenomeDescription>> genomeData;
                ar(genomeData);
                if (std::holds_alternative<MakeGenomeCopy>(genomeData)) {
                    data.genome = MakeGenomeCopy();
                } else {
                    GenomeDescription genomeDesc;
                    genomeDesc.cells = std::get<std::vector<CellGenomeDescription>>(genomeData);
                    data.genome = GenomeDescriptionConverter::convertDescriptionToBytes(genomeDesc);
                }
            }
        } else {
            std::variant<MakeGenomeCopy, GenomeDescription> genomeData;
            if (std::holds_alternative<MakeGenomeCopy>(data.genome)) {
                genomeData = MakeGenomeCopy();
            } else {
                genomeData = GenomeDescriptionConverter::convertBytesToDescription(std::get<std::vector<uint8_t>>(data.genome));
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
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(MuscleGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderGenomeDescription& data)
    {
        DefenderGenomeDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_DefenderGenome_Mode, data.mode, defaultObject.mode);
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DefenderGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorGenomeDescription& data)
    {
        auto auxiliaries = getLoadSaveMap(task, ar);
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectorGenomeDescription)

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
        setLoadSaveMap(task, ar, auxiliaries);

        ar(data.cellFunction);
    }
    SPLIT_SERIALIZATION(CellGenomeDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, GenomeHeaderDescription& data)
    {
        GenomeHeaderDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_GenomeHeader_Shape, data.shape, defaultObject.shape);
        loadSave<bool>(task, auxiliaries, Id_GenomeHeader_SingleConstruction, data.singleConstruction, defaultObject.singleConstruction);
        loadSave<bool>(task, auxiliaries, Id_GenomeHeader_SeparateConstruction, data.separateConstruction, defaultObject.separateConstruction);
        loadSave<int>(task, auxiliaries, Id_GenomeHeader_AngleAlignment, data.angleAlignment, defaultObject.angleAlignment);
        loadSave<float>(task, auxiliaries, Id_GenomeHeader_Stiffness, data.stiffness, defaultObject.stiffness);
        loadSave<float>(task, auxiliaries, Id_GenomeHeader_ConnectionDistance, data.connectionDistance, defaultObject.connectionDistance);
        loadSave<int>(task, auxiliaries, Id_GenomeHeader_NumRepetitions, data.numRepetitions, defaultObject.numRepetitions);
        loadSave<float>(task, auxiliaries, Id_GenomeHeader_ConcatenationAngle1, data.concatenationAngle1, defaultObject.concatenationAngle1);
        loadSave<float>(task, auxiliaries, Id_GenomeHeader_ConcatenationAngle2, data.concatenationAngle2, defaultObject.concatenationAngle2);
        setLoadSaveMap(task, ar, auxiliaries);
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
        setLoadSaveMap(task, ar, auxiliaries);

        ar(data.weights, data.biases);
    }
    SPLIT_SERIALIZATION(NeuronDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, TransmitterDescription& data)
    {
        TransmitterDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Transmitter_Mode, data.mode, defaultObject.mode);
        setLoadSaveMap(task, ar, auxiliaries);
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
        loadSave<int>(task, auxiliaries, Id_Constructor_GenomeCurrentCopy, data.genomeCurrentRepetition, defaultObject.genomeCurrentRepetition);
        loadSave<bool>(task, auxiliaries, Id_Constructor_IsConstructionBuilt, data.isConstructionBuilt, defaultObject.isConstructionBuilt);
        loadSave<int>(task, auxiliaries, Id_Constructor_OffspringCreatureId, data.offspringCreatureId, defaultObject.offspringCreatureId);
        loadSave<int>(task, auxiliaries, Id_Constructor_OffspringMutationId, data.offspringMutationId, defaultObject.offspringMutationId);
        loadSave<int>(task, auxiliaries, Id_Constructor_GenomeGeneration, data.genomeGeneration, defaultObject.genomeGeneration);
        loadSave<float>(task, auxiliaries, Id_Constructor_ConstructionAngle1, data.constructionAngle1, defaultObject.constructionAngle1);
        loadSave<float>(task, auxiliaries, Id_Constructor_ConstructionAngle2, data.constructionAngle2, defaultObject.constructionAngle2);
        if (task == SerializationTask::Save) {
            auxiliaries[Id_Constructor_GenomeHeader] = true;
        }
        setLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            auto hasGenomeHeader = auxiliaries.contains(Id_Constructor_GenomeHeader);
            auto useNewGenomeIndex = auxiliaries.contains(Id_Constructor_IsConstructionBuilt);

            if (hasGenomeHeader && useNewGenomeIndex) {
                GenomeDescription genomeDesc;
                ar(genomeDesc);
                data.genome = GenomeDescriptionConverter::convertDescriptionToBytes(genomeDesc);
            }

            //compatibility with older versions
            //>>>
            if (!hasGenomeHeader) {
                GenomeDescription genomeDesc;
                ar(genomeDesc.cells);
                genomeDesc.header.singleConstruction = std::get<bool>(auxiliaries.at(Id_Constructor_SingleConstruction));
                genomeDesc.header.separateConstruction = std::get<bool>(auxiliaries.at(Id_Constructor_SeparateConstruction));
                genomeDesc.header.angleAlignment = std::get<int>(auxiliaries.at(Id_Constructor_AngleAlignment));
                genomeDesc.header.stiffness = std::get<float>(auxiliaries.at(Id_Constructor_Stiffness));
                data.genome = GenomeDescriptionConverter::convertDescriptionToBytes(genomeDesc);

                data.isConstructionBuilt = toInt(data.genome.size()) <= data.genomeCurrentNodeIndex;  //in old versions genomeCurrentNodeIndex was the byte index
                data.genomeCurrentNodeIndex = 0;

                if (!genomeDesc.cells.empty()) {
                    data.constructionAngle1 = genomeDesc.cells.front().referenceAngle;
                    data.constructionAngle2 = genomeDesc.cells.back().referenceAngle;
                }
            }

            if (hasGenomeHeader && !useNewGenomeIndex) {
                GenomeDescription genomeDesc;
                ar(genomeDesc);
                data.genome = GenomeDescriptionConverter::convertDescriptionToBytes(genomeDesc);

                auto oldVersionSpec =
                    GenomeEncodingSpecification().numRepetitions(false).concatenationAngle1(false).concatenationAngle2(false);
                auto oldGenome = GenomeDescriptionConverter::convertDescriptionToBytes(genomeDesc, oldVersionSpec);
                data.isConstructionBuilt = toInt(oldGenome.size()) <= data.genomeCurrentNodeIndex;  //in old versions genomeCurrentNodeIndex was the byte index
                data.genomeCurrentNodeIndex = GenomeDescriptionConverter::convertNodeAddressToNodeIndex(oldGenome, data.genomeCurrentNodeIndex, oldVersionSpec);
                if (data.genomeCurrentNodeIndex >= toInt(genomeDesc.cells.size())) {
                    data.genomeCurrentNodeIndex = 0;
                }
            }
            //<<<

        } else {
            GenomeDescription genomeDesc = GenomeDescriptionConverter::convertBytesToDescription(data.genome);
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
        loadSave<int>(task, auxiliaries, Id_Sensor_Color, data.color, defaultObject.color);
        loadSave<int>(task, auxiliaries, Id_Sensor_TargetedCreatureId, data.targetedCreatureId, defaultObject.targetedCreatureId);
        loadSave<float>(task, auxiliaries, Id_Sensor_MemoryChannel1, data.memoryChannel1, defaultObject.memoryChannel1);
        loadSave<float>(task, auxiliaries, Id_Sensor_MemoryChannel2, data.memoryChannel2, defaultObject.memoryChannel2);
        loadSave<float>(task, auxiliaries, Id_Sensor_MemoryChannel3, data.memoryChannel3, defaultObject.memoryChannel3);
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(SensorDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NerveDescription& data)
    {
        NerveDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Nerve_PulseMode, data.pulseMode, defaultObject.pulseMode);
        loadSave<int>(task, auxiliaries, Id_Nerve_AlternationMode, data.alternationMode, defaultObject.alternationMode);
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(NerveDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerDescription& data)
    {
        AttackerDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Attacker_Mode, data.mode, defaultObject.mode);
        setLoadSaveMap(task, ar, auxiliaries);
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
        setLoadSaveMap(task, ar, auxiliaries);

        if (task == SerializationTask::Load) {
            auto hasGenomeHeader = auxiliaries.contains(Id_Injector_GenomeHeader);
            if (hasGenomeHeader) {
                GenomeDescription genomeDesc;
                ar(genomeDesc);
                data.genome = GenomeDescriptionConverter::convertDescriptionToBytes(genomeDesc);
            } else {
                GenomeDescription genomeDesc;
                ar(genomeDesc.cells);
                data.genome = GenomeDescriptionConverter::convertDescriptionToBytes(genomeDesc);
            }
        } else {
            GenomeDescription genomeDesc = GenomeDescriptionConverter::convertBytesToDescription(data.genome);
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
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(MuscleDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderDescription& data)
    {
        DefenderDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Defender_Mode, data.mode, defaultObject.mode);
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(DefenderDescription)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorDescription& data)
    {
        ReconnectorDescription defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        loadSave<int>(task, auxiliaries, Id_Reconnector_Color, data.color, defaultObject.color);
        setLoadSaveMap(task, ar, auxiliaries);
    }
    SPLIT_SERIALIZATION(ReconnectorDescription)

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
        loadSave<std::optional<int>>(
            task, auxiliaries, Id_Cell_InputExecutionOrderNumber, data.inputExecutionOrderNumber, defaultObject.inputExecutionOrderNumber);
        loadSave<bool>(task, auxiliaries, Id_Cell_OutputBlocked, data.outputBlocked, defaultObject.outputBlocked);
        loadSave<int>(task, auxiliaries, Id_Cell_ActivationTime, data.activationTime, defaultObject.activationTime);
        loadSave<int>(task, auxiliaries, Id_Cell_GenomeNumNodes, data.genomeNumNodes, defaultObject.genomeNumNodes);
        setLoadSaveMap(task, ar, auxiliaries);

        ar(data.id, data.connections, data.pos, data.vel, data.energy, data.maxConnections, data.cellFunction, data.activity, data.metadata);
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
        setLoadSaveMap(task, ar, auxiliaries);

        ar(data.id, data.pos, data.vel, data.energy);
    }
    SPLIT_SERIALIZATION(ParticleDescription)

    template <class Archive>
    void serialize(Archive& ar, ClusteredDataDescription& data)
    {
        ar(data.clusters, data.particles);
    }
}

bool Serializer::serializeSimulationToFiles(std::string const& filename, DeserializedSimulation const& data)
{
    try {

        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));

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
            stream.close();
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::deserializeSimulationFromFiles(DeserializedSimulation& data, std::string const& filename)
{
    try {
        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));

        if (!deserializeDataDescription(data.mainData, filename)) {
            return false;
        }
        {
            std::ifstream stream(settingsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeAuxiliaryData(data.auxiliaryData, stream);
            stream.close();
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::serializeSimulationToStrings(SerializedSimulation& output, DeserializedSimulation const& input)
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
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::deserializeSimulationFromStrings(DeserializedSimulation& output, SerializedSimulation const& input)
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
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::serializeGenomeToFile(std::string const& filename, std::vector<uint8_t> const& genome)
{
    try {
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

bool Serializer::deserializeGenomeFromFile(std::vector<uint8_t>& genome, std::string const& filename)
{
    try {
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

bool Serializer::serializeGenomeToString(std::string& output, std::vector<uint8_t> const& input)
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

bool Serializer::deserializeGenomeFromString(std::vector<uint8_t>& output, std::string const& input)
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

bool Serializer::serializeSimulationParametersToFile(std::string const& filename, SimulationParameters const& parameters)
{
    try {
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

bool Serializer::deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::string const& filename)
{
    try {
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

bool Serializer::serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content)
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

bool Serializer::deserializeContentFromFile(ClusteredDataDescription& content, std::string const& filename)
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

void Serializer::serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream)
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(Const::ProgramVersion);
    archive(data);
}

bool Serializer::deserializeDataDescription(ClusteredDataDescription& data, std::string const& filename)
{
    zstr::ifstream stream(filename, std::ios::binary);
    if (!stream) {
        return false;
    }
    deserializeDataDescription(data, stream);
    return true;
}

void Serializer::deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream)
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

void Serializer::serializeAuxiliaryData(AuxiliaryData const& auxiliaryData, std::ostream& stream)
{
    boost::property_tree::json_parser::write_json(stream, AuxiliaryDataParser::encodeAuxiliaryData(auxiliaryData));
}

void Serializer::deserializeAuxiliaryData(AuxiliaryData& auxiliaryData, std::istream& stream)
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    auxiliaryData = AuxiliaryDataParser::decodeAuxiliaryData(tree);
}

void Serializer::serializeSimulationParameters(SimulationParameters const& parameters, std::ostream& stream)
{
    boost::property_tree::json_parser::write_json(stream, AuxiliaryDataParser::encodeSimulationParameters(parameters));
}

void Serializer::deserializeSimulationParameters(SimulationParameters& parameters, std::istream& stream)
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    parameters = AuxiliaryDataParser::decodeSimulationParameters(tree);
}

bool Serializer::wrapGenome(ClusteredDataDescription& output, std::vector<uint8_t> const& input)
{
    output.clear();
    output.addCluster(ClusterDescription().addCell(CellDescription().setCellFunction(ConstructorDescription().setGenome(input))));
    return true;
}


bool Serializer::unwrapGenome(std::vector<uint8_t>& output, ClusteredDataDescription const& input)
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
