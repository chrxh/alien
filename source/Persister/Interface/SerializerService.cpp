#include "SerializerService.h"

#include <algorithm>
#include <chrono>
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
#include <cereal/types/vector.hpp>

#include <Base/Interface/LoggingService.h>
#include <Base/Interface/Resources.h>
#include <Base/Interface/VersionParserService.h>

#include <Data/Interface/Descs.h>
#include <Data/Interface/GenomeDesc.h>
#include <Data/Interface/ParametersValidationService.h>
#include <Data/Interface/SimulationParameters.h>

#include "JsonSerializationScope.h"
#include "SerializationScope.h"
#include "SettingsParserService.h"

#include "ZstdStream.h"

/************************************************************************/
/* Genome data                                                          */
/************************************************************************/
namespace
{
    auto constexpr Id_Genome_Id = SerializationKey(0, "id");
    auto constexpr Id_Genome_Name = SerializationKey(1, "name");
    auto constexpr Id_Genome_FrontAngle = SerializationKey(2, "frontAngle");
    auto constexpr Id_Genome_ResistanceToInjection = SerializationKey(6, "resistanceToInjection");
    auto constexpr Id_Genome_ApplyMetaMutations = SerializationKey(7, "applyMetaMutations");

    auto constexpr Id_NeuronMutation_NodeProbability = SerializationKey(0, "nodeProbability");
    auto constexpr Id_NeuronMutation_WeightChangeSigma = SerializationKey(1, "weightChangeSigma");
    auto constexpr Id_NeuronMutation_BiasChangeSigma = SerializationKey(2, "biasChangeSigma");
    auto constexpr Id_NeuronMutation_ActfnChangeProbability = SerializationKey(3, "actfnChangeProbability");

    auto constexpr Id_ConnectionMutation_NodeProbability = SerializationKey(0, "nodeProbability");
    auto constexpr Id_ConnectionMutation_ValueChangeSigma = SerializationKey(1, "valueChangeSigma");

    auto constexpr Id_CellTypePropertiesMutation_NodeProbability = SerializationKey(0, "nodeProbability");
    auto constexpr Id_CellTypePropertiesMutation_ValueChangeSigma = SerializationKey(1, "valueChangeSigma");
    auto constexpr Id_CellTypePropertiesMutation_EnumChangeProbability = SerializationKey(2, "enumChangeProbability");

    auto constexpr Id_GeometryMutation_GeneProbability = SerializationKey(0, "geneProbability");
    auto constexpr Id_GeometryMutation_ValueChangeSigma = SerializationKey(1, "valueChangeSigma");
    auto constexpr Id_GeometryMutation_EnumChangeProbability = SerializationKey(2, "enumChangeProbability");

    auto constexpr Id_CellTypeModeMutation_NodeProbability = SerializationKey(0, "nodeProbability");

    auto constexpr Id_CellTypeMutation_NodeProbability = SerializationKey(0, "nodeProbability");

    auto constexpr Id_CustomizationMutation_GenomeProbability = SerializationKey(0, "genomeProbability");

    auto constexpr Id_VoidMutation_NodeProbability = SerializationKey(0, "nodeProbability");

    auto constexpr Id_ExtendGeneMutation_GeneProbability = SerializationKey(0, "geneProbability");

    auto constexpr Id_AddNodeMutation_NodeProbability = SerializationKey(0, "nodeProbability");

    auto constexpr Id_TrimGeneMutation_GeneProbability = SerializationKey(0, "geneProbability");

    auto constexpr Id_DeleteNodeMutation_NodeProbability = SerializationKey(0, "nodeProbability");

    auto constexpr Id_AddGeneMutation_GeneProbability = SerializationKey(0, "geneProbability");

    auto constexpr Id_DuplicateGeneMutation_GeneProbability = SerializationKey(0, "geneProbability");

    auto constexpr Id_DeleteGeneMutation_GeneProbability = SerializationKey(0, "geneProbability");

    auto constexpr Id_SwapGeneMutation_GeneProbability = SerializationKey(0, "geneProbability");

    auto constexpr Id_CopyNodeSectionMutation_GeneProbability = SerializationKey(0, "geneProbability");

    auto constexpr Id_MoveNodeSectionMutation_GeneProbability = SerializationKey(0, "geneProbability");

    auto constexpr Id_ConstructorMutation_NodeProbability = SerializationKey(0, "nodeProbability");
    auto constexpr Id_ConstructorMutation_ValueChangeSigma = SerializationKey(1, "valueChangeSigma");
    auto constexpr Id_ConstructorMutation_EnumChangeProbability = SerializationKey(2, "enumChangeProbability");
    auto constexpr Id_ConstructorMutation_ConstructorToggleProbability = SerializationKey(3, "constructorToggleProbability");

    auto constexpr Id_Gene_Name = SerializationKey(0, "name");
    auto constexpr Id_Gene_Shape = SerializationKey(1, "shape");
    auto constexpr Id_Gene_Stiffness = SerializationKey(5, "stiffness");
    auto constexpr Id_Gene_ConnectionDistance = SerializationKey(6, "connectionDistance");
    auto constexpr Id_Gene_HomogeneousCellType = SerializationKey(7, "homogeneousCellType");

    auto constexpr Id_Node_ReferenceAngle = SerializationKey(0, "referenceAngle");
    auto constexpr Id_Node_Color = SerializationKey(1, "color");

    auto constexpr Id_NeuralNetGenome_Weights = SerializationKey(0, "weights");
    auto constexpr Id_NeuralNetGenome_Biases = SerializationKey(1, "biases");
    auto constexpr Id_NeuralNetGenome_ActivationFunctions = SerializationKey(2, "activationFunctions");
    auto constexpr Id_NeuralNetGenome_ConnectionWeights = SerializationKey(3, "connectionWeights");

    auto constexpr Id_DepotGenome_storageLimit = SerializationKey(0, "storageLimit");

    auto constexpr Id_DefenderGenome_Mode = SerializationKey(0, "mode");

    auto constexpr Id_ConstructorGenome_AutoTriggerInterval = SerializationKey(0, "autoTriggerInterval");
    auto constexpr Id_ConstructorGenome_GeneIndex = SerializationKey(1, "geneIndex");
    auto constexpr Id_ConstructorGenome_ConstructionAngle = SerializationKey(3, "constructionAngle");
    auto constexpr Id_ConstructorGenome_ProvideEnergy = SerializationKey(4, "provideEnergy");
    auto constexpr Id_ConstructorGenome_Separation = SerializationKey(6, "separation");
    auto constexpr Id_ConstructorGenome_NumBranches = SerializationKey(7, "numBranches");
    auto constexpr Id_ConstructorGenome_NumConcatenations = SerializationKey(8, "numConcatenations");

    auto constexpr Id_SensorGenome_AutoTrigger = SerializationKey(0, "autoTrigger");
    auto constexpr Id_SensorGenome_MinRange = SerializationKey(1, "minRange");
    auto constexpr Id_SensorGenome_MaxRange = SerializationKey(2, "maxRange");
    auto constexpr Id_SensorGenome_TagForAttackers = SerializationKey(3, "tagForAttackers");

    auto constexpr Id_SensorModeGenome_DetectEnergy_MinDensity = SerializationKey(0, "minDensity");

    auto constexpr Id_SensorModeGenome_DetectFreeCell_MinDensity = SerializationKey(0, "minDensity");
    auto constexpr Id_SensorModeGenome_DetectFreeCell_RestrictToColor = SerializationKey(1, "restrictToColors");

    auto constexpr Id_SensorModeGenome_DetectCreature_MinNumCells = SerializationKey(0, "minNumCells");
    auto constexpr Id_SensorModeGenome_DetectCreature_MaxNumCells = SerializationKey(1, "maxNumCells");
    auto constexpr Id_SensorModeGenome_DetectCreature_RestrictToColor = SerializationKey(2, "restrictToColors");
    auto constexpr Id_SensorModeGenome_DetectCreature_RestrictToLineage = SerializationKey(3, "restrictToLineage");

    auto constexpr Id_MuscleModeGenome_AutoBending_MaxAngleDeviation = SerializationKey(0, "maxAngleDeviation");
    auto constexpr Id_MuscleModeGenome_AutoBending_ForwardBackwardRatio = SerializationKey(4, "forwardBackwardRatio");

    auto constexpr Id_MuscleModeGenome_ManualBending_MaxAngleDeviation = SerializationKey(0, "maxAngleDeviation");
    auto constexpr Id_MuscleModeGenome_ManualBending_ForwardBackwardRatio = SerializationKey(1, "forwardBackwardRatio");

    auto constexpr Id_MuscleModeGenome_AngleBending_MaxAngleDeviation = SerializationKey(0, "maxAngleDeviation");
    auto constexpr Id_MuscleModeGenome_AngleBending_AttractionRepulsionRatio = SerializationKey(1, "attractionRepulsionRatio");

    auto constexpr Id_MuscleModeGenome_AutoCrawling_MaxDistanceDeviation = SerializationKey(0, "maxDistanceDeviation");
    auto constexpr Id_MuscleModeGenome_AutoCrawling_ForwardBackwardRatio = SerializationKey(1, "forwardBackwardRatio");

    auto constexpr Id_MuscleModeGenome_ManualCrawling_MaxDistanceDeviation = SerializationKey(0, "maxDistanceDeviation");
    auto constexpr Id_MuscleModeGenome_ManualCrawling_ForwardBackwardRatio = SerializationKey(1, "forwardBackwardRatio");

    auto constexpr Id_GeneratorGenome_Additive = SerializationKey(0, "additive");
    auto constexpr Id_GeneratorGenome_MinValue = SerializationKey(4, "minValue");
    auto constexpr Id_GeneratorGenome_MaxValue = SerializationKey(5, "maxValue");
    auto constexpr Id_GeneratorGenome_TimeOffset = SerializationKey(2, "timeOffset");

    auto constexpr Id_GeneratorModeGenome_SquareSignal_Amplitude = SerializationKey(0, "amplitude");
    auto constexpr Id_GeneratorModeGenome_SquareSignal_Period = SerializationKey(1, "period");

    auto constexpr Id_GeneratorModeGenome_SawtoothSignal_Amplitude = SerializationKey(0, "amplitude");
    auto constexpr Id_GeneratorModeGenome_SawtoothSignal_Period = SerializationKey(1, "period");

    auto constexpr Id_AttackerModeGenome_FreeCell_RestrictToColor = SerializationKey(0, "restrictToColors");

    auto constexpr Id_AttackerModeGenome_Creature_MinNumCells = SerializationKey(0, "minNumCells");
    auto constexpr Id_AttackerModeGenome_Creature_MaxNumCells = SerializationKey(1, "maxNumCells");
    auto constexpr Id_AttackerModeGenome_Creature_RestrictToColor = SerializationKey(2, "restrictToColor");
    auto constexpr Id_AttackerModeGenome_Creature_RestrictToLineage = SerializationKey(3, "restrictToLineage");

    auto constexpr Id_InjectorGenome_GeneIndex = SerializationKey(0, "geneIndex");

    auto constexpr Id_ReconnectorModeGenome_FreeCell_RestrictToColor = SerializationKey(0, "restrictToColors");

    auto constexpr Id_ReconnectorModeGenome_Creature_MinNumCells = SerializationKey(0, "minNumCells");
    auto constexpr Id_ReconnectorModeGenome_Creature_MaxNumCells = SerializationKey(1, "maxNumCells");
    auto constexpr Id_ReconnectorModeGenome_Creature_RestrictToColor = SerializationKey(2, "restrictToColors");
    auto constexpr Id_ReconnectorModeGenome_Creature_RestrictToLineage = SerializationKey(3, "restrictToLineage");

    auto constexpr Id_DetonatorGenome_Countdown = SerializationKey(0, "countdown");

    auto constexpr Id_DigestorGenome_RawEnergyConductivity = SerializationKey(0, "rawEnergyConductivity");

    auto constexpr Id_SignalEntryGenome_Channels = SerializationKey(0, "channels");

    auto constexpr Id_SignalDelayGenome_Delay = SerializationKey(0, "delay");

    auto constexpr Id_SignalRecorderGenome_ReadOnly = SerializationKey(0, "readOnly");
    auto constexpr Id_SignalRecorderGenome_NumSavedSignalEntries = SerializationKey(1, "numWrittenSignalEntries");

    auto constexpr Id_SignalStorageGenome_ReadOnly = SerializationKey(0, "readOnly");

    auto constexpr Id_SignalIntegratorGenome_NewSignalWeight = SerializationKey(0, "newSignalWeight");

    auto constexpr Id_MemoryGenome_ChannelBitMask = SerializationKey(0, "channelBitMask");

    auto constexpr Id_SenderGenome_Range = SerializationKey(0, "range");
    auto constexpr Id_SenderGenome_Oneway = SerializationKey(3, "oneway");

    auto constexpr Id_ReceiverGenome_RestrictToColor = SerializationKey(1, "restrictToColors");
    auto constexpr Id_ReceiverGenome_RestrictToLineage = SerializationKey(2, "restrictToLineage");

    // Description member keys
    auto constexpr Id_Node_NeuralNetwork = SerializationKey(2, "neuralNetwork");
    auto constexpr Id_Node_CellType = SerializationKey(3, "cellType");
    auto constexpr Id_Node_Constructor = SerializationKey(4, "constructor");

    auto constexpr Id_Gene_Nodes = SerializationKey(8, "nodes");

    auto constexpr Id_MutationRates_NeuronMutation1 = SerializationKey(1, "neuronMutation1");
    auto constexpr Id_MutationRates_NeuronMutation2 = SerializationKey(2, "neuronMutation2");
    auto constexpr Id_MutationRates_ConnectionMutation1 = SerializationKey(3, "connectionMutation1");
    auto constexpr Id_MutationRates_ConnectionMutation2 = SerializationKey(4, "connectionMutation2");
    auto constexpr Id_MutationRates_CellTypePropertiesMutation1 = SerializationKey(5, "cellTypePropertiesMutation1");
    auto constexpr Id_MutationRates_CellTypePropertiesMutation2 = SerializationKey(6, "cellTypePropertiesMutation2");
    auto constexpr Id_MutationRates_CellTypeModeMutation = SerializationKey(7, "cellTypeModeMutation");
    auto constexpr Id_MutationRates_CellTypeMutation = SerializationKey(8, "cellTypeMutation");
    auto constexpr Id_MutationRates_VoidMutation = SerializationKey(9, "voidMutation");
    auto constexpr Id_MutationRates_ConstructorMutation1 = SerializationKey(10, "constructorMutation1");
    auto constexpr Id_MutationRates_ConstructorMutation2 = SerializationKey(11, "constructorMutation2");
    auto constexpr Id_MutationRates_ExtendGeneMutation = SerializationKey(12, "extendGeneMutation");
    auto constexpr Id_MutationRates_AddNodeMutation = SerializationKey(13, "addNodeMutation");
    auto constexpr Id_MutationRates_TrimGeneMutation = SerializationKey(14, "trimGeneMutation");
    auto constexpr Id_MutationRates_DeleteNodeMutation = SerializationKey(15, "deleteNodeMutation");
    auto constexpr Id_MutationRates_DuplicateGeneMutation = SerializationKey(16, "duplicateGeneMutation");
    auto constexpr Id_MutationRates_DeleteGeneMutation = SerializationKey(17, "deleteGeneMutation");
    auto constexpr Id_MutationRates_CopyNodeSectionMutation = SerializationKey(18, "copyNodeSectionMutation");
    auto constexpr Id_MutationRates_MoveNodeSectionMutation = SerializationKey(19, "moveNodeSectionMutation");
    auto constexpr Id_MutationRates_GeometryMutation1 = SerializationKey(20, "geometryMutation1");
    auto constexpr Id_MutationRates_GeometryMutation2 = SerializationKey(21, "geometryMutation2");
    auto constexpr Id_MutationRates_CustomizationMutation = SerializationKey(22, "customizationMutation");
    auto constexpr Id_MutationRates_AddGeneMutation = SerializationKey(23, "addGeneMutation");
    auto constexpr Id_MutationRates_SwapGeneMutation = SerializationKey(24, "swapGeneMutation");

    auto constexpr Id_Genome_Genes = SerializationKey(6, "genes");
    auto constexpr Id_Genome_MutationRates = SerializationKey(7, "mutationRates");

    auto constexpr Id_SensorGenome_Mode = SerializationKey(4, "mode");
    auto constexpr Id_GeneratorGenome_Mode = SerializationKey(3, "mode");
    auto constexpr Id_AttackerGenome_Mode = SerializationKey(0, "mode");
    auto constexpr Id_MuscleGenome_Mode = SerializationKey(0, "mode");
    auto constexpr Id_ReconnectorGenome_Mode = SerializationKey(0, "mode");
    auto constexpr Id_MemoryGenome_Mode = SerializationKey(1, "mode");
    auto constexpr Id_MemoryGenome_SignalEntries = SerializationKey(2, "signalEntries");
    auto constexpr Id_CommunicatorGenome_Mode = SerializationKey(0, "mode");

    // Serialized type ids
    auto constexpr Id_CellTypeGenome_Base = SerializationKey(0, "base");
    auto constexpr Id_CellTypeGenome_Depot = SerializationKey(1, "depot");
    auto constexpr Id_CellTypeGenome_Sensor = SerializationKey(2, "sensor");
    auto constexpr Id_CellTypeGenome_Generator = SerializationKey(3, "generator");
    auto constexpr Id_CellTypeGenome_Attacker = SerializationKey(4, "attacker");
    auto constexpr Id_CellTypeGenome_Injector = SerializationKey(5, "injector");
    auto constexpr Id_CellTypeGenome_Muscle = SerializationKey(6, "muscle");
    auto constexpr Id_CellTypeGenome_Defender = SerializationKey(7, "defender");
    auto constexpr Id_CellTypeGenome_Reconnector = SerializationKey(8, "reconnector");
    auto constexpr Id_CellTypeGenome_Detonator = SerializationKey(9, "detonator");
    auto constexpr Id_CellTypeGenome_Digestor = SerializationKey(10, "digestor");
    auto constexpr Id_CellTypeGenome_Memory = SerializationKey(11, "memory");
    auto constexpr Id_CellTypeGenome_Communicator = SerializationKey(12, "communicator");
    auto constexpr Id_CellTypeGenome_Void = SerializationKey(13, "void");

    auto constexpr Id_SensorModeGenome_DetectEnergy = SerializationKey(0, "detectEnergy");
    auto constexpr Id_SensorModeGenome_DetectSolid = SerializationKey(1, "detectSolid");
    auto constexpr Id_SensorModeGenome_DetectFreeCell = SerializationKey(2, "detectFreeCell");
    auto constexpr Id_SensorModeGenome_DetectCreature = SerializationKey(3, "detectCreature");

    auto constexpr Id_GeneratorModeGenome_SquareSignal = SerializationKey(0, "squareSignal");
    auto constexpr Id_GeneratorModeGenome_SawtoothSignal = SerializationKey(1, "sawtoothSignal");

    auto constexpr Id_AttackerModeGenome_AttackFreeCell = SerializationKey(0, "attackFreeCell");
    auto constexpr Id_AttackerModeGenome_AttackCreature = SerializationKey(1, "attackCreature");

    auto constexpr Id_MuscleModeGenome_AutoBending = SerializationKey(0, "autoBending");
    auto constexpr Id_MuscleModeGenome_ManualBending = SerializationKey(1, "manualBending");
    auto constexpr Id_MuscleModeGenome_AngleBending = SerializationKey(2, "angleBending");
    auto constexpr Id_MuscleModeGenome_AutoCrawling = SerializationKey(3, "autoCrawling");
    auto constexpr Id_MuscleModeGenome_ManualCrawling = SerializationKey(4, "manualCrawling");
    auto constexpr Id_MuscleModeGenome_DirectMovement = SerializationKey(5, "directMovement");

    auto constexpr Id_ReconnectorModeGenome_ReconnectSolid = SerializationKey(0, "reconnectSolid");
    auto constexpr Id_ReconnectorModeGenome_ReconnectFreeCell = SerializationKey(1, "reconnectFreeCell");
    auto constexpr Id_ReconnectorModeGenome_ReconnectCreature = SerializationKey(2, "reconnectCreature");

    auto constexpr Id_MemoryModeGenome_SignalDelay = SerializationKey(0, "signalDelay");
    auto constexpr Id_MemoryModeGenome_SignalRecorder = SerializationKey(1, "signalRecorder");
    auto constexpr Id_MemoryModeGenome_SignalStorage = SerializationKey(2, "signalStorage");
    auto constexpr Id_MemoryModeGenome_SignalIntegrator = SerializationKey(3, "signalIntegrator");

    auto constexpr Id_CommunicatorModeGenome_Sender = SerializationKey(0, "sender");
    auto constexpr Id_CommunicatorModeGenome_Receiver = SerializationKey(1, "receiver");
}

namespace cereal
{
    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetGenomeDesc& data)
    {
        NeuralNetGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addFixedSizeMember(Id_NeuralNetGenome_Weights, data._weights, defaultObject._weights);
        scope.addFixedSizeMember(Id_NeuralNetGenome_Biases, data._biases, defaultObject._biases);
        scope.addFixedSizeMember(Id_NeuralNetGenome_ActivationFunctions, data._activationFunctions, defaultObject._activationFunctions);
        scope.addFixedSizeMember(Id_NeuralNetGenome_ConnectionWeights, data._connectionWeights, defaultObject._connectionWeights);
    }
    SPLIT_SERIALIZATION(NeuralNetGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, BaseGenomeDesc& data)
    {
        BaseGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(BaseGenomeDesc, Id_CellTypeGenome_Base)
    SPLIT_SERIALIZATION(BaseGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DepotGenomeDesc& data)
    {
        DepotGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DepotGenome_storageLimit, data._storageLimit, defaultObject._storageLimit);
    }
    REGISTER_SERIALIZED_TYPE(DepotGenomeDesc, Id_CellTypeGenome_Depot)
    SPLIT_SERIALIZATION(DepotGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorGenomeDesc& data)
    {
        ConstructorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ConstructorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
        scope.addMember(Id_ConstructorGenome_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        scope.addMember(Id_ConstructorGenome_ConstructionAngle, data._constructionAngle, defaultObject._constructionAngle);
        scope.addMember(Id_ConstructorGenome_ProvideEnergy, data._provideEnergy, defaultObject._provideEnergy);
        scope.addMember(Id_ConstructorGenome_Separation, data._separation, defaultObject._separation);
        scope.addMember(Id_ConstructorGenome_NumBranches, data._numBranches, defaultObject._numBranches);
        scope.addMember(Id_ConstructorGenome_NumConcatenations, data._numConcatenations, defaultObject._numConcatenations);
    }
    SPLIT_SERIALIZATION(ConstructorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectEnergyGenomeDesc& data)
    {
        DetectEnergyGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorModeGenome_DetectEnergy_MinDensity, data._minDensity, defaultObject._minDensity);
    }
    REGISTER_SERIALIZED_TYPE(DetectEnergyGenomeDesc, Id_SensorModeGenome_DetectEnergy)
    SPLIT_SERIALIZATION(DetectEnergyGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectSolidGenomeDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(DetectSolidGenomeDesc, Id_SensorModeGenome_DetectSolid)
    SPLIT_SERIALIZATION(DetectSolidGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectFreeCellGenomeDesc& data)
    {
        DetectFreeCellGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorModeGenome_DetectFreeCell_MinDensity, data._minDensity, defaultObject._minDensity);
        scope.addMember(Id_SensorModeGenome_DetectFreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    REGISTER_SERIALIZED_TYPE(DetectFreeCellGenomeDesc, Id_SensorModeGenome_DetectFreeCell)
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
    REGISTER_SERIALIZED_TYPE(DetectCreatureGenomeDesc, Id_SensorModeGenome_DetectCreature)
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
    REGISTER_SERIALIZED_TYPE(SensorGenomeDesc, Id_CellTypeGenome_Sensor)
    SPLIT_SERIALIZATION(SensorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SquareSignalGenomeDesc& data)
    {
        SquareSignalGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeneratorModeGenome_SquareSignal_Period, data._period, defaultObject._period);
    }
    REGISTER_SERIALIZED_TYPE(SquareSignalGenomeDesc, Id_GeneratorModeGenome_SquareSignal)
    SPLIT_SERIALIZATION(SquareSignalGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SawtoothSignalGenomeDesc& data)
    {
        SawtoothSignalGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeneratorModeGenome_SawtoothSignal_Period, data._period, defaultObject._period);
    }
    REGISTER_SERIALIZED_TYPE(SawtoothSignalGenomeDesc, Id_GeneratorModeGenome_SawtoothSignal)
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
    REGISTER_SERIALIZED_TYPE(GeneratorGenomeDesc, Id_CellTypeGenome_Generator)
    SPLIT_SERIALIZATION(GeneratorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackFreeCellGenomeDesc& data)
    {
        AttackFreeCellGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_AttackerModeGenome_FreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    REGISTER_SERIALIZED_TYPE(AttackFreeCellGenomeDesc, Id_AttackerModeGenome_AttackFreeCell)
    SPLIT_SERIALIZATION(AttackFreeCellGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackCreatureGenomeDesc& data)
    {
        AttackCreatureGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(AttackCreatureGenomeDesc, Id_AttackerModeGenome_AttackCreature)
    SPLIT_SERIALIZATION(AttackCreatureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerGenomeDesc& data)
    {
        AttackerGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_AttackerGenome_Mode, data._mode);
    }
    REGISTER_SERIALIZED_TYPE(AttackerGenomeDesc, Id_CellTypeGenome_Attacker)
    SPLIT_SERIALIZATION(AttackerGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorGenomeDesc& data)
    {
        InjectorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_InjectorGenome_GeneIndex, data._geneIndex, defaultObject._geneIndex);
    }
    REGISTER_SERIALIZED_TYPE(InjectorGenomeDesc, Id_CellTypeGenome_Injector)
    SPLIT_SERIALIZATION(InjectorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoBendingGenomeDesc& data)
    {
        AutoBendingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_AutoBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        scope.addMember(Id_MuscleModeGenome_AutoBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
    }
    REGISTER_SERIALIZED_TYPE(AutoBendingGenomeDesc, Id_MuscleModeGenome_AutoBending)
    SPLIT_SERIALIZATION(AutoBendingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualBendingGenomeDesc& data)
    {
        ManualBendingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_ManualBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        scope.addMember(Id_MuscleModeGenome_ManualBending_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
    }
    REGISTER_SERIALIZED_TYPE(ManualBendingGenomeDesc, Id_MuscleModeGenome_ManualBending)
    SPLIT_SERIALIZATION(ManualBendingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AngleBendingGenomeDesc& data)
    {
        AngleBendingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_AngleBending_MaxAngleDeviation, data._maxAngleDeviation, defaultObject._maxAngleDeviation);
        scope.addMember(Id_MuscleModeGenome_AngleBending_AttractionRepulsionRatio, data._attractionRepulsionRatio, defaultObject._attractionRepulsionRatio);
    }
    REGISTER_SERIALIZED_TYPE(AngleBendingGenomeDesc, Id_MuscleModeGenome_AngleBending)
    SPLIT_SERIALIZATION(AngleBendingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AutoCrawlingGenomeDesc& data)
    {
        AutoCrawlingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_AutoCrawling_MaxDistanceDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        scope.addMember(Id_MuscleModeGenome_AutoCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
    }
    REGISTER_SERIALIZED_TYPE(AutoCrawlingGenomeDesc, Id_MuscleModeGenome_AutoCrawling)
    SPLIT_SERIALIZATION(AutoCrawlingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ManualCrawlingGenomeDesc& data)
    {
        ManualCrawlingGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_MuscleModeGenome_ManualCrawling_MaxDistanceDeviation, data._maxDistanceDeviation, defaultObject._maxDistanceDeviation);
        scope.addMember(Id_MuscleModeGenome_ManualCrawling_ForwardBackwardRatio, data._forwardBackwardRatio, defaultObject._forwardBackwardRatio);
    }
    REGISTER_SERIALIZED_TYPE(ManualCrawlingGenomeDesc, Id_MuscleModeGenome_ManualCrawling)
    SPLIT_SERIALIZATION(ManualCrawlingGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DirectMovementGenomeDesc& data)
    {
        DirectMovementGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(DirectMovementGenomeDesc, Id_MuscleModeGenome_DirectMovement)
    SPLIT_SERIALIZATION(DirectMovementGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, MuscleGenomeDesc& data)
    {
        MuscleGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_MuscleGenome_Mode, data._mode);
    }
    REGISTER_SERIALIZED_TYPE(MuscleGenomeDesc, Id_CellTypeGenome_Muscle)
    SPLIT_SERIALIZATION(MuscleGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderGenomeDesc& data)
    {
        DefenderGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DefenderGenome_Mode, data._mode, defaultObject._mode);
    }
    REGISTER_SERIALIZED_TYPE(DefenderGenomeDesc, Id_CellTypeGenome_Defender)
    SPLIT_SERIALIZATION(DefenderGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectSolidGenomeDesc& data)
    {
        ReconnectSolidGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(ReconnectSolidGenomeDesc, Id_ReconnectorModeGenome_ReconnectSolid)
    SPLIT_SERIALIZATION(ReconnectSolidGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectFreeCellGenomeDesc& data)
    {
        ReconnectFreeCellGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ReconnectorModeGenome_FreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    REGISTER_SERIALIZED_TYPE(ReconnectFreeCellGenomeDesc, Id_ReconnectorModeGenome_ReconnectFreeCell)
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
    REGISTER_SERIALIZED_TYPE(ReconnectCreatureGenomeDesc, Id_ReconnectorModeGenome_ReconnectCreature)
    SPLIT_SERIALIZATION(ReconnectCreatureGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorGenomeDesc& data)
    {
        ReconnectorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_ReconnectorGenome_Mode, data._mode);
    }
    REGISTER_SERIALIZED_TYPE(ReconnectorGenomeDesc, Id_CellTypeGenome_Reconnector)
    SPLIT_SERIALIZATION(ReconnectorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorGenomeDesc& data)
    {
        DetonatorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DetonatorGenome_Countdown, data._countdown, defaultObject._countdown);
    }
    REGISTER_SERIALIZED_TYPE(DetonatorGenomeDesc, Id_CellTypeGenome_Detonator)
    SPLIT_SERIALIZATION(DetonatorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DigestorGenomeDesc& data)
    {
        DigestorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_DigestorGenome_RawEnergyConductivity, data._rawEnergyConductivity, defaultObject._rawEnergyConductivity);
    }
    REGISTER_SERIALIZED_TYPE(DigestorGenomeDesc, Id_CellTypeGenome_Digestor)
    SPLIT_SERIALIZATION(DigestorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalDelayGenomeDesc& data)
    {
        SignalDelayGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalDelayGenome_Delay, data._delay, defaultObject._delay);
    }
    REGISTER_SERIALIZED_TYPE(SignalDelayGenomeDesc, Id_MemoryModeGenome_SignalDelay)
    SPLIT_SERIALIZATION(SignalDelayGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalRecorderGenomeDesc& data)
    {
        SignalRecorderGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalRecorderGenome_ReadOnly, data._readOnly, defaultObject._readOnly);
        scope.addMember(Id_SignalRecorderGenome_NumSavedSignalEntries, data._numWrittenSignalEntries, defaultObject._numWrittenSignalEntries);
    }
    REGISTER_SERIALIZED_TYPE(SignalRecorderGenomeDesc, Id_MemoryModeGenome_SignalRecorder)
    SPLIT_SERIALIZATION(SignalRecorderGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalStorageGenomeDesc& data)
    {
        SignalStorageGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalStorageGenome_ReadOnly, data._readOnly, defaultObject._readOnly);
    }
    REGISTER_SERIALIZED_TYPE(SignalStorageGenomeDesc, Id_MemoryModeGenome_SignalStorage)
    SPLIT_SERIALIZATION(SignalStorageGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalIntegratorGenomeDesc& data)
    {
        SignalIntegratorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalIntegratorGenome_NewSignalWeight, data._newSignalWeight, defaultObject._newSignalWeight);
    }
    REGISTER_SERIALIZED_TYPE(SignalIntegratorGenomeDesc, Id_MemoryModeGenome_SignalIntegrator)
    SPLIT_SERIALIZATION(SignalIntegratorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalEntryGenomeDesc& data)
    {
        SignalEntryGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addFixedSizeMember(Id_SignalEntryGenome_Channels, data._channels, defaultObject._channels);
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
    REGISTER_SERIALIZED_TYPE(MemoryGenomeDesc, Id_CellTypeGenome_Memory)
    SPLIT_SERIALIZATION(MemoryGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SenderGenomeDesc& data)
    {
        SenderGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SenderGenome_Range, data._range, defaultObject._range);
        scope.addMember(Id_SenderGenome_Oneway, data._oneway, defaultObject._oneway);
    }
    REGISTER_SERIALIZED_TYPE(SenderGenomeDesc, Id_CommunicatorModeGenome_Sender)
    SPLIT_SERIALIZATION(SenderGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReceiverGenomeDesc& data)
    {
        ReceiverGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ReceiverGenome_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
        scope.addMember(Id_ReceiverGenome_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
    }
    REGISTER_SERIALIZED_TYPE(ReceiverGenomeDesc, Id_CommunicatorModeGenome_Receiver)
    SPLIT_SERIALIZATION(ReceiverGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CommunicatorGenomeDesc& data)
    {
        CommunicatorGenomeDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_CommunicatorGenome_Mode, data._mode);
    }
    REGISTER_SERIALIZED_TYPE(CommunicatorGenomeDesc, Id_CellTypeGenome_Communicator)
    SPLIT_SERIALIZATION(CommunicatorGenomeDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, VoidGenomeDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(VoidGenomeDesc, Id_CellTypeGenome_Void)
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
    void loadSave(SerializationTask task, Archive& ar, CustomizationMutationDesc& data)
    {
        CustomizationMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_CustomizationMutation_GenomeProbability, data._genomeProbability, defaultObject._genomeProbability);
    }
    SPLIT_SERIALIZATION(CustomizationMutationDesc)

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
    void loadSave(SerializationTask task, Archive& ar, AddGeneMutationDesc& data)
    {
        AddGeneMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_AddGeneMutation_GeneProbability, data._geneProbability, defaultObject._geneProbability);
    }
    SPLIT_SERIALIZATION(AddGeneMutationDesc)

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
    void loadSave(SerializationTask task, Archive& ar, SwapGeneMutationDesc& data)
    {
        SwapGeneMutationDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SwapGeneMutation_GeneProbability, data._geneProbability, defaultObject._geneProbability);
    }
    SPLIT_SERIALIZATION(SwapGeneMutationDesc)

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
        scope.addDesc(Id_MutationRates_CustomizationMutation, data._customizationMutation);
        scope.addDesc(Id_MutationRates_VoidMutation, data._voidMutation);
        scope.addDesc(Id_MutationRates_ExtendGeneMutation, data._extendGeneMutation);
        scope.addDesc(Id_MutationRates_AddNodeMutation, data._addNodeMutation);
        scope.addDesc(Id_MutationRates_TrimGeneMutation, data._trimGeneMutation);
        scope.addDesc(Id_MutationRates_DeleteNodeMutation, data._deleteNodeMutation);
        scope.addDesc(Id_MutationRates_AddGeneMutation, data._addGeneMutation);
        scope.addDesc(Id_MutationRates_DuplicateGeneMutation, data._duplicateGeneMutation);
        scope.addDesc(Id_MutationRates_DeleteGeneMutation, data._deleteGeneMutation);
        scope.addDesc(Id_MutationRates_SwapGeneMutation, data._swapGeneMutation);
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
    auto constexpr Id_Particle_Id = SerializationKey(0, "id");
    auto constexpr Id_Particle_Pos = SerializationKey(1, "pos");
    auto constexpr Id_Particle_Vel = SerializationKey(2, "vel");
    auto constexpr Id_Particle_Energy = SerializationKey(3, "energy");
    auto constexpr Id_Particle_Color = SerializationKey(4, "color");

    auto constexpr Id_Creature_Id = SerializationKey(0, "id");
    auto constexpr Id_Creature_AncestorId = SerializationKey(1, "ancestorId");
    auto constexpr Id_Creature_Generation = SerializationKey(2, "generation");
    auto constexpr Id_Creature_NumCells = SerializationKey(4, "numCells");
    auto constexpr Id_Creature_HeadUpdateId = SerializationKey(5, "headUpdateId");
    auto constexpr Id_Creature_GenomeId = SerializationKey(6, "genomeId");
    auto constexpr Id_Creature_MutationState = SerializationKey(7, "mutationState");
    auto constexpr Id_Creature_LineageId = SerializationKey(3, "lineageId");
    auto constexpr Id_Creature_AccumulatedMutations = SerializationKey(9, "accumulatedMutations");
    auto constexpr Id_Creature_AccumulatedMutationsInLineage = SerializationKey(10, "accumulatedMutationsInLineage");
    auto constexpr Id_Creature_NextConstructionId = SerializationKey(11, "nextConstructionId");

    auto constexpr Id_Solid_Energy = SerializationKey(0, "energy");

    auto constexpr Id_Fluid_Energy = SerializationKey(0, "energy");
    auto constexpr Id_Fluid_Glow = SerializationKey(1, "glow");

    auto constexpr Id_FreeCell_Energy = SerializationKey(0, "energy");
    auto constexpr Id_FreeCell_Age = SerializationKey(1, "age");

    auto constexpr Id_Cell_UsableEnergy = SerializationKey(0, "usableEnergy");
    auto constexpr Id_Cell_RawEnergy = SerializationKey(1, "rawEnergy");
    auto constexpr Id_Cell_ReservedEnergy = SerializationKey(17, "reservedEnergy");
    auto constexpr Id_Cell_Age = SerializationKey(2, "age");
    auto constexpr Id_Cell_CellState = SerializationKey(3, "cellState");
    auto constexpr Id_Cell_ActivationTime = SerializationKey(4, "activationTime");
    auto constexpr Id_Cell_NodeIndex = SerializationKey(6, "nodeIndex");
    auto constexpr Id_Cell_GeneIndex = SerializationKey(8, "geneIndex");
    auto constexpr Id_Cell_AngleToFront = SerializationKey(10, "frontAngle");
    auto constexpr Id_Cell_HeadUpdateId = SerializationKey(11, "headUpdateId");
    auto constexpr Id_Cell_HeadCell = SerializationKey(12, "headCell");
    auto constexpr Id_Cell_CreatureId = SerializationKey(13, "creatureId");
    auto constexpr Id_Cell_Event = SerializationKey(14, "event");
    auto constexpr Id_Cell_EventCounter = SerializationKey(15, "eventCounter");
    auto constexpr Id_Cell_EventPos = SerializationKey(16, "eventPos");
    auto constexpr Id_Cell_HighlightIntensity = SerializationKey(25, "highlightIntensity");
    auto constexpr Id_Cell_LastUpdate = SerializationKey(18, "lastUpdate");
    auto constexpr Id_Cell_ConcatenationIndex = SerializationKey(19, "concatenationIndex");
    auto constexpr Id_Cell_BranchIndex = SerializationKey(20, "branchIndex");
    auto constexpr Id_Cell_ConstructionId = SerializationKey(26, "constructionId");

    auto constexpr Id_Object_Id = SerializationKey(0, "id");
    auto constexpr Id_Object_Pos = SerializationKey(2, "pos");
    auto constexpr Id_Object_Vel = SerializationKey(3, "vel");
    auto constexpr Id_Object_Stiffness = SerializationKey(4, "stiffness");
    auto constexpr Id_Object_Color = SerializationKey(5, "color");
    auto constexpr Id_Object_Static = SerializationKey(6, "isStatic");
    auto constexpr Id_Object_Sticky = SerializationKey(17, "sticky");

    auto constexpr Id_NeuralActivity_Signals = SerializationKey(0, "signals");
    auto constexpr Id_NeuralActivity_Memory = SerializationKey(1, "memory");

    auto constexpr Id_Connection_ObjectId = SerializationKey(0, "objectId");
    auto constexpr Id_Connection_Distance = SerializationKey(1, "distance");
    auto constexpr Id_Connection_AngleFromPrevious = SerializationKey(2, "angleFromPrevious");

    auto constexpr Id_NeuralNet_Weights = SerializationKey(0, "weights");
    auto constexpr Id_NeuralNet_Biases = SerializationKey(1, "biases");
    auto constexpr Id_NeuralNet_ActivationFunctions = SerializationKey(2, "activationFunctions");
    auto constexpr Id_NeuralNet_ConnectionWeights = SerializationKey(3, "connectionWeights");

    auto constexpr Id_Constructor_AutoTriggerInterval = SerializationKey(0, "autoTriggerInterval");
    auto constexpr Id_Constructor_GeneIndex = SerializationKey(2, "geneIndex");
    auto constexpr Id_Constructor_LastConstructedCellId = SerializationKey(5, "lastConstructedCellId");
    auto constexpr Id_Constructor_ConstructionAngle = SerializationKey(7, "constructionAngle");
    auto constexpr Id_Constructor_ProvideEnergy = SerializationKey(8, "provideEnergy");
    auto constexpr Id_Constructor_CurrentOffspring = SerializationKey(9, "currentOffspring");
    auto constexpr Id_Constructor_ReservedEnergy = SerializationKey(10, "reservedEnergy");
    auto constexpr Id_Constructor_Separation = SerializationKey(11, "separation");
    auto constexpr Id_Constructor_NumBranches = SerializationKey(12, "numBranches");
    auto constexpr Id_Constructor_NumConcatenations = SerializationKey(13, "numConcatenations");

    auto constexpr Id_Defender_Mode = SerializationKey(0, "mode");

    auto constexpr Id_Muscle_LastMovementX = SerializationKey(4, "lastMovementX");
    auto constexpr Id_Muscle_LastMovementY = SerializationKey(5, "lastMovementY");

    auto constexpr Id_MuscleMode_AutoBending_MaxAngleDeviation = SerializationKey(0, "maxAngleDeviation");
    auto constexpr Id_MuscleMode_AutoBending_ForwardBackwardRatio = SerializationKey(6, "forwardBackwardRatio");
    auto constexpr Id_MuscleMode_AutoBending_InitialAngle = SerializationKey(7, "initialAngle");
    auto constexpr Id_MuscleMode_AutoBending_Forward = SerializationKey(8, "forward");

    auto constexpr Id_MuscleMode_ManualBending_MaxAngleDeviation = SerializationKey(0, "maxAngleDeviation");
    auto constexpr Id_MuscleMode_ManualBending_ForwardBackwardRatio = SerializationKey(1, "forwardBackwardRatio");
    auto constexpr Id_MuscleMode_ManualBending_InitialAngle = SerializationKey(2, "initialAngle");
    auto constexpr Id_MuscleMode_ManualBending_LastAngleDelta = SerializationKey(5, "lastAngleDelta");

    auto constexpr Id_MuscleMode_AngleBending_MaxAngleDeviation = SerializationKey(0, "maxAngleDeviation");
    auto constexpr Id_MuscleMode_AngleBending_AttractionRepulsionRatio = SerializationKey(1, "attractionRepulsionRatio");
    auto constexpr Id_MuscleMode_AngleBending_InitialAngle = SerializationKey(2, "initialAngle");

    auto constexpr Id_MuscleMode_AutoCrawling_MaxAngleDeviation = SerializationKey(0, "maxDistanceDeviation");
    auto constexpr Id_MuscleMode_AutoCrawling_ForwardBackwardRatio = SerializationKey(1, "forwardBackwardRatio");
    auto constexpr Id_MuscleMode_AutoCrawling_InitialDistance = SerializationKey(2, "initialDistance");
    auto constexpr Id_MuscleMode_AutoCrawling_Forward = SerializationKey(3, "forward");
    auto constexpr Id_MuscleMode_AutoCrawling_LastActualDistance = SerializationKey(6, "lastActualDistance");

    auto constexpr Id_MuscleMode_ManualCrawling_MaxAngleDeviation = SerializationKey(0, "maxDistanceDeviation");
    auto constexpr Id_MuscleMode_ManualCrawling_ForwardBackwardRatio = SerializationKey(1, "forwardBackwardRatio");
    auto constexpr Id_MuscleMode_ManualCrawling_InitialDistance = SerializationKey(2, "initialDistance");
    auto constexpr Id_MuscleMode_ManualCrawling_LastActualDistance = SerializationKey(3, "lastActualDistance");
    auto constexpr Id_MuscleMode_ManualCrawling_LastDistanceDelta = SerializationKey(4, "lastDistanceDelta");

    auto constexpr Id_Injector_GeneIndex = SerializationKey(0, "geneIndex");

    auto constexpr Id_Generator_Additive = SerializationKey(0, "additive");
    auto constexpr Id_Generator_NumPulses = SerializationKey(1, "numPulses");
    auto constexpr Id_Generator_TimeOffset = SerializationKey(3, "timeOffset");
    auto constexpr Id_Generator_MinValue = SerializationKey(5, "minValue");
    auto constexpr Id_Generator_MaxValue = SerializationKey(6, "maxValue");

    auto constexpr Id_GeneratorMode_SquareSignal_Period = SerializationKey(1, "period");
    auto constexpr Id_GeneratorMode_SawtoothSignal_Period = SerializationKey(1, "period");

    auto constexpr Id_AttackerMode_FreeCell_RestrictToColor = SerializationKey(0, "restrictToColors");

    auto constexpr Id_Sensor_MinRange = SerializationKey(0, "minRange");
    auto constexpr Id_Sensor_MaxRange = SerializationKey(1, "maxRange");
    auto constexpr Id_Sensor_AutoTrigger = SerializationKey(2, "autoTrigger");
    auto constexpr Id_Sensor_TagForAttackers = SerializationKey(3, "tagForAttackers");

    auto constexpr Id_SensorMode_DetectEnergy_MinDensity = SerializationKey(0, "minDensity");

    auto constexpr Id_SensorMode_DetectFreeCell_MinDensity = SerializationKey(0, "minDensity");
    auto constexpr Id_SensorMode_DetectFreeCell_RestrictToColor = SerializationKey(1, "restrictToColors");

    auto constexpr Id_SensorMode_SensorLastMatch_CreatureIdPart = SerializationKey(0, "creatureIdPart");
    auto constexpr Id_SensorMode_SensorLastMatch_Pos = SerializationKey(1, "pos");

    auto constexpr Id_SensorMode_DetectCreature_MinNumCells = SerializationKey(0, "minNumCells");
    auto constexpr Id_SensorMode_DetectCreature_MaxNumCells = SerializationKey(1, "maxNumCells");
    auto constexpr Id_SensorMode_DetectCreature_RestrictToColor = SerializationKey(2, "restrictToColors");
    auto constexpr Id_SensorMode_DetectCreature_RestrictToLineage = SerializationKey(3, "restrictToLineage");

    auto constexpr Id_Depot_storageLimit = SerializationKey(1, "storageLimit");
    auto constexpr Id_Depot_StoredUsableEnergy = SerializationKey(2, "storedUsableEnergy");

    auto constexpr Id_ReconnectorMode_FreeCell_RestrictToColor = SerializationKey(0, "restrictToColors");

    auto constexpr Id_ReconnectorMode_Creature_MinNumCells = SerializationKey(0, "minNumCells");
    auto constexpr Id_ReconnectorMode_Creature_MaxNumCells = SerializationKey(1, "maxNumCells");
    auto constexpr Id_ReconnectorMode_Creature_RestrictToColor = SerializationKey(2, "restrictToColors");
    auto constexpr Id_ReconnectorMode_Creature_RestrictToLineage = SerializationKey(3, "restrictToLineage");

    auto constexpr Id_Detonator_State = SerializationKey(0, "state");
    auto constexpr Id_Detonator_Countdown = SerializationKey(1, "countdown");

    auto constexpr Id_Digestor_RawEnergyConductivity = SerializationKey(0, "rawEnergyConductivity");

    auto constexpr Id_SignalEntry_Channels = SerializationKey(0, "channels");

    auto constexpr Id_SignalDelay_Delay = SerializationKey(0, "delay");
    auto constexpr Id_SignalDelay_NumMemoryEntriesInitialized = SerializationKey(1, "numSignalEntriesInitialized");
    auto constexpr Id_SignalDelay_RingBufferIndex = SerializationKey(2, "ringBufferIndex");

    auto constexpr Id_SignalRecorder_ReadOnly = SerializationKey(0, "readOnly");
    auto constexpr Id_SignalRecorder_State = SerializationKey(1, "state");
    auto constexpr Id_SignalRecorder_NumSavedSignalEntries = SerializationKey(2, "numWrittenSignalEntries");
    auto constexpr Id_SignalRecorder_NumReadSignalEntries = SerializationKey(3, "numReadSignalEntries");

    auto constexpr Id_SignalStorage_ReadOnly = SerializationKey(0, "readOnly");

    auto constexpr Id_SignalIntegrator_NewSignalWeight = SerializationKey(0, "newSignalWeight");

    auto constexpr Id_Memory_ChannelBitMask = SerializationKey(0, "channelBitMask");

    auto constexpr Id_Sender_Range = SerializationKey(0, "range");
    auto constexpr Id_Sender_Oneway = SerializationKey(3, "oneway");

    auto constexpr Id_Receiver_RestrictToColor = SerializationKey(1, "restrictToColors");
    auto constexpr Id_Receiver_RestrictToLineage = SerializationKey(2, "restrictToLineage");

    // Description member keys for objects data
    auto constexpr Id_Cell_CellType = SerializationKey(21, "cellType");
    auto constexpr Id_Cell_Constructor = SerializationKey(22, "constructor");
    auto constexpr Id_Cell_NeuralActivity = SerializationKey(23, "neuralActivity");
    auto constexpr Id_Cell_NeuralNetwork = SerializationKey(24, "neuralNetwork");

    auto constexpr Id_Object_Connections = SerializationKey(7, "connections");
    auto constexpr Id_Object_Type = SerializationKey(8, "type");

    auto constexpr Id_Sensor_Mode = SerializationKey(4, "mode");
    auto constexpr Id_Sensor_LastMatch = SerializationKey(5, "lastMatch");
    auto constexpr Id_Generator_Mode = SerializationKey(4, "mode");
    auto constexpr Id_Attacker_Mode = SerializationKey(0, "mode");
    auto constexpr Id_Muscle_Mode = SerializationKey(3, "mode");
    auto constexpr Id_Reconnector_Mode = SerializationKey(0, "mode");
    auto constexpr Id_Memory_Mode = SerializationKey(1, "mode");
    auto constexpr Id_Memory_SignalEntries = SerializationKey(2, "signalEntries");
    auto constexpr Id_Communicator_Mode = SerializationKey(0, "mode");

    auto constexpr Id_Desc_Objects = SerializationKey(0, "objects");
    auto constexpr Id_Desc_Energies = SerializationKey(1, "energies");
    auto constexpr Id_Desc_Creatures = SerializationKey(2, "creatures");
    auto constexpr Id_Desc_Genomes = SerializationKey(3, "genomes");

    // Serialized type ids
    auto constexpr Id_ObjectType_Solid = SerializationKey(0, "solid");
    auto constexpr Id_ObjectType_Fluid = SerializationKey(1, "fluid");
    auto constexpr Id_ObjectType_FreeCell = SerializationKey(2, "freeCell");
    auto constexpr Id_ObjectType_Cell = SerializationKey(3, "cell");

    auto constexpr Id_CellType_Base = SerializationKey(0, "base");
    auto constexpr Id_CellType_Depot = SerializationKey(1, "depot");
    auto constexpr Id_CellType_Sensor = SerializationKey(2, "sensor");
    auto constexpr Id_CellType_Generator = SerializationKey(3, "generator");
    auto constexpr Id_CellType_Attacker = SerializationKey(4, "attacker");
    auto constexpr Id_CellType_Injector = SerializationKey(5, "injector");
    auto constexpr Id_CellType_Muscle = SerializationKey(6, "muscle");
    auto constexpr Id_CellType_Defender = SerializationKey(7, "defender");
    auto constexpr Id_CellType_Reconnector = SerializationKey(8, "reconnector");
    auto constexpr Id_CellType_Detonator = SerializationKey(9, "detonator");
    auto constexpr Id_CellType_Digestor = SerializationKey(10, "digestor");
    auto constexpr Id_CellType_Memory = SerializationKey(11, "memory");
    auto constexpr Id_CellType_Communicator = SerializationKey(12, "communicator");
    auto constexpr Id_CellType_Void = SerializationKey(13, "void");

    auto constexpr Id_SensorMode_DetectEnergy = SerializationKey(0, "detectEnergy");
    auto constexpr Id_SensorMode_DetectSolid = SerializationKey(1, "detectSolid");
    auto constexpr Id_SensorMode_DetectFreeCell = SerializationKey(2, "detectFreeCell");
    auto constexpr Id_SensorMode_DetectCreature = SerializationKey(3, "detectCreature");

    auto constexpr Id_GeneratorMode_SquareSignal = SerializationKey(0, "squareSignal");
    auto constexpr Id_GeneratorMode_SawtoothSignal = SerializationKey(1, "sawtoothSignal");

    auto constexpr Id_AttackerMode_AttackFreeCell = SerializationKey(0, "attackFreeCell");
    auto constexpr Id_AttackerMode_AttackCreature = SerializationKey(1, "attackCreature");

    auto constexpr Id_MuscleMode_AutoBending = SerializationKey(0, "autoBending");
    auto constexpr Id_MuscleMode_ManualBending = SerializationKey(1, "manualBending");
    auto constexpr Id_MuscleMode_AngleBending = SerializationKey(2, "angleBending");
    auto constexpr Id_MuscleMode_AutoCrawling = SerializationKey(3, "autoCrawling");
    auto constexpr Id_MuscleMode_ManualCrawling = SerializationKey(4, "manualCrawling");
    auto constexpr Id_MuscleMode_DirectMovement = SerializationKey(5, "directMovement");

    auto constexpr Id_ReconnectorMode_ReconnectSolid = SerializationKey(0, "reconnectSolid");
    auto constexpr Id_ReconnectorMode_ReconnectFreeCell = SerializationKey(1, "reconnectFreeCell");
    auto constexpr Id_ReconnectorMode_ReconnectCreature = SerializationKey(2, "reconnectCreature");

    auto constexpr Id_MemoryMode_SignalDelay = SerializationKey(0, "signalDelay");
    auto constexpr Id_MemoryMode_SignalRecorder = SerializationKey(1, "signalRecorder");
    auto constexpr Id_MemoryMode_SignalStorage = SerializationKey(2, "signalStorage");
    auto constexpr Id_MemoryMode_SignalIntegrator = SerializationKey(3, "signalIntegrator");

    auto constexpr Id_CommunicatorMode_Sender = SerializationKey(0, "sender");
    auto constexpr Id_CommunicatorMode_Receiver = SerializationKey(1, "receiver");
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
    void loadSave(SerializationTask task, Archive& ar, NeuralActivityDesc& data)
    {
        NeuralActivityDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addFixedSizeMember(Id_NeuralActivity_Signals, data._signals, defaultObject._signals);
        scope.addFixedSizeMember(Id_NeuralActivity_Memory, data._memory, defaultObject._memory);
    }
    SPLIT_SERIALIZATION(NeuralActivityDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, NeuralNetDesc& data)
    {
        NeuralNetDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addFixedSizeMember(Id_NeuralNet_Weights, data._weights, defaultObject._weights);
        scope.addFixedSizeMember(Id_NeuralNet_Biases, data._biases, defaultObject._biases);
        scope.addFixedSizeMember(Id_NeuralNet_ActivationFunctions, data._activationFunctions, defaultObject._activationFunctions);
        scope.addFixedSizeMember(Id_NeuralNet_ConnectionWeights, data._connectionWeights, defaultObject._connectionWeights);
    }
    SPLIT_SERIALIZATION(NeuralNetDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, BaseDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(BaseDesc, Id_CellType_Base)
    SPLIT_SERIALIZATION(BaseDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DepotDesc& data)
    {
        DepotDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Depot_storageLimit, data._storageLimit, defaultObject._storageLimit);
        scope.addMember(Id_Depot_StoredUsableEnergy, data._storedUsableEnergy, defaultObject._storedUsableEnergy);
    }
    REGISTER_SERIALIZED_TYPE(DepotDesc, Id_CellType_Depot)
    SPLIT_SERIALIZATION(DepotDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ConstructorDesc& data)
    {
        ConstructorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Constructor_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
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
    void loadSave(SerializationTask task, Archive& ar, DetectEnergyDesc& data)
    {
        DetectEnergyDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorMode_DetectEnergy_MinDensity, data._minDensity, defaultObject._minDensity);
    }
    REGISTER_SERIALIZED_TYPE(DetectEnergyDesc, Id_SensorMode_DetectEnergy)
    SPLIT_SERIALIZATION(DetectEnergyDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectSolidDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(DetectSolidDesc, Id_SensorMode_DetectSolid)
    SPLIT_SERIALIZATION(DetectSolidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetectFreeCellDesc& data)
    {
        DetectFreeCellDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SensorMode_DetectFreeCell_MinDensity, data._minDensity, defaultObject._minDensity);
        scope.addMember(Id_SensorMode_DetectFreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    REGISTER_SERIALIZED_TYPE(DetectFreeCellDesc, Id_SensorMode_DetectFreeCell)
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
    REGISTER_SERIALIZED_TYPE(DetectCreatureDesc, Id_SensorMode_DetectCreature)
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
    REGISTER_SERIALIZED_TYPE(SensorDesc, Id_CellType_Sensor)
    SPLIT_SERIALIZATION(SensorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SquareSignalDesc& data)
    {
        SquareSignalDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeneratorMode_SquareSignal_Period, data._period, defaultObject._period);
    }
    REGISTER_SERIALIZED_TYPE(SquareSignalDesc, Id_GeneratorMode_SquareSignal)
    SPLIT_SERIALIZATION(SquareSignalDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SawtoothSignalDesc& data)
    {
        SawtoothSignalDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_GeneratorMode_SawtoothSignal_Period, data._period, defaultObject._period);
    }
    REGISTER_SERIALIZED_TYPE(SawtoothSignalDesc, Id_GeneratorMode_SawtoothSignal)
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
    REGISTER_SERIALIZED_TYPE(GeneratorDesc, Id_CellType_Generator)
    SPLIT_SERIALIZATION(GeneratorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackFreeCellDesc& data)
    {
        AttackFreeCellDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_AttackerMode_FreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    REGISTER_SERIALIZED_TYPE(AttackFreeCellDesc, Id_AttackerMode_AttackFreeCell)
    SPLIT_SERIALIZATION(AttackFreeCellDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackCreatureDesc& data)
    {
        AttackCreatureDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(AttackCreatureDesc, Id_AttackerMode_AttackCreature)
    SPLIT_SERIALIZATION(AttackCreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, AttackerDesc& data)
    {
        AttackerDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_Attacker_Mode, data._mode);
    }
    REGISTER_SERIALIZED_TYPE(AttackerDesc, Id_CellType_Attacker)
    SPLIT_SERIALIZATION(AttackerDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, InjectorDesc& data)
    {
        InjectorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Injector_GeneIndex, data._geneIndex, defaultObject._geneIndex);
    }
    REGISTER_SERIALIZED_TYPE(InjectorDesc, Id_CellType_Injector)
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
    REGISTER_SERIALIZED_TYPE(AutoBendingDesc, Id_MuscleMode_AutoBending)
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
    REGISTER_SERIALIZED_TYPE(ManualBendingDesc, Id_MuscleMode_ManualBending)
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
    REGISTER_SERIALIZED_TYPE(AngleBendingDesc, Id_MuscleMode_AngleBending)
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
    REGISTER_SERIALIZED_TYPE(AutoCrawlingDesc, Id_MuscleMode_AutoCrawling)
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
    REGISTER_SERIALIZED_TYPE(ManualCrawlingDesc, Id_MuscleMode_ManualCrawling)
    SPLIT_SERIALIZATION(ManualCrawlingDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DirectMovementDesc& data)
    {
        DirectMovementDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(DirectMovementDesc, Id_MuscleMode_DirectMovement)
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
    REGISTER_SERIALIZED_TYPE(MuscleDesc, Id_CellType_Muscle)
    SPLIT_SERIALIZATION(MuscleDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DefenderDesc& data)
    {
        DefenderDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Defender_Mode, data._mode, defaultObject._mode);
    }
    REGISTER_SERIALIZED_TYPE(DefenderDesc, Id_CellType_Defender)
    SPLIT_SERIALIZATION(DefenderDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectSolidDesc& data)
    {
        ReconnectSolidDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(ReconnectSolidDesc, Id_ReconnectorMode_ReconnectSolid)
    SPLIT_SERIALIZATION(ReconnectSolidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectFreeCellDesc& data)
    {
        ReconnectFreeCellDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_ReconnectorMode_FreeCell_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
    }
    REGISTER_SERIALIZED_TYPE(ReconnectFreeCellDesc, Id_ReconnectorMode_ReconnectFreeCell)
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
    REGISTER_SERIALIZED_TYPE(ReconnectCreatureDesc, Id_ReconnectorMode_ReconnectCreature)
    SPLIT_SERIALIZATION(ReconnectCreatureDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReconnectorDesc& data)
    {
        ReconnectorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_Reconnector_Mode, data._mode);
    }
    REGISTER_SERIALIZED_TYPE(ReconnectorDesc, Id_CellType_Reconnector)
    SPLIT_SERIALIZATION(ReconnectorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DetonatorDesc& data)
    {
        DetonatorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Detonator_State, data._state, defaultObject._state);
        scope.addMember(Id_Detonator_Countdown, data._countdown, defaultObject._countdown);
    }
    REGISTER_SERIALIZED_TYPE(DetonatorDesc, Id_CellType_Detonator)
    SPLIT_SERIALIZATION(DetonatorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, DigestorDesc& data)
    {
        DigestorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Digestor_RawEnergyConductivity, data._rawEnergyConductivity, defaultObject._rawEnergyConductivity);
    }
    REGISTER_SERIALIZED_TYPE(DigestorDesc, Id_CellType_Digestor)
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
    REGISTER_SERIALIZED_TYPE(SignalDelayDesc, Id_MemoryMode_SignalDelay)
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
    REGISTER_SERIALIZED_TYPE(SignalRecorderDesc, Id_MemoryMode_SignalRecorder)
    SPLIT_SERIALIZATION(SignalRecorderDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalStorageDesc& data)
    {
        SignalStorageDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalStorage_ReadOnly, data._readOnly, defaultObject._readOnly);
    }
    REGISTER_SERIALIZED_TYPE(SignalStorageDesc, Id_MemoryMode_SignalStorage)
    SPLIT_SERIALIZATION(SignalStorageDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalIntegratorDesc& data)
    {
        SignalIntegratorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_SignalIntegrator_NewSignalWeight, data._newSignalWeight, defaultObject._newSignalWeight);
    }
    REGISTER_SERIALIZED_TYPE(SignalIntegratorDesc, Id_MemoryMode_SignalIntegrator)
    SPLIT_SERIALIZATION(SignalIntegratorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SignalEntryDesc& data)
    {
        SignalEntryDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addFixedSizeMember(Id_SignalEntry_Channels, data._channels, defaultObject._channels);
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
    REGISTER_SERIALIZED_TYPE(MemoryDesc, Id_CellType_Memory)
    SPLIT_SERIALIZATION(MemoryDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SenderDesc& data)
    {
        SenderDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Sender_Range, data._range, defaultObject._range);
        scope.addMember(Id_Sender_Oneway, data._oneway, defaultObject._oneway);
    }
    REGISTER_SERIALIZED_TYPE(SenderDesc, Id_CommunicatorMode_Sender)
    SPLIT_SERIALIZATION(SenderDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, ReceiverDesc& data)
    {
        ReceiverDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Receiver_RestrictToColor, data._restrictToColors, defaultObject._restrictToColors);
        scope.addMember(Id_Receiver_RestrictToLineage, data._restrictToLineage, defaultObject._restrictToLineage);
    }
    REGISTER_SERIALIZED_TYPE(ReceiverDesc, Id_CommunicatorMode_Receiver)
    SPLIT_SERIALIZATION(ReceiverDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, CommunicatorDesc& data)
    {
        CommunicatorDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_Communicator_Mode, data._mode);
    }
    REGISTER_SERIALIZED_TYPE(CommunicatorDesc, Id_CellType_Communicator)
    SPLIT_SERIALIZATION(CommunicatorDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, VoidDesc& data)
    {
        auto scope = getSerializationScope(task, ar);
    }
    REGISTER_SERIALIZED_TYPE(VoidDesc, Id_CellType_Void)
    SPLIT_SERIALIZATION(VoidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SolidDesc& data)
    {
        SolidDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Solid_Energy, data._energy, defaultObject._energy);
    }
    REGISTER_SERIALIZED_TYPE(SolidDesc, Id_ObjectType_Solid)
    SPLIT_SERIALIZATION(SolidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, FluidDesc& data)
    {
        FluidDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_Fluid_Energy, data._energy, defaultObject._energy);
        scope.addMember(Id_Fluid_Glow, data._glow, defaultObject._glow);
    }
    REGISTER_SERIALIZED_TYPE(FluidDesc, Id_ObjectType_Fluid)
    SPLIT_SERIALIZATION(FluidDesc)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, FreeCellDesc& data)
    {
        FreeCellDesc defaultObject;
        auto scope = getSerializationScope(task, ar);
        scope.addMember(Id_FreeCell_Energy, data._energy, defaultObject._energy);
        scope.addMember(Id_FreeCell_Age, data._age, defaultObject._age);
    }
    REGISTER_SERIALIZED_TYPE(FreeCellDesc, Id_ObjectType_FreeCell)
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
        scope.addMember(Id_Cell_ConcatenationIndex, data._concatenationIndex, defaultObject._concatenationIndex);
        scope.addMember(Id_Cell_BranchIndex, data._branchIndex, defaultObject._branchIndex);
        scope.addMember(Id_Cell_GeneIndex, data._geneIndex, defaultObject._geneIndex);
        scope.addMember(Id_Cell_ConstructionId, data._constructionId, defaultObject._constructionId);
        scope.addMember(Id_Cell_HeadUpdateId, data._headUpdateId, defaultObject._headUpdateId);
        scope.addMember(Id_Cell_HeadCell, data._headCell, defaultObject._headCell);
        scope.addMember(Id_Cell_CreatureId, data._creatureId, defaultObject._creatureId);
        scope.addMember(Id_Cell_Event, data._event, defaultObject._event);
        scope.addMember(Id_Cell_EventCounter, data._eventCounter, defaultObject._eventCounter);
        scope.addMember(Id_Cell_HighlightIntensity, data._highlightIntensity, defaultObject._highlightIntensity);
        scope.addMember(Id_Cell_EventPos, data._eventPos, defaultObject._eventPos);
        scope.addMember(Id_Cell_LastUpdate, data._lastUpdate, defaultObject._lastUpdate);
        scope.addDesc(Id_Cell_CellType, data._cellType);
        scope.addDesc(Id_Cell_Constructor, data._constructor);
        scope.addDesc(Id_Cell_NeuralActivity, data._neuralActivity);
        scope.addDesc(Id_Cell_NeuralNetwork, data._neuralNetwork);
    }
    REGISTER_SERIALIZED_TYPE(CellDesc, Id_ObjectType_Cell)
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
        scope.addMember(Id_Creature_NextConstructionId, data._nextConstructionId, defaultObject._nextConstructionId);
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
    void loadSave(SerializationTask task, Archive& ar, ContentDesc& description)
    {
        auto scope = getSerializationScope(task, ar);
        scope.addDesc(Id_Desc_Objects, description._objects);
        scope.addDesc(Id_Desc_Energies, description._energies);
        scope.addDesc(Id_Desc_Creatures, description._creatures);
        scope.addDesc(Id_Desc_Genomes, description._genomes);
    }
    SPLIT_SERIALIZATION(ContentDesc)
}

bool SerializerService::serializeSimulationToFiles(std::filesystem::path const& filename, SimulationDesc const& data) const
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

bool SerializerService::deserializeSimulationFromFiles(SimulationDesc& data, std::filesystem::path const& filename) const
{
    try {
        log(Priority::Important, "load simulation from " + filename.string());

        zstd::ifstream stream(filename.string(), std::ios::binary);
        if (!stream) {
            return false;
        }
        deserializeSimulation(data, stream);

        ParametersValidationService::get().validateAndCorrect({data._worldSize}, data._simulationParameters);
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

bool SerializerService::serializeSimulationToString(std::string& output, SimulationDesc const& input) const
{
    try {
        std::stringstream stdStream;
        zstd::ostream stream(stdStream, zstd::DefaultCompressionLevel, zstd::recommendedWorkerCount());
        if (!stream) {
            return false;
        }
        serializeSimulation(input, stream);
        stream.flush();
        output = stdStream.str();
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeSimulationFromString(SimulationDesc& output, std::string const& input) const
{
    try {
        std::stringstream stdStream(input);
        zstd::istream stream(stdStream);
        if (!stream) {
            return false;
        }
        deserializeSimulation(output, stream);

        ParametersValidationService::get().validateAndCorrect({output._worldSize}, output._simulationParameters);
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
        ContentDesc description;
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
        ContentDesc description;
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

bool SerializerService::serializeGenomeToString(std::string& output, GenomeDesc const& genome) const
{
    try {
        std::stringstream stdStream;
        zstd::ostream stream(stdStream);
        if (!stream) {
            return false;
        }

        ContentDesc description;
        if (!wrapGenome(description, genome)) {
            return false;
        }

        serializeDescription(description, stream);
        stream.flush();
        output = stdStream.str();
        return true;
    } catch (...) {
        return false;
    }
}

bool SerializerService::deserializeGenomeFromString(GenomeDesc& genome, std::string const& input) const
{
    try {
        std::stringstream stdStream(input);
        zstd::istream stream(stdStream);
        if (!stream) {
            return false;
        }

        ContentDesc description;
        deserializeDescription(description, stream);

        return unwrapGenome(genome, description);
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

bool SerializerService::serializeContentToFile(std::filesystem::path const& filename, ContentDesc const& content) const
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

bool SerializerService::deserializeContentFromFile(ContentDesc& content, std::filesystem::path const& filename) const
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

namespace
{
    template <typename T>
    boost::json::value serializeDescToJson(T const& desc, JsonSerializationSettings const& settings)
    {
        return cereal::saveDescToJson(desc, settings, "");
    }
}

boost::json::value SerializerService::serializeToJson(ObjectDesc const& object, bool omitDefaultValues) const
{
    return serializeDescToJson(object, {.omitDefaultValues = omitDefaultValues});
}

boost::json::value SerializerService::serializeToJson(EnergyDesc const& energy, bool omitDefaultValues) const
{
    return serializeDescToJson(energy, {.omitDefaultValues = omitDefaultValues});
}

boost::json::value SerializerService::serializeToJson(CreatureDesc const& creature, bool omitDefaultValues) const
{
    return serializeDescToJson(creature, {.omitDefaultValues = omitDefaultValues});
}

boost::json::value SerializerService::serializeToJson(GenomeDesc const& genome, bool omitDefaultValues) const
{
    return serializeDescToJson(genome, {.omitDefaultValues = omitDefaultValues});
}

namespace
{
    template <typename T>
    void deserializeDescFromJson(T& desc, boost::json::value const& json, bool patch)
    {
        auto result = patch ? desc : T();
        cereal::loadDescFromJson(json, result, "", patch);
        desc = std::move(result);
    }
}

void SerializerService::deserializeFromJson(ObjectDesc& object, boost::json::value const& json, bool patch) const
{
    deserializeDescFromJson(object, json, patch);
}

void SerializerService::deserializeFromJson(EnergyDesc& energy, boost::json::value const& json, bool patch) const
{
    deserializeDescFromJson(energy, json, patch);
}

void SerializerService::deserializeFromJson(CreatureDesc& creature, boost::json::value const& json, bool patch) const
{
    deserializeDescFromJson(creature, json, patch);
}

void SerializerService::deserializeFromJson(GenomeDesc& genome, boost::json::value const& json, bool patch) const
{
    deserializeDescFromJson(genome, json, patch);
}

boost::json::value SerializerService::getJsonFormatOfObject() const
{
    return serializeDescToJson(ObjectDesc(), {.omitDefaultValues = false, .describeFormat = true});
}

boost::json::value SerializerService::getJsonFormatOfGenome() const
{
    return serializeDescToJson(GenomeDesc(), {.omitDefaultValues = false, .describeFormat = true});
}

void SerializerService::serializeDescription(ContentDesc const& description, std::ostream& stream) const
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(Const::ProgramVersion);
    archive(description);
}

bool SerializerService::deserializeDescription(ContentDesc& description, std::filesystem::path const& filename) const
{
    zstd::ifstream stream(filename.string(), std::ios::binary);
    if (!stream) {
        return false;
    }
    deserializeDescription(description, stream);
    return true;
}

void SerializerService::deserializeDescription(ContentDesc& description, std::istream& stream) const
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

    DeserializationContext context;
    archive(description);
    context.throwOnFailure();
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

    std::unordered_map<uint32_t, double> countCreaturesByLineage(ContentDesc const& mainData)
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

    StatisticsTimelines convertToTimelines(StatisticsHistoryData const& statistics)
    {
        StatisticsTimelines result;
        result.overallTimeline = convertToTimeline(statistics.colors);
        for (auto const& [lineageId, samples] : statistics.lineages) {
            result.lineageTimelines.emplace(lineageId, convertToTimeline(samples));
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
    void loadSave(SerializationTask task, Archive& ar, StatisticsHistoryData& data)
    {
        StatisticsTimelines timelines;
        if (task == SerializationTask::Save) {
            timelines = convertToTimelines(data);
        }
        {
            auto scope = getSerializationScope(task, ar);
            scope.addDesc(Id_StatisticsHistory_Overall, timelines.overallTimeline);
            scope.addDesc(Id_StatisticsHistory_Lineages, timelines.lineageTimelines);
        }
        if (task == SerializationTask::Load) {
            data = convertToStatisticsHistory(timelines);
        }
    }
    SPLIT_SERIALIZATION(StatisticsHistoryData)
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
    auto constexpr Id_Simulation_Content = 107;

    auto constexpr Id_SimulationParameters_Encoded = 0;

    StatisticsHistoryData selectStatisticsToSave(StatisticsHistoryData const& statistics, ContentDesc const& mainData)
    {
        StatisticsHistoryData result;
        result.colors = statistics.colors;
        for (auto const lineageId : selectLineagesToSave(statistics.lineages, countCreaturesByLineage(mainData))) {
            result.lineages.emplace(lineageId, statistics.lineages.at(lineageId));
        }
        return result;
    }
}

namespace cereal
{
    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SimulationParameters& data)
    {
        std::string encodedParameters;
        if (task == SerializationTask::Save) {
            encodedParameters = SettingsParserService::get().encodeSimulationParametersToString(data);
        }
        {
            auto scope = getSerializationScope(task, ar);
            scope.addMember(Id_SimulationParameters_Encoded, encodedParameters, std::string());
        }
        if (task == SerializationTask::Load && !encodedParameters.empty()) {
            data = SettingsParserService::get().decodeSimulationParametersFromString(encodedParameters);
        }
    }
    SPLIT_SERIALIZATION(SimulationParameters)

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, SimulationDesc& data)
    {
        StatisticsHistoryData statistics;
        if (task == SerializationTask::Save) {
            statistics = selectStatisticsToSave(data._statistics, data._mainData);
        }
        {
            SimulationDesc defaultData;
            auto scope = getSerializationScope(task, ar);

            scope.addMember(Id_Simulation_Timestep, data._timestep, defaultData._timestep);
            scope.addMember(Id_Simulation_RealTime, data._realTime, defaultData._realTime);
            scope.addMember(Id_Simulation_Zoom, data._zoom, defaultData._zoom);
            scope.addMember(Id_Simulation_Center, data._center, defaultData._center);
            scope.addMember(Id_Simulation_WorldSize, data._worldSize, defaultData._worldSize);

            scope.addDesc(Id_Simulation_SimulationParameters, data._simulationParameters);
            scope.addDesc(Id_Simulation_Content, data._mainData);
            scope.addDesc(Id_Simulation_Statistics, statistics);
        }
        if (task == SerializationTask::Load) {
            data._statistics = std::move(statistics);
        }
    }
    SPLIT_SERIALIZATION(SimulationDesc)
}

void SerializerService::serializeSimulation(SimulationDesc const& data, std::ostream& stream) const
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(Const::ProgramVersion);
    archive(data);
}

void SerializerService::deserializeSimulation(SimulationDesc& data, std::istream& stream) const
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

    DeserializationContext context;
    archive(data);
    context.throwOnFailure();
}

bool SerializerService::wrapGenome(ContentDesc& output, GenomeDesc const& input) const
{
    output.clear();
    output._genomes.emplace_back(input);
    output._creatures.emplace_back(CreatureDesc().genomeId(input._id));
    return true;
}

bool SerializerService::unwrapGenome(GenomeDesc& output, ContentDesc& input) const
{
    if (input._genomes.size() != 1) {
        return false;
    }
    output = input._genomes.front();
    return true;
}
