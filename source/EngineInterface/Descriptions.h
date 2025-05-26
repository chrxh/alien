#pragma once

#include <variant>
#include <optional>
#include <string>
#include <unordered_set>

#include "Base/Macros.h"
#include "Base/Vector2D.h"
#include "EngineInterface/EngineConstants.h"

#include "Definitions.h"
#include "GenomeDescriptions.h"

struct CellMetadataDescription
{
    auto operator<=>(CellMetadataDescription const&) const = default;

    MEMBER(CellMetadataDescription, std::string, name, "");
    MEMBER(CellMetadataDescription, std::string, description, "");
};

struct ConnectionDescription
{
    auto operator<=>(ConnectionDescription const&) const = default;

    MEMBER(ConnectionDescription, uint64_t, cellId, 0ull);  // value of 0 means cell not present in CollectionDescription
    MEMBER(ConnectionDescription, float, distance, 0.0f);
    MEMBER(ConnectionDescription, float, angleFromPrevious, 0.0f);
};

struct StructureCellDescription
{
    auto operator<=>(StructureCellDescription const&) const = default;
};

struct FreeCellDescription
{
    auto operator<=>(FreeCellDescription const&) const = default;
};

struct NeuralNetworkDescription
{
    NeuralNetworkDescription()
    {
        _weights.resize(MAX_CHANNELS * MAX_CHANNELS, 0);
        _biases.resize(MAX_CHANNELS, 0);
        _activationFunctions.resize(MAX_CHANNELS, ActivationFunction_Identity);
        // #TODO GCC incompatibily:
        // auto md = std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            // #TODO GCC incompatibily:
            // md[i, i] = 1.0f;
            _weights[i * MAX_CHANNELS + i] = 1.0f;
        }
    }
    auto operator<=>(NeuralNetworkDescription const&) const = default;

    MEMBER(NeuralNetworkDescription, std::vector<float>, weights, {});
    MEMBER(NeuralNetworkDescription, std::vector<float>, biases, {});
    MEMBER(NeuralNetworkDescription, std::vector<ActivationFunction>, activationFunctions, {});

    NeuralNetworkDescription& weight(int row, int col, float value)
    {
        // #TODO GCC incompatibily:
        // auto md = std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        // md[row, col] = value;
        _weights[row * MAX_CHANNELS + col] = value;
        return *this;
    }
    // #TODO GCC incompatibily:
    // auto getWeights() const { return std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS); }
    // auto getWeights() { return std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS); }
};

struct BaseDescription
{
    auto operator<=>(BaseDescription const&) const = default;
};

struct DepotDescription
{
    auto operator<=>(DepotDescription const&) const = default;

    MEMBER(DepotDescription, EnergyDistributionMode, mode, EnergyDistributionMode_TransmittersAndConstructors);
};

struct ConstructorDescription
{
    ConstructorDescription();
    auto operator<=>(ConstructorDescription const&) const = default;

    // Properties
    MEMBER(ConstructorDescription, int, autoTriggerInterval, 100);  // 0 = manual (triggered by signal), > 0 = auto trigger
    MEMBER(ConstructorDescription, int, constructionActivationTime, 100);

    // Genome data
    MEMBER(ConstructorDescription, std::vector<uint8_t>, genome, {});
    MEMBER(ConstructorDescription, int, numInheritedGenomeNodes, 0);
    MEMBER(ConstructorDescription, int, genomeGeneration, 0);
    MEMBER(ConstructorDescription, float, constructionAngle1, 0);
    MEMBER(ConstructorDescription, float, constructionAngle2, 0);

    // Process data
    MEMBER(ConstructorDescription, uint64_t, lastConstructedCellId, 0);
    MEMBER(ConstructorDescription, int, genomeCurrentNodeIndex, 0);
    MEMBER(ConstructorDescription, int, genomeCurrentRepetition, 0);
    MEMBER(ConstructorDescription, int, genomeCurrentBranch, 0);
    MEMBER(ConstructorDescription, int, offspringCreatureId, 0);
    MEMBER(ConstructorDescription, int, offspringMutationId, 0);

    bool isGenomeInherited() const { return _numInheritedGenomeNodes != 0; }
};

struct SensorDescription
{
    auto operator<=>(SensorDescription const&) const = default;

    MEMBER(SensorDescription, int, autoTriggerInterval, 100);  // 0 = manual (triggered by signal), > 0 = auto trigger
    MEMBER(SensorDescription, float, minDensity, 0.05f);
    MEMBER(SensorDescription, std::optional<int>, minRange, std::nullopt);
    MEMBER(SensorDescription, std::optional<int>, maxRange, std::nullopt);
    MEMBER(SensorDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(SensorDescription, SensorRestrictToMutants, restrictToMutants, SensorRestrictToMutants_NoRestriction);
};

struct OscillatorDescription
{
    auto operator<=>(OscillatorDescription const&) const = default;

    // Fixed data
    MEMBER(OscillatorDescription, int, autoTriggerInterval, 100);
    MEMBER(OscillatorDescription, int, alternationInterval, 0);  // 0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.

    // Process data
    MEMBER(OscillatorDescription, int, numPulses, 0);
};

struct AttackerDescription
{
    auto operator<=>(AttackerDescription const&) const = default;
};

struct InjectorDescription
{
    InjectorDescription();
    auto operator<=>(InjectorDescription const&) const = default;

    MEMBER(InjectorDescription, InjectorMode, mode, InjectorMode_InjectAll);
    MEMBER(InjectorDescription, int, counter, 0);
    MEMBER(InjectorDescription, std::vector<uint8_t>, genome, {});
    MEMBER(InjectorDescription, int, genomeGeneration, 0);
};

struct AutoBendingDescription
{
    auto operator<=>(AutoBendingDescription const&) const = default;

    // Fixed data
    MEMBER(AutoBendingDescription, float, maxAngleDeviation, 0.2f);    // Between 0 and 1
    MEMBER(AutoBendingDescription, float, frontBackVelRatio, 0.2f);  // Between 0 and 1

    // Process data
    MEMBER(AutoBendingDescription, float, initialAngle, 0.0f);
    MEMBER(AutoBendingDescription, float, lastActualAngle, 0.0f);
    MEMBER(AutoBendingDescription, bool, forward, true);    // Current direction
    MEMBER(AutoBendingDescription, float, activation, 0);
    MEMBER(AutoBendingDescription, int, activationCountdown, 0);
    MEMBER(AutoBendingDescription, bool, impulseAlreadyApplied, false);
};

struct ManualBendingDescription
{
    auto operator<=>(ManualBendingDescription const&) const = default;

    // Fixed data
    MEMBER(ManualBendingDescription, float, maxAngleDeviation, 0.2f);   // Between 0 and 1
    MEMBER(ManualBendingDescription, float, frontBackVelRatio, 0.2f);   // Between 0 and 1

    // Process data
    MEMBER(ManualBendingDescription, float, initialAngle, 0.0f);
    MEMBER(ManualBendingDescription, float, lastActualAngle, 0.0f);
    MEMBER(ManualBendingDescription, float, lastAngleDelta, 0.0f);
    MEMBER(ManualBendingDescription, bool, impulseAlreadyApplied, false);
};

struct AngleBendingDescription
{
    auto operator<=>(AngleBendingDescription const&) const = default;

    // Fixed data
    MEMBER(AngleBendingDescription, float, maxAngleDeviation, 0.2f);   // Between 0 and 1
    MEMBER(AngleBendingDescription, float, frontBackVelRatio, 0.2f);  // Between 0 and 1

    // Process data
    MEMBER(AngleBendingDescription, float, initialAngle, 0.0f);
};

struct AutoCrawlingDescription
{
    auto operator<=>(AutoCrawlingDescription const&) const = default;

    // Fixed data
    MEMBER(AutoCrawlingDescription, float, maxDistanceDeviation, 0.8f); // Between 0 and 1
    MEMBER(AutoCrawlingDescription, float, frontBackVelRatio, 0.2f);    // Between 0 and 1

    // Process data
    MEMBER(AutoCrawlingDescription, float, initialDistance, 0.0f);
    MEMBER(AutoCrawlingDescription, float, lastActualDistance, 0.0f);
    MEMBER(AutoCrawlingDescription, bool, forward, true);               // Current direction
    MEMBER(AutoCrawlingDescription, float, activation, 0.0f);
    MEMBER(AutoCrawlingDescription, int, activationCountdown, 0);
    MEMBER(AutoCrawlingDescription, bool, impulseAlreadyApplied, false);
};

struct ManualCrawlingDescription
{
    auto operator<=>(ManualCrawlingDescription const&) const = default;

    // Fixed data
    MEMBER(ManualCrawlingDescription, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(ManualCrawlingDescription, float, frontBackVelRatio, 0.2f);   // Between 0 and 1

    // Process data
    MEMBER(ManualCrawlingDescription, float, initialDistance, 0.0f);
    MEMBER(ManualCrawlingDescription, float, lastActualDistance, 0.0f);
    MEMBER(ManualCrawlingDescription, float, lastDistanceDelta, 0.0f);
    MEMBER(ManualCrawlingDescription, bool, impulseAlreadyApplied, false);
};

struct DirectMovementDescription
{
    auto operator<=>(DirectMovementDescription const&) const = default;
};

using MuscleModeDescription = std::variant<
    AutoBendingDescription,
    ManualBendingDescription,
    AngleBendingDescription,
    AutoCrawlingDescription,
    ManualCrawlingDescription,
    DirectMovementDescription>;

struct MuscleDescription
{
    auto operator<=>(MuscleDescription const&) const = default;

    MEMBER(MuscleDescription, MuscleModeDescription, mode, MuscleModeDescription());

    // Additional rendering data
    MEMBER(MuscleDescription, float, lastMovementX, 0.0f);
    MEMBER(MuscleDescription, float, lastMovementY, 0.0f);

    MuscleMode getMode() const
    {
        if (std::holds_alternative<AutoBendingDescription>(_mode)) {
            return MuscleMode_AutoBending;
        } else if (std::holds_alternative<ManualBendingDescription>(_mode)) {
            return MuscleMode_ManualBending;
        } else if (std::holds_alternative<AngleBendingDescription>(_mode)) {
            return MuscleMode_AngleBending;
        } else if (std::holds_alternative<AutoCrawlingDescription>(_mode)) {
            return MuscleMode_AutoCrawling;
        } else if (std::holds_alternative<ManualCrawlingDescription>(_mode)) {
            return MuscleMode_ManualCrawling;
        } else if (std::holds_alternative<DirectMovementDescription>(_mode)) {
            return MuscleMode_DirectMovement;
        }
        THROW_NOT_IMPLEMENTED();
    }
};

struct DefenderDescription
{
    auto operator<=>(DefenderDescription const&) const = default;

    MEMBER(DefenderDescription, DefenderMode, mode, DefenderMode_DefendAgainstAttacker);
};

struct ReconnectorDescription
{
    auto operator<=>(ReconnectorDescription const&) const = default;

    MEMBER(ReconnectorDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(ReconnectorDescription, ReconnectorRestrictToMutants, restrictToMutants, ReconnectorRestrictToMutants_NoRestriction);
};

struct DetonatorDescription
{
    auto operator<=>(DetonatorDescription const&) const = default;

    MEMBER(DetonatorDescription, DetonatorState, state, DetonatorState_Ready);
    MEMBER(DetonatorDescription, int, countdown, 10);
};

using CellTypeDescription = std::variant<
    StructureCellDescription,
    FreeCellDescription,
    BaseDescription,
    DepotDescription,
    ConstructorDescription,
    SensorDescription,
    OscillatorDescription,
    AttackerDescription,
    InjectorDescription,
    MuscleDescription,
    DefenderDescription,
    ReconnectorDescription,
    DetonatorDescription>;

struct SignalRoutingRestrictionDescription
{
    auto operator<=>(SignalRoutingRestrictionDescription const&) const = default;

    MEMBER(SignalRoutingRestrictionDescription, bool, active, false);
    MEMBER(SignalRoutingRestrictionDescription, float, baseAngle, 0);
    MEMBER(SignalRoutingRestrictionDescription, float, openingAngle, 0);
};

struct SignalDescription
{
    SignalDescription() { _channels.resize(MAX_CHANNELS, 0); }
    auto operator<=>(SignalDescription const&) const = default;

    MEMBER(SignalDescription, std::vector<float>, channels, {});
};

struct CellDescription
{
    CellDescription() = default;
    auto operator<=>(CellDescription const&) const = default;

    // General
    MEMBER(CellDescription, uint64_t, id, 0ull);
    MEMBER(CellDescription, std::optional<uint64_t>, genomeId, std::nullopt);
    MEMBER(CellDescription, std::vector<ConnectionDescription>, connections, {});
    MEMBER(CellDescription, RealVector2D, pos, RealVector2D());
    MEMBER(CellDescription, RealVector2D, vel, RealVector2D());
    MEMBER(CellDescription, float, energy, 100.0f);
    MEMBER(CellDescription, float, stiffness, 1.0f);
    MEMBER(CellDescription, int, color, 0);
    MEMBER(CellDescription, float, angleToFront, 0);  // Angle between [cell, cell->connection[0]] and front direction in reference configuration
    MEMBER(CellDescription, bool, barrier, false);
    MEMBER(CellDescription, bool, sticky, false);
    MEMBER(CellDescription, int, age, 0);
    MEMBER(CellDescription, LivingState, livingState, LivingState_Ready);
    MEMBER(CellDescription, int, creatureId, 0);
    MEMBER(CellDescription, int, mutationId, 0);
    MEMBER(CellDescription, uint8_t, ancestorMutationId, 0);
    MEMBER(CellDescription, float, genomeComplexity, 0);
    MEMBER(CellDescription, uint16_t, genomeNodeIndex, 0);

    // Cell type-specific data
    MEMBER(CellDescription, std::optional<NeuralNetworkDescription>, neuralNetwork, std::nullopt);
    CellTypeDescription _cellTypeData = BaseDescription();
    MEMBER(CellDescription, SignalRoutingRestrictionDescription, signalRoutingRestriction, SignalRoutingRestrictionDescription());
    MEMBER(CellDescription, uint8_t, signalRelaxationTime, 0);
    std::optional<SignalDescription> _signal;
    MEMBER(CellDescription, int, activationTime, 0);
    MEMBER(CellDescription, int, detectedByCreatureId, 0);  // Only the first 16 bits from the creature id
    MEMBER(CellDescription, CellTriggered, cellTypeUsed, CellTriggered_No);

    // Misc
    MEMBER(CellDescription, CellMetadataDescription, metadata, CellMetadataDescription());

    CellType getCellType() const;
    template <typename CellTypeDesc>
    CellDescription& cellType(CellTypeDesc const& value)
    {
        _cellTypeData = value;
        auto cellTypeEnum = getCellType();
        if (cellTypeEnum == CellType_Structure || cellTypeEnum == CellType_Free) {
            _neuralNetwork.reset();
        } else if (!_neuralNetwork.has_value()) {
            _neuralNetwork = NeuralNetworkDescription();
        }
        return *this;
    }
    CellDescription& signal(SignalDescription const& value)
    {
        _signal = value;
        _signalRelaxationTime = MAX_SIGNAL_RELAXATION_TIME;
        return *this;
    }
    CellDescription& signal(std::vector<float> const& value)
    {
        CHECK(value.size() == MAX_CHANNELS);

        SignalDescription newSignal;
        newSignal._channels = value;
        _signal = newSignal;
        _signalRelaxationTime = MAX_SIGNAL_RELAXATION_TIME;
        return *this;
    }
    CellDescription& signalRoutingRestriction(float baseAngle, float openingAngle)
    {
        SignalRoutingRestrictionDescription routingRestriction;
        routingRestriction._active = true;
        routingRestriction._baseAngle = baseAngle;
        routingRestriction._openingAngle = openingAngle;
        _signalRoutingRestriction = routingRestriction;
        return *this;
    }
    bool DEPRECATED_hasGenome() const;
    std::vector<uint8_t>& getGenomeRef();

    bool isConnectedTo(uint64_t id) const;
};

struct ClusterDescription
{
    ClusterDescription() = default;
    auto operator<=>(ClusterDescription const&) const = default;

    MEMBER(ClusterDescription, std::vector<CellDescription>, cells, {});

    ClusterDescription& addCells(std::vector<CellDescription> const& value)
    {
        _cells.insert(_cells.end(), value.begin(), value.end());
        return *this;
    }
    ClusterDescription& addCell(CellDescription const& value)
    {
        addCells({value});
        return *this;
    }

    RealVector2D getClusterPosFromCells() const;
};

struct ParticleDescription
{
    ParticleDescription() = default;
    auto operator<=>(ParticleDescription const&) const = default;

    MEMBER(ParticleDescription, uint64_t, id, 0ull);
    MEMBER(ParticleDescription, RealVector2D, pos, RealVector2D());
    MEMBER(ParticleDescription, RealVector2D, vel, RealVector2D());
    MEMBER(ParticleDescription, float, energy, 0.0f);
    MEMBER(ParticleDescription, int, color, 0);
};

struct ClusteredCollectionDescription
{
    ClusteredCollectionDescription() = default;
    auto operator<=>(ClusteredCollectionDescription const&) const = default;

    MEMBER(ClusteredCollectionDescription, std::vector<ClusterDescription>, clusters, {});
    MEMBER(ClusteredCollectionDescription, std::vector<ParticleDescription>, particles, {});
    MEMBER(ClusteredCollectionDescription, std::vector<GenomeDescription_New>, genomes, {});

    ClusteredCollectionDescription& addClusters(std::vector<ClusterDescription> const& value)
    {
        _clusters.insert(_clusters.end(), value.begin(), value.end());
        return *this;
    }
    ClusteredCollectionDescription& addCluster(ClusterDescription const& value)
    {
        addClusters({value});
        return *this;
    }

    ClusteredCollectionDescription& addParticles(std::vector<ParticleDescription> const& value)
    {
        _particles.insert(_particles.end(), value.begin(), value.end());
        return *this;
    }
    ClusteredCollectionDescription& addParticle(ParticleDescription const& value)
    {
        addParticles({value});
        return *this;
    }
    void clear()
    {
        _clusters.clear();
        _particles.clear();
    }
    bool isEmpty() const
    {
        if (!_clusters.empty()) {
            return false;
        }
        if (!_particles.empty()) {
            return false;
        }
        return true;
    }
    void setCenter(RealVector2D const& center);

    RealVector2D calcCenter() const;
    void shift(RealVector2D const& delta);
    int getNumberOfCellAndParticles() const;
};

struct CollectionDescription
{
    CollectionDescription() = default;
    explicit CollectionDescription(ClusteredCollectionDescription const& clusteredData);
    auto operator<=>(CollectionDescription const&) const = default;

    MEMBER(CollectionDescription, std::vector<CellDescription>, cells, {});
    MEMBER(CollectionDescription, std::vector<ParticleDescription>, particles, {});
    MEMBER(CollectionDescription, std::vector<GenomeDescription_New>, genomes, {});

    CollectionDescription& add(CollectionDescription const& other);
    CollectionDescription& addCells(std::vector<CellDescription> const& value);
    CollectionDescription& addCell(CellDescription const& value);

    CollectionDescription& addParticles(std::vector<ParticleDescription> const& value);
    CollectionDescription& addParticle(ParticleDescription const& value);

    CollectionDescription& addCreature(GenomeDescription_New const& genome, std::vector<CellDescription> const& cells);

    void clear();
    bool isEmpty() const;
    void setCenter(RealVector2D const& center);

    RealVector2D calcCenter() const;
    void shift(RealVector2D const& delta);
    void rotate(float angle);
    void accelerate(RealVector2D const& velDelta, float angularVelDelta);

    std::unordered_set<uint64_t> getCellIds() const;

    CollectionDescription& addConnection(uint64_t const& cellId1, uint64_t const& cellId2, std::unordered_map<uint64_t, int>* cache = nullptr);
    CollectionDescription&
    addConnection(uint64_t const& cellId1, uint64_t const& cellId2, RealVector2D const& refPosCell2, std::unordered_map<uint64_t, int>* cache = nullptr);

private:
    CellDescription& getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>* cache = nullptr);
};

using CellOrParticleDescription = std::variant<CellDescription, ParticleDescription>;
