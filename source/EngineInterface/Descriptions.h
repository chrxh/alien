#pragma once

#include <mdspan>
#include <variant>

#include "Base/Definitions.h"
#include "EngineInterface/EngineConstants.h"

#include "Definitions.h"

struct CellMetadataDescription
{
    auto operator<=>(CellMetadataDescription const&) const = default;

    CellMetadataDescription& name(std::string const& value)
    {
        _name = value;
        return *this;
    }
    CellMetadataDescription& description(std::string const& value)
    {
        _description = value;
        return *this;
    }

    std::string _name;
    std::string _description;
};

struct ConnectionDescription
{
    auto operator<=>(ConnectionDescription const&) const = default;

    ConnectionDescription& cellId(uint64_t const& value)
    {
        _cellId = value;
        return *this;
    }
    ConnectionDescription& distance(float const& value)
    {
        _distance = value;
        return *this;
    }
    ConnectionDescription& angleFromPrevious(float const& value)
    {
        _angleFromPrevious = value;
        return *this;
    }

    uint64_t _cellId = 0;  //value of 0 means cell not present in DataDescription
    float _distance = 0;
    float _angleFromPrevious = 0;
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
        auto md = std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            md[i, i] = 1.0f;
        }
    }
    auto operator<=>(NeuralNetworkDescription const&) const = default;

    NeuralNetworkDescription& weight(int row, int col, float value)
    {
        auto md = std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        md[row, col] = value;
        return *this;
    }
    auto getWeights() const { return std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS); }
    auto getWeights() { return std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS); }

    std::vector<float> _weights;
    std::vector<float> _biases;
    std::vector<ActivationFunction> _activationFunctions;
};

struct BaseDescription
{
    auto operator<=>(BaseDescription const&) const = default;
};

struct DepotDescription
{
    auto operator<=>(DepotDescription const&) const = default;

    DepotDescription& mode(EnergyDistributionMode value)
    {
        _mode = value;
        return *this;
    }

    EnergyDistributionMode _mode = EnergyDistributionMode_TransmittersAndConstructors;
};

struct ConstructorDescription
{
    ConstructorDescription();
    auto operator<=>(ConstructorDescription const&) const = default;

    ConstructorDescription& autoTriggerInterval(int value)
    {
        _autoTriggerInterval = toUInt8(value);
        return *this;
    }
    ConstructorDescription& constructionActivationTime(int value)
    {
        _constructionActivationTime = value;
        return *this;
    }
    ConstructorDescription& genome(std::vector<uint8_t> const& value)
    {
        _genome = value;
        return *this;
    }
    ConstructorDescription& genomeCurrentNodeIndex(int value)
    {
        _genomeCurrentNodeIndex = value;
        return *this;
    }
    ConstructorDescription& genomeCurrentRepetition(int value)
    {
        _genomeCurrentRepetition = value;
        return *this;
    }
    ConstructorDescription& genomeCurrentBranch(int value)
    {
        _genomeCurrentBranch = value;
        return *this;
    }
    int getNumInheritedGenomeNodes() const { return _numInheritedGenomeNodes; }
    bool isGenomeInherited() const { return _numInheritedGenomeNodes != 0; }
    ConstructorDescription& numInheritedGenomeNodes(int value)
    {
        _numInheritedGenomeNodes = value;
        return *this;
    }
    ConstructorDescription& genomeGeneration(int value)
    {
        _genomeGeneration = value;
        return *this;
    }
    ConstructorDescription& constructionAngle1(float value)
    {
        _constructionAngle1 = value;
        return *this;
    }
    ConstructorDescription& constructionAngle2(float value)
    {
        _constructionAngle2 = value;
        return *this;
    }
    ConstructorDescription& lastConstructedCellId(uint64_t value)
    {
        _lastConstructedCellId = value;
        return *this;
    }

    // Properties
    int _autoTriggerInterval = 100;  // 0 = manual (triggered by signal), > 0 = auto trigger
    int _constructionActivationTime = 100;

    // Genome data
    std::vector<uint8_t> _genome;
    int _numInheritedGenomeNodes = 0;
    int _genomeGeneration = 0;
    float _constructionAngle1 = 0;
    float _constructionAngle2 = 0;

    // Process data
    uint64_t _lastConstructedCellId = 0;
    int _genomeCurrentNodeIndex = 0;
    int _genomeCurrentRepetition = 0;
    int _genomeCurrentBranch = 0;
    int _offspringCreatureId = 0;
    int _offspringMutationId = 0;
};

struct SensorDescription
{
    auto operator<=>(SensorDescription const&) const = default;

    SensorDescription& autoTriggerInterval(int value)
    {
        _autoTriggerInterval = value;
        return *this;
    }
    SensorDescription& color(int value)
    {
        _restrictToColor = value;
        return *this;
    }
    SensorDescription& minDensity(float value)
    {
        _minDensity = value;
        return *this;
    }
    SensorDescription& minRange(int value)
    {
        _minRange = value;
        return *this;
    }
    SensorDescription& maxRange(int value)
    {
        _maxRange = value;
        return *this;
    }
    SensorDescription& restrictToMutants(SensorRestrictToMutants value)
    {
        _restrictToMutants = value;
        return *this;
    }

    int _autoTriggerInterval = 100;  // 0 = manual (triggered by signal), > 0 = auto trigger
    float _minDensity = 0.05f;
    std::optional<int> _minRange;
    std::optional<int> _maxRange;
    std::optional<int> _restrictToColor;
    SensorRestrictToMutants _restrictToMutants = SensorRestrictToMutants_NoRestriction;
};

struct OscillatorDescription
{
    auto operator<=>(OscillatorDescription const&) const = default;

    OscillatorDescription& autoTriggerInterval(int value)
    {
        _autoTriggerInterval = value;
        return *this;
    }
    OscillatorDescription& alternationInterval(int value)
    {
        _alternationInterval = value;
        return *this;
    }

    // Fixed data
    int _autoTriggerInterval = 0;
    int _alternationInterval = 0;  // 0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.

    // Process data
    int _numPulses = 0;
};

struct AttackerDescription
{
    auto operator<=>(AttackerDescription const&) const = default;

    AttackerDescription& mode(EnergyDistributionMode value)
    {
        _mode = value;
        return *this;
    }

    EnergyDistributionMode _mode = EnergyDistributionMode_TransmittersAndConstructors;
};

struct InjectorDescription
{
    InjectorDescription();
    auto operator<=>(InjectorDescription const&) const = default;

    InjectorDescription& mode(InjectorMode value)
    {
        _mode = value;
        return *this;
    }
    InjectorDescription& genome(std::vector<uint8_t> const& value)
    {
        _genome = value;
        return *this;
    }
    InjectorDescription& genomeGeneration(int value)
    {
        _genomeGeneration = value;
        return *this;
    }

    InjectorMode _mode = InjectorMode_InjectAll;
    int _counter = 0;
    std::vector<uint8_t> _genome;
    int _genomeGeneration = 0;
};

struct AutoBendingDescription
{
    auto operator<=>(AutoBendingDescription const&) const = default;

    AutoBendingDescription& maxAngleDeviation(float value)
    {
        _maxAngleDeviation = value;
        return *this;
    }

    // Fixed data
    float _maxAngleDeviation = 20.0f; // Between 0 and 1
    float _frontBackVelRatio = 0.2f;  // Between 0 and 1

    // Process data
    BendingMode _bendingMode = BendingMode_BackAndForth;
    float _initialAngle = 0;
    float _lastAngle = 0;
    bool _forward = true;  // Current direction
    float _activation = 0;
    int _activationCountdown = 0;
    bool _impulseAlreadyApplied = false;
};

struct AutoCrawlingDescription
{
    auto operator<=>(AutoCrawlingDescription const&) const = default;

    AutoCrawlingDescription& maxDistanceDeviation(float value)
    {
        _maxDistanceDeviation = value;
        return *this;
    }

    // Fixed data
    float _maxDistanceDeviation = 20.0f; // Between 0 and 1
    float _frontBackVelRatio = 0.2f;  // Between 0 and 1

    // Process data
    CrawlingMode _bendingMode = CrawlingMode_BackAndForth;
    float _initialAngle = 0;
    float _lastAngle = 0;
    bool _forward = true;  // Current direction
    float _activation = 0;
    int _activationCountdown = 0;
    bool _impulseAlreadyApplied = false;
};
using MuscleModeDescription = std::variant<AutoBendingDescription, AutoCrawlingDescription>;

struct MuscleDescription
{
    auto operator<=>(MuscleDescription const&) const = default;

    MuscleMode getMode() const { return MuscleMode_Bending; }
    MuscleDescription& mode(MuscleModeDescription const& value)
    {
        _mode = value;
        return *this;
    }
    MuscleDescription& frontAngle(float value)
    {
        _frontAngle = value;
        return *this;
    }

    MuscleModeDescription _mode;

    // Temp
    float _frontAngle = 0;  // Can be removed in new genome model

    // Additional rendering data
    float _lastMovementX = 0;
    float _lastMovementY = 0;
};

struct DefenderDescription
{
    auto operator<=>(DefenderDescription const&) const = default;

    DefenderDescription& mode(DefenderMode value)
    {
        _mode = value;
        return *this;
    }

    DefenderMode _mode = DefenderMode_DefendAgainstAttacker;
};

struct ReconnectorDescription
{
    auto operator<=>(ReconnectorDescription const&) const = default;

    ReconnectorDescription& restrictToColor(int value)
    {
        _restrictToColor = value;
        return *this;
    }
    ReconnectorDescription& restrictToMutants(ReconnectorRestrictToMutants value)
    {
        _restrictToMutants = value;
        return *this;
    }

    std::optional<int> _restrictToColor;
    ReconnectorRestrictToMutants _restrictToMutants = ReconnectorRestrictToMutants_NoRestriction;
};

struct DetonatorDescription
{
    auto operator<=>(DetonatorDescription const&) const = default;

    DetonatorDescription& state(DetonatorState value)
    {
        _state = value;
        return *this;
    }

    DetonatorDescription& countDown(int value)
    {
        _countdown = value;
        return *this;
    }

    DetonatorState _state = DetonatorState_Ready;
    int _countdown = 10;
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

    bool _active = false;
    float _baseAngle = 0;
    float _openingAngle = 0;
};

struct SignalDescription
{
    SignalDescription()
    {
        _channels.resize(MAX_CHANNELS, 0);
    }
    auto operator<=>(SignalDescription const&) const = default;

    SignalDescription& channels(std::vector<float> const& value)
    {
        CHECK(value.size() == MAX_CHANNELS);
        _channels = value;
        return *this;
    }

    std::vector<float> _channels;
    SignalOrigin _origin = SignalOrigin_Unknown;
    float _targetX = 0;
    float _targetY = 0;
};

struct CellDescription
{
    CellDescription() = default;
    auto operator<=>(CellDescription const&) const = default;

    CellDescription& id(uint64_t value)
    {
        _id = value;
        return *this;
    }
    CellDescription& pos(RealVector2D const& value)
    {
        _pos = value;
        return *this;
    }
    CellDescription& vel(RealVector2D const& value)
    {
        _vel = value;
        return *this;
    }
    CellDescription& energy(float value)
    {
        _energy = value;
        return *this;
    }
    CellDescription& stiffness(float value)
    {
        _stiffness = value;
        return *this;
    }
    CellDescription& color(int value)
    {
        _color = value;
        return *this;
    }
    CellDescription& absAngleToConnection0(float value)
    {
        _absAngleToConnection0 = value;
        return *this;
    }
    CellDescription& barrier(bool value)
    {
        _barrier = value;
        return *this;
    }
    CellDescription& age(int value)
    {
        _age = value;
        return *this;
    }

    CellDescription& connectingCells(std::vector<ConnectionDescription> const& value)
    {
        _connections = value;
        return *this;
    }
    CellDescription& livingState(LivingState value)
    {
        _livingState = value;
        return *this;
    }
    CellDescription& constructionId(int value)
    {
        _creatureId = value;
        return *this;
    }
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
    CellDescription& neuralNetwork(NeuralNetworkDescription const& value)
    {
        _neuralNetwork = value;
        return *this;
    }
    CellDescription& metadata(CellMetadataDescription const& value)
    {
        _metadata = value;
        return *this;
    }
    CellDescription& signalRelaxationTime(uint8_t value)
    {
        _signalRelaxationTime = value;
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
    CellDescription& activationTime(int value)
    {
        _activationTime = value;
        return *this;
    }
    CellDescription& creatureId(int value)
    {
        _creatureId = value;
        return *this;
    }
    CellDescription& mutationId(int value)
    {
        _mutationId = value;
        return *this;
    }
    CellDescription& genomeComplexity(float value)
    {
        _genomeComplexity = value;
        return *this;
    }

    bool hasGenome() const;
    std::vector<uint8_t>& getGenomeRef();

    bool isConnectedTo(uint64_t id) const;

    // General
    uint64_t _id = 0;
    std::vector<ConnectionDescription> _connections;
    RealVector2D _pos;
    RealVector2D _vel;
    float _energy = 100.0f;
    float _stiffness = 1.0f;
    int _color = 0;
    float _absAngleToConnection0 = 0;
    bool _barrier = false;
    int _age = 0;
    LivingState _livingState = LivingState_Ready;
    int _creatureId = 0;
    int _mutationId = 0;
    uint8_t _ancestorMutationId = 0;
    float _genomeComplexity = 0;
    uint16_t _genomeNodeIndex = 0;

    // Cell type data
    std::optional<NeuralNetworkDescription> _neuralNetwork = NeuralNetworkDescription();
    CellTypeDescription _cellTypeData = BaseDescription();
    SignalRoutingRestrictionDescription _signalRoutingRestriction;
    uint8_t _signalRelaxationTime = 0;
    std::optional<SignalDescription> _signal;
    int _activationTime = 0;
    int _detectedByCreatureId = 0;  //only the first 16 bits from the creature id
    CellTriggered _cellTypeUsed = CellTriggered_No;

    // Misc
    CellMetadataDescription _metadata;
};

struct ClusterDescription
{
    ClusterDescription() = default;
    auto operator<=>(ClusterDescription const&) const = default;

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

    std::vector<CellDescription> _cells;
};

struct ParticleDescription
{
    ParticleDescription() = default;
    auto operator<=>(ParticleDescription const&) const = default;

    ParticleDescription& id(uint64_t value)
    {
        _id = value;
        return *this;
    }
    ParticleDescription& pos(RealVector2D const& value)
    {
        _pos = value;
        return *this;
    }
    ParticleDescription& vel(RealVector2D const& value)
    {
        _vel = value;
        return *this;
    }
    ParticleDescription& energy(float value)
    {
        _energy = value;
        return *this;
    }
    ParticleDescription& color(int value)
    {
        _color = value;
        return *this;
    }

    uint64_t _id = 0;

    RealVector2D _pos;
    RealVector2D _vel;
    float _energy = 0;
    int _color = 0;
};

struct ClusteredDataDescription
{
    ClusteredDataDescription() = default;
    auto operator<=>(ClusteredDataDescription const&) const = default;

    ClusteredDataDescription& addClusters(std::vector<ClusterDescription> const& value)
    {
        _clusters.insert(_clusters.end(), value.begin(), value.end());
        return *this;
    }
    ClusteredDataDescription& addCluster(ClusterDescription const& value)
    {
        addClusters({value});
        return *this;
    }

    ClusteredDataDescription& addParticles(std::vector<ParticleDescription> const& value)
    {
        _particles.insert(_particles.end(), value.begin(), value.end());
        return *this;
    }
    ClusteredDataDescription& addParticle(ParticleDescription const& value)
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

    std::vector<ClusterDescription> _clusters;
    std::vector<ParticleDescription> _particles;
};

struct DataDescription
{
    DataDescription() = default;
    explicit DataDescription(ClusteredDataDescription const& clusteredData);
    auto operator<=>(DataDescription const&) const = default;

    DataDescription& add(DataDescription const& other);
    DataDescription& addCells(std::vector<CellDescription> const& value);
    DataDescription& addCell(CellDescription const& value);

    DataDescription& addParticles(std::vector<ParticleDescription> const& value);
    DataDescription& addParticle(ParticleDescription const& value);
    void clear();
    bool isEmpty() const;
    void setCenter(RealVector2D const& center);

    RealVector2D calcCenter() const;
    void shift(RealVector2D const& delta);
    void rotate(float angle);
    void accelerate(RealVector2D const& velDelta, float angularVelDelta);

    std::unordered_set<uint64_t> getCellIds() const;

    DataDescription& addConnection(uint64_t const& cellId1, uint64_t const& cellId2, std::unordered_map<uint64_t, int>* cache = nullptr);
    DataDescription& addConnection(uint64_t const& cellId1, uint64_t const& cellId2, RealVector2D const& refPosCell2, std::unordered_map<uint64_t, int>* cache = nullptr);

    std::vector<CellDescription> _cells;
    std::vector<ParticleDescription> _particles;

private:
    CellDescription& getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>* cache = nullptr);
};

using CellOrParticleDescription = std::variant<CellDescription, ParticleDescription>;
