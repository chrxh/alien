#pragma once

#include <mdspan>
#include <variant>

#include "Base/Definitions.h"
#include "EngineInterface/EngineConstants.h"

#include "Definitions.h"

struct CellMetadataDescription
{
    std::string name;
    std::string description;

    auto operator<=>(CellMetadataDescription const&) const = default;

    CellMetadataDescription& setName(std::string const& value)
    {
        name = value;
        return *this;
    }
    CellMetadataDescription& setDescription(std::string const& value)
    {
        description = value;
        return *this;
    }
};

struct ConnectionDescription
{
    uint64_t cellId = 0;    //value of 0 means cell not present in DataDescription
    float distance = 0;
    float angleFromPrevious = 0;

    auto operator<=>(ConnectionDescription const&) const = default;

    ConnectionDescription& setCellId(uint64_t const& value)
    {
        cellId = value;
        return *this;
    }
    ConnectionDescription& setDistance(float const& value)
    {
        distance = value;
        return *this;
    }
    ConnectionDescription& setAngleFromPrevious(float const& value)
    {
        angleFromPrevious = value;
        return *this;
    }
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
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<ActivationFunction> activationFunctions;
    NeuralNetworkDescription()
    {
        weights.resize(MAX_CHANNELS * MAX_CHANNELS, 0);
        biases.resize(MAX_CHANNELS, 0);
        activationFunctions.resize(MAX_CHANNELS, 0);
    }
    auto operator<=>(NeuralNetworkDescription const&) const = default;
    NeuralNetworkDescription& setWeight(int row, int col, float value)
    {
        auto md = std::mdspan(weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        md[row, col] = value;
        return *this;
    }
    auto getWeights() const { return std::mdspan(weights.data(), MAX_CHANNELS, MAX_CHANNELS); }
    auto getWeights() { return std::mdspan(weights.data(), MAX_CHANNELS, MAX_CHANNELS); }
};

struct BaseDescription
{
    auto operator<=>(BaseDescription const&) const = default;
};

struct DepotDescription
{
    EnergyDistributionMode mode = EnergyDistributionMode_TransmittersAndConstructors;

    auto operator<=>(DepotDescription const&) const = default;

    DepotDescription& setMode(EnergyDistributionMode value)
    {
        mode = value;
        return *this;
    }
};

struct ConstructorDescription
{
    int activationMode = 100;   //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third timestep, etc.
    int constructionActivationTime = 100;
    std::vector<uint8_t> genome;
    int numInheritedGenomeNodes = 0;
    int genomeGeneration = 0;
    float constructionAngle1 = 0;
    float constructionAngle2 = 0;

    //process data
    uint64_t lastConstructedCellId = 0;
    int genomeCurrentNodeIndex = 0;
    int genomeCurrentRepetition = 0;
    int currentBranch = 0;
    int offspringCreatureId = 0;
    int offspringMutationId = 0;

    ConstructorDescription();
    auto operator<=>(ConstructorDescription const&) const = default;

    ConstructorDescription& setActivationMode(int value)
    {
        activationMode = value;
        return *this;
    }
    ConstructorDescription& setConstructionActivationTime(int value)
    {
        constructionActivationTime = value;
        return *this;
    }
    ConstructorDescription& setGenome(std::vector<uint8_t> const& value)
    {
        genome = value;
        return *this;
    }
    ConstructorDescription& setGenomeCurrentNodeIndex(int value)
    {
        genomeCurrentNodeIndex = value;
        return *this;
    }
    ConstructorDescription& setGenomeCurrentRepetition(int value)
    {
        genomeCurrentRepetition = value;
        return *this;
    }
    ConstructorDescription& setCurrentBranch(int value)
    {
        currentBranch = value;
        return *this;
    }
    int getNumInheritedGenomeNodes() const { return numInheritedGenomeNodes; }
    bool isGenomeInherited() const { return numInheritedGenomeNodes != 0; }
    ConstructorDescription& setNumInheritedGenomeNodes(int value)
    {
        numInheritedGenomeNodes = value;
        return *this;
    }
    ConstructorDescription& setGenomeGeneration(int value)
    {
        genomeGeneration = value;
        return *this;
    }
    ConstructorDescription& setConstructionAngle1(float value)
    {
        constructionAngle1 = value;
        return *this;
    }
    ConstructorDescription& setConstructionAngle2(float value)
    {
        constructionAngle2 = value;
        return *this;
    }
    ConstructorDescription& setLastConstructedCellId(uint64_t value)
    {
        lastConstructedCellId = value;
        return *this;
    }
};

struct SensorDescription
{
    float minDensity = 0.05f;
    std::optional<int> minRange;
    std::optional<int> maxRange;
    std::optional<int> restrictToColor;
    SensorRestrictToMutants restrictToMutants = SensorRestrictToMutants_NoRestriction;

    //process data
    float memoryChannel1 = 0;
    float memoryChannel2 = 0;
    float memoryChannel3 = 0;
    float memoryTargetX = 0;
    float memoryTargetY = 0;

    auto operator<=>(SensorDescription const&) const = default;

    SensorDescription& setColor(int value)
    {
        restrictToColor = value;
        return *this;
    }
    SensorDescription& setMinDensity(float value)
    {
        minDensity = value;
        return *this;
    }
    SensorDescription& setMinRange(int value)
    {
        minRange = value;
        return *this;
    }
    SensorDescription& setMaxRange(int value)
    {
        maxRange = value;
        return *this;
    }
    SensorDescription& setRestrictToMutants(SensorRestrictToMutants value)
    {
        restrictToMutants = value;
        return *this;
    }
};

struct OscillatorDescription
{
    int pulseMode = 0;          //0 = none, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    int alternationMode = 0;    //0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.

    auto operator<=>(OscillatorDescription const&) const = default;

    OscillatorDescription& setPulseMode(int value)
    {
        pulseMode = value;
        return *this;
    }
    OscillatorDescription& setAlternationMode(int value)
    {
        alternationMode = value;
        return *this;
    }
};

struct AttackerDescription
{
    EnergyDistributionMode mode = EnergyDistributionMode_TransmittersAndConstructors;

    auto operator<=>(AttackerDescription const&) const = default;

    AttackerDescription& setMode(EnergyDistributionMode value)
    {
        mode = value;
        return *this;
    }
};

struct InjectorDescription
{
    InjectorMode mode = InjectorMode_InjectAll;
    int counter = 0;
    std::vector<uint8_t> genome;
    int genomeGeneration = 0;

    InjectorDescription();
    auto operator<=>(InjectorDescription const&) const = default;
    InjectorDescription& setMode(InjectorMode value)
    {
        mode = value;
        return *this;
    }
    InjectorDescription& setGenome(std::vector<uint8_t> const& value)
    {
        genome = value;
        return *this;
    }
    InjectorDescription& setGenomeGeneration(int value)
    {
        genomeGeneration = value;
        return *this;
    }
};

struct MuscleDescription
{
    MuscleMode mode = MuscleMode_Movement;
    MuscleBendingDirection lastBendingDirection = MuscleBendingDirection_None;
    int lastBendingSourceIndex = 0;
    float consecutiveBendingAngle = 0;

    //additional rendering data
    float lastMovementX = 0;
    float lastMovementY = 0;

    auto operator<=>(MuscleDescription const&) const = default;

    MuscleDescription& setMode(MuscleMode value)
    {
        mode = value;
        return *this;
    }
};

struct DefenderDescription
{
    DefenderMode mode = DefenderMode_DefendAgainstAttacker;

    auto operator<=>(DefenderDescription const&) const = default;

    DefenderDescription& setMode(DefenderMode value)
    {
        mode = value;
        return *this;
    }
};

struct ReconnectorDescription
{
    std::optional<int> restrictToColor;
    ReconnectorRestrictToMutants restrictToMutants = ReconnectorRestrictToMutants_NoRestriction;

    auto operator<=>(ReconnectorDescription const&) const = default;

    ReconnectorDescription& setRestrictToColor(int value)
    {
        restrictToColor = value;
        return *this;
    }
    ReconnectorDescription& setRestrictToMutants(ReconnectorRestrictToMutants value)
    {
        restrictToMutants = value;
        return *this;
    }
};

struct DetonatorDescription
{
    DetonatorState state = DetonatorState_Ready;
    int countdown = 10;

    auto operator<=>(DetonatorDescription const&) const = default;

    DetonatorDescription& setState(DetonatorState value)
    {
        state = value;
        return *this;
    }

    DetonatorDescription& setCountDown(int value)
    {
        countdown = value;
        return *this;
    }
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
    bool active = false;
    float baseAngle = 0;
    float openingAngle = 0;

    auto operator<=>(SignalRoutingRestrictionDescription const&) const = default;
};

struct SignalDescription
{
    std::vector<float> channels;
    SignalOrigin origin = SignalOrigin_Unknown;
    float targetX = 0;
    float targetY = 0;
    std::vector<uint64_t> prevCellIds;

    SignalDescription()
    {
        channels.resize(MAX_CHANNELS, 0);
        prevCellIds.resize(MAX_CELL_BONDS, 0);
    }
    auto operator<=>(SignalDescription const&) const = default;

    SignalDescription& setChannels(std::vector<float> const& value)
    {
        CHECK(value.size() == MAX_CHANNELS);
        channels = value;
        return *this;
    }
    SignalDescription& setPrevCellIds(std::vector<uint64_t> const& value)
    {
        prevCellIds = value;
        return *this;
    }
};

struct CellDescription
{
    uint64_t id = 0;

    //general
    std::vector<ConnectionDescription> connections;
    RealVector2D pos;
    RealVector2D vel;
    float energy = 100.0f;
    float stiffness = 1.0f;
    int color = 0;
    bool barrier = false;
    int age = 0;
    LivingState livingState = LivingState_Ready;
    int creatureId = 0;
    int mutationId = 0;
    uint8_t ancestorMutationId = 0;
    float genomeComplexity = 0;
    uint16_t genomeNodeIndex = 0;

    //cell type data
    std::optional<NeuralNetworkDescription> neuralNetwork = NeuralNetworkDescription();
    CellTypeDescription cellTypeData = BaseDescription();
    SignalRoutingRestrictionDescription signalRoutingRestriction;
    std::optional<SignalDescription> signal;
    int activationTime = 0;
    int detectedByCreatureId = 0;   //only the first 16 bits from the creature id
    CellTriggered cellTypeUsed = CellTriggered_No;

    CellMetadataDescription metadata;

    CellDescription() = default;
    auto operator<=>(CellDescription const&) const = default;

    CellDescription& setId(uint64_t value)
    {
        id = value;
        return *this;
    }
    CellDescription& setPos(RealVector2D const& value)
    {
        pos = value;
        return *this;
    }
    CellDescription& setVel(RealVector2D const& value)
    {
        vel = value;
        return *this;
    }
    CellDescription& setEnergy(float value)
    {
        energy = value;
        return *this;
    }
    CellDescription& setStiffness(float value)
    {
        stiffness = value;
        return *this;
    }
    CellDescription& setColor(int value)
    {
        color = value;
        return *this;
    }
    CellDescription& setBarrier(bool value)
    {
        barrier = value;
        return *this;
    }
    CellDescription& setAge(int value)
    {
        age = value;
        return *this;
    }

    CellDescription& setConnectingCells(std::vector<ConnectionDescription> const& value)
    {
        connections = value;
        return *this;
    }
    CellDescription& setLivingState(LivingState value)
    {
        livingState = value;
        return *this;
    }
    CellDescription& setConstructionId(int value)
    {
        creatureId = value;
        return *this;
    }
    CellType getCellType() const;
    template <typename CellTypeDesc>
    CellDescription& setCellTypeData(CellTypeDesc const& value)
    {
        cellTypeData = value;
        auto cellType = getCellType();
        if (cellType == CellType_Structure || cellType == CellType_Free) {
            neuralNetwork.reset();
        } else if (!neuralNetwork.has_value()) {
            neuralNetwork = NeuralNetworkDescription();
        }
        return *this;
    }
    CellDescription& setNeuralNetwork(NeuralNetworkDescription const& value)
    {
        neuralNetwork = value;
        return *this;
    }
    CellDescription& setMetadata(CellMetadataDescription const& value)
    {
        metadata = value;
        return *this;
    }
    CellDescription& setSignal(SignalDescription const& value)
    {
        signal = value;
        return *this;
    }
    CellDescription& setSignal(std::vector<float> const& value)
    {
        CHECK(value.size() == MAX_CHANNELS);

        SignalDescription newSignal;
        newSignal.channels = value;
        signal = newSignal;
        return *this;
    }
    CellDescription& setSignalRoutingRestriction(float baseAngle, float openingAngle)
    {
        SignalRoutingRestrictionDescription routingRestriction;
        routingRestriction.active = true;
        routingRestriction.baseAngle = baseAngle;
        routingRestriction.openingAngle = openingAngle;
        signalRoutingRestriction = routingRestriction;
        return *this;
    }
    CellDescription& setActivationTime(int value)
    {
        activationTime = value;
        return *this;
    }
    CellDescription& setCreatureId(int value)
    {
        creatureId = value;
        return *this;
    }
    CellDescription& setMutationId(int value)
    {
        mutationId = value;
        return *this;
    }
    CellDescription& setGenomeComplexity(float value)
    {
        genomeComplexity = value;
        return *this;
    }

    bool hasGenome() const;
    std::vector<uint8_t>& getGenomeRef();

    bool isConnectedTo(uint64_t id) const;
};

struct ClusterDescription
{
    std::vector<CellDescription> cells;

    ClusterDescription() = default;
    auto operator<=>(ClusterDescription const&) const = default;

    ClusterDescription& addCells(std::vector<CellDescription> const& value)
    {
        cells.insert(cells.end(), value.begin(), value.end());
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
    uint64_t id = 0;

    RealVector2D pos;
    RealVector2D vel;
    float energy = 0;
    int color = 0;

    ParticleDescription() = default;
    auto operator<=>(ParticleDescription const&) const = default;
    ParticleDescription& setId(uint64_t value)
    {
        id = value;
        return *this;
    }
    ParticleDescription& setPos(RealVector2D const& value)
    {
        pos = value;
        return *this;
    }
    ParticleDescription& setVel(RealVector2D const& value)
    {
        vel = value;
        return *this;
    }
    ParticleDescription& setEnergy(float value)
    {
        energy = value;
        return *this;
    }
    ParticleDescription& setColor(int value)
    {
        color = value;
        return *this;
    }
};

struct ClusteredDataDescription
{
    std::vector<ClusterDescription> clusters;
    std::vector<ParticleDescription> particles;

    ClusteredDataDescription() = default;
    auto operator<=>(ClusteredDataDescription const&) const = default;

    ClusteredDataDescription& addClusters(std::vector<ClusterDescription> const& value)
    {
        clusters.insert(clusters.end(), value.begin(), value.end());
        return *this;
    }
    ClusteredDataDescription& addCluster(ClusterDescription const& value)
    {
        addClusters({value});
        return *this;
    }

    ClusteredDataDescription& addParticles(std::vector<ParticleDescription> const& value)
    {
        particles.insert(particles.end(), value.begin(), value.end());
        return *this;
    }
    ClusteredDataDescription& addParticle(ParticleDescription const& value)
    {
        addParticles({value});
        return *this;
    }
    void clear()
    {
        clusters.clear();
        particles.clear();
    }
    bool isEmpty() const
    {
        if (!clusters.empty()) {
            return false;
        }
        if (!particles.empty()) {
            return false;
        }
        return true;
    }
    void setCenter(RealVector2D const& center);

    RealVector2D calcCenter() const;
    void shift(RealVector2D const& delta);
    int getNumberOfCellAndParticles() const;
};

struct DataDescription
{
    std::vector<CellDescription> cells;
    std::vector<ParticleDescription> particles;

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

private:
    CellDescription& getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>* cache = nullptr);
};

using CellOrParticleDescription = std::variant<CellDescription, ParticleDescription>;
