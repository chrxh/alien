#pragma once

#include <variant>

#include "Base/Definitions.h"
#include "EngineInterface/FundamentalConstants.h"

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

struct ActivityDescription
{
    std::vector<float> channels;

    ActivityDescription() { channels.resize(MAX_CHANNELS, 0); }
    auto operator<=>(ActivityDescription const&) const = default;

    ActivityDescription& setChannels(std::vector<float> const& value)
    {
        CHECK(value.size() == MAX_CHANNELS);
        channels = value;
        return *this;
    }
};


struct NeuronDescription
{
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    NeuronDescription()
    {
        weights.resize(MAX_CHANNELS, std::vector<float>(MAX_CHANNELS, 0));
        biases.resize(MAX_CHANNELS, 0);
    }
    auto operator<=>(NeuronDescription const&) const = default;
};

struct TransmitterDescription
{
    EnergyDistributionMode mode = EnergyDistributionMode_TransmittersAndConstructors;

    auto operator<=>(TransmitterDescription const&) const = default;

    TransmitterDescription& setMode(EnergyDistributionMode value)
    {
        mode = value;
        return *this;
    }
};

struct ConstructorDescription
{
    int activationMode = 13;   //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    int constructionActivationTime = 100;
    std::vector<uint8_t> genome;
    int genomeGeneration = 0;
    float constructionAngle1 = 0;
    float constructionAngle2 = 0;

    //process data
    int genomeCurrentNodeIndex = 0;
    bool isConstructionBuilt = false;
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
};

struct SensorDescription
{
    std::optional<float> fixedAngle;  //nullopt = entire neighborhood
    float minDensity = 0.05f;
    int color = 0;
    int targetedCreatureId = 0;

    //process data
    float memoryChannel1 = 0;
    float memoryChannel2 = 0;
    float memoryChannel3 = 0;

    auto operator<=>(SensorDescription const&) const = default;

    SensorMode getSensorMode() const { return fixedAngle.has_value() ? SensorMode_FixedAngle : SensorMode_Neighborhood; }
    SensorDescription& setFixedAngle(float value)
    {
        fixedAngle = value;
        return *this;
    }
    SensorDescription& setColor(int value)
    {
        color = value;
        return *this;
    }
    SensorDescription& setMinDensity(float value)
    {
        minDensity = value;
        return *this;
    }
};

struct NerveDescription
{
    int pulseMode = 0;          //0 = none, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    int alternationMode = 0;    //0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.

    auto operator<=>(NerveDescription const&) const = default;

    NerveDescription& setPulseMode(int value)
    {
        pulseMode = value;
        return *this;
    }
    NerveDescription& setAlternationMode(int value)
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

struct PlaceHolderDescription
{
    auto operator<=>(PlaceHolderDescription const&) const = default;
};

using CellFunctionDescription = std::optional<std::variant<
    NeuronDescription,
    TransmitterDescription,
    ConstructorDescription,
    SensorDescription,
    NerveDescription,
    AttackerDescription,
    InjectorDescription,
    MuscleDescription,
    DefenderDescription,
    PlaceHolderDescription>>;

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
    int maxConnections = 0;
    bool barrier = false;
    int age = 0;
    LivingState livingState = LivingState_Ready;
    int creatureId = 0;
    int mutationId = 0;

    //cell function
    int executionOrderNumber = 0;
    std::optional<int> inputExecutionOrderNumber;
    bool outputBlocked = false;
    CellFunctionDescription cellFunction;
    ActivityDescription activity;
    int activationTime = 0;
    int genomeSize = 0;

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
    CellDescription& setColor(unsigned char value)
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

    CellDescription& setMaxConnections(int value)
    {
        maxConnections = value;
        return *this;
    }
    CellDescription& setConnectingCells(std::vector<ConnectionDescription> const& value)
    {
        connections = value;
        return *this;
    }
    CellDescription& setExecutionOrderNumber(int value)
    {
        executionOrderNumber = value;
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
    CellDescription& setInputExecutionOrderNumber(int value)
    {
        inputExecutionOrderNumber = value;
        return *this;
    }
    CellDescription& setOutputBlocked(bool value)
    {
        outputBlocked = value;
        return *this;
    }
    CellFunction getCellFunctionType() const;
    template <typename CellFunctionDesc>
    CellDescription& setCellFunction(CellFunctionDesc const& value)
    {
        cellFunction = value;
        return *this;
    }
    CellDescription& setMetadata(CellMetadataDescription const& value)
    {
        metadata = value;
        return *this;
    }
    CellDescription& setActivity(ActivityDescription const& value)
    {
        activity = value;
        return *this;
    }
    CellDescription& setActivity(std::vector<float> const& value)
    {
        CHECK(value.size() == MAX_CHANNELS);

        ActivityDescription newActivity;
        newActivity.channels = value;
        activity = newActivity;
        return *this;
    }
    CellDescription& setActivationTime(int value)
    {
        activationTime = value;
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

private:
    CellDescription& getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>* cache = nullptr);
};

using CellOrParticleDescription = std::variant<CellDescription, ParticleDescription>;
