#pragma once

#include <variant>
#include <array>

#include "Base/Definitions.h"
#include "EngineInterface/Constants.h"

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
    std::vector<float> bias;

    NeuronDescription()
    {
        weights.resize(MAX_CHANNELS, std::vector<float>(MAX_CHANNELS, 0));
        bias.resize(MAX_CHANNELS, 0);
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
    int activationMode = 5;   //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    bool singleConstruction = false;
    bool separateConstruction = true;
    bool adaptMaxConnections = true;
    ConstructorAngleAlignment angleAlignment = ConstructorAngleAlignment_60;
    float stiffness = 1.0f;
    int constructionActivationTime = 100;
    std::vector<uint8_t> genome;

    //process data
    int currentGenomePos = 0;

    auto operator<=>(ConstructorDescription const&) const = default;

    ConstructorDescription& setActivationMode(int value)
    {
        activationMode = value;
        return *this;
    }
    ConstructorDescription& setSingleConstruction(bool value)
    {
        singleConstruction = value;
        return *this;
    }
    ConstructorDescription& setSeparateConstruction(bool value)
    {
        separateConstruction = value;
        return *this;
    }
    ConstructorDescription& setAdaptConnections(bool value)
    {
        adaptMaxConnections = value;
        return *this;
    }
    ConstructorDescription& setAngleAlignment(ConstructorAngleAlignment value)
    {
        angleAlignment = value;
        return *this;
    }
    ConstructorDescription& setStiffness(float value)
    {
        stiffness = value;
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
    ConstructorDescription& setCurrentGenomePos(int value)
    {
        currentGenomePos = value;
        return *this;
    }
};

struct SensorDescription
{
    std::optional<float> fixedAngle;  //nullopt = entire neighborhood
    float minDensity = 0.3f;
    int color = 0;

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
    std::vector<uint8_t> genome;

    auto operator<=>(InjectorDescription const&) const = default;
};

struct MuscleDescription
{
    MuscleMode mode = MuscleMode_Movement;

    auto operator<=>(MuscleDescription const&) const = default;

    MuscleDescription& setMode(MuscleMode value)
    {
        mode = value;
        return *this;
    }
};

struct PlaceHolderDescription1
{
    auto operator<=>(PlaceHolderDescription1 const&) const = default;
};

struct PlaceHolderDescription2
{
    auto operator<=>(PlaceHolderDescription2 const&) const = default;
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
    PlaceHolderDescription1,
    PlaceHolderDescription2>>;

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

    //cell function
    LivingState livingState = LivingState_Ready;
    int executionOrderNumber = 0;
    bool inputBlocked = false;
    bool outputBlocked = false;
    CellFunctionDescription cellFunction;
    ActivityDescription activity;
    int activationTime = 0;

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
    CellDescription& setConstructionState(LivingState value)
    {
        livingState = value;
        return *this;
    }
    CellDescription& setInputBlocked(bool value)
    {
        inputBlocked = value;
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

    bool isConnectedTo(uint64_t id) const;
};

struct ClusterDescription
{
    uint64_t id = 0;

    std::vector<CellDescription> cells;

    ClusterDescription() = default;
    auto operator<=>(ClusterDescription const&) const = default;

    ClusterDescription& setId(uint64_t value)
    {
        id = value;
        return *this;
    }
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
