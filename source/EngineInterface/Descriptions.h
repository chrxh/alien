#pragma once

#include <variant>

#include "Base/Definitions.h"

#include "Definitions.h"
#include "Metadata.h"

struct ConnectionDescription
{
    uint64_t cellId;    //value of 0 means cell not present in DataDescription
    float distance = 0;
    float angleFromPrevious = 0;
};

struct CellDescription
{
    uint64_t id = 0;
    std::vector<ConnectionDescription> connections;

    RealVector2D pos;
    RealVector2D vel;
    double energy;
    int color;
    int maxConnections;
    int executionOrderNumber;
    bool barrier;
    int age;

    bool underConstruction;
    bool inputBlocked;
    bool outputBlocked;
    Enums::CellFunction cellFunction = Enums::CellFunction_None;

    CellMetadata metadata;

    CellDescription() = default;
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
    CellDescription& setEnergy(double value)
    {
        energy = value;
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
    CellDescription& setUnderConstruction(bool value)
    {
        underConstruction = value;
        return *this;
    }
    CellDescription& setExecutionOrderNumber(int value)
    {
        executionOrderNumber = value;
        return *this;
    }
    CellDescription& setMetadata(CellMetadata const& value)
    {
        metadata = value;
        return *this;
    }
    bool isConnectedTo(uint64_t id) const;
};

struct ClusterDescription
{
    uint64_t id = 0;

    std::vector<CellDescription> cells;

    ClusterDescription() = default;

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
    double energy;
    ParticleMetadata metadata;

    ParticleDescription() = default;
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
    ParticleDescription& setEnergy(double value)
    {
        energy = value;
        return *this;
    }
    ParticleDescription& setMetadata(ParticleMetadata const& value)
    {
        metadata = value;
        return *this;
    }
};

struct ClusteredDataDescription
{
    std::vector<ClusterDescription> clusters;
    std::vector<ParticleDescription> particles;

    ClusteredDataDescription() = default;
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

    DataDescription& addConnection(uint64_t const& cellId1, uint64_t const& cellId2, std::unordered_map<uint64_t, int>& cache);

private:
    CellDescription& getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>& cache);
};

using CellOrParticleDescription = std::variant<CellDescription, ParticleDescription>;
