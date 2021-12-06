#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"
#include "Metadata.h"
#include "DllExport.h"

struct CellFeatureDescription
{
	std::string volatileData;
    std::string constData;

    Enums::CellFunction::Type getType() const
    {
        return static_cast<Enums::CellFunction::Type>(static_cast<unsigned char>(_type) % Enums::CellFunction::_COUNTER);
    }
	CellFeatureDescription& setType(Enums::CellFunction::Type value) { _type = value; return *this; }
    CellFeatureDescription& setVolatileData(std::string const& value)
    {
        volatileData = value;
        return *this;
    }
    CellFeatureDescription& setConstData(std::string const& value)
    {
        constData = value;
        return *this;
    }
	bool operator==(CellFeatureDescription const& other) const {
		return _type == other._type
			&& volatileData == other.volatileData
			&& constData == other.constData;
	}
	bool operator!=(CellFeatureDescription const& other) const { return !operator==(other); }

private:
    Enums::CellFunction::Type _type = Enums::CellFunction::COMPUTER;
};

struct TokenDescription2
{
    double energy = 0;
    std::string data;

    TokenDescription2& setEnergy(double value)
    {
        energy = value;
        return *this;
    }
    TokenDescription2& setData(std::string const& value)
    {
        data = value;
        return *this;
    }
    bool operator==(TokenDescription2 const& other) const { return energy == other.energy && data == other.data; }
    bool operator!=(TokenDescription2 const& other) const { return !operator==(other); }
};

struct ConnectionDescription
{
    uint64_t cellId;
    float distance = 0;
    float angleFromPrevious = 0;
};

struct CellDescription2
{
    uint64_t id = 0;

    RealVector2D pos;
    RealVector2D vel;
    double energy;
    int maxConnections;
    std::vector<ConnectionDescription> connections;
    bool tokenBlocked;
    int tokenBranchNumber;
    CellMetadata metadata;
    CellFeatureDescription cellFeature;
    std::vector<TokenDescription2> tokens;
    int tokenUsages;

    ENGINEINTERFACE_EXPORT CellDescription2() = default;
    ENGINEINTERFACE_EXPORT CellDescription2(CellChangeDescription const& change);
    CellDescription2& setId(uint64_t value)
    {
        id = value;
        return *this;
    }
    CellDescription2& setPos(RealVector2D const& value)
    {
        pos = value;
        return *this;
    }
    CellDescription2& setVel(RealVector2D const& value)
    {
        vel = value;
        return *this;
    }
    CellDescription2& setEnergy(double value)
    {
        energy = value;
        return *this;
    }
    CellDescription2& setMaxConnections(int value)
    {
        maxConnections = value;
        return *this;
    }
    CellDescription2& setConnectingCells(std::vector<ConnectionDescription> const& value)
    {
        connections = value;
        return *this;
    }
    CellDescription2& setFlagTokenBlocked(bool value)
    {
        tokenBlocked = value;
        return *this;
    }
    CellDescription2& setTokenBranchNumber(int value)
    {
        tokenBranchNumber = value;
        return *this;
    }
    CellDescription2& setMetadata(CellMetadata const& value)
    {
        metadata = value;
        return *this;
    }
    CellDescription2& setCellFeature(CellFeatureDescription const& value)
    {
        cellFeature = value;
        return *this;
    }
    CellDescription2& setTokens(vector<TokenDescription2> const& value)
    {
        tokens = value;
        return *this;
    }
    ENGINEINTERFACE_EXPORT CellDescription2& addToken(TokenDescription2 const& value);
    ENGINEINTERFACE_EXPORT CellDescription2& addToken(int index, TokenDescription2 const& value);
    ENGINEINTERFACE_EXPORT CellDescription2& delToken(int index);
    CellDescription2& setTokenUsages(int value)
    {
        tokenUsages = value;
        return *this;
    }
    ENGINEINTERFACE_EXPORT bool isConnectedTo(uint64_t id) const;
};

struct ClusterDescription2
{
    uint64_t id = 0;

    std::vector<CellDescription2> cells;

    ENGINEINTERFACE_EXPORT ClusterDescription2() = default;

    ClusterDescription2& setId(uint64_t value)
    {
        id = value;
        return *this;
    }
    ClusterDescription2& addCells(std::list<CellDescription2> const& value)
    {
        cells.insert(cells.end(), value.begin(), value.end());
        return *this;
    }
    ClusterDescription2& addCell(CellDescription2 const& value)
    {
        addCells({value});
        return *this;
    }

    ENGINEINTERFACE_EXPORT ClusterDescription2&
    addConnection(uint64_t const& cellId1, uint64_t const& cellId2, std::unordered_map<uint64_t, int>& cache);

    ENGINEINTERFACE_EXPORT RealVector2D getClusterPosFromCells() const;

private:
    CellDescription2& getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>& cache);
};

struct ParticleDescription2
{
    uint64_t id = 0;

    RealVector2D pos;
    RealVector2D vel;
    double energy;
    ParticleMetadata metadata;

    ENGINEINTERFACE_EXPORT ParticleDescription2() = default;
    ENGINEINTERFACE_EXPORT ParticleDescription2(ParticleChangeDescription const& change);
    ParticleDescription2& setId(uint64_t value)
    {
        id = value;
        return *this;
    }
    ParticleDescription2& setPos(RealVector2D const& value)
    {
        pos = value;
        return *this;
    }
    ParticleDescription2& setVel(RealVector2D const& value)
    {
        vel = value;
        return *this;
    }
    ParticleDescription2& setEnergy(double value)
    {
        energy = value;
        return *this;
    }
    ParticleDescription2& setMetadata(ParticleMetadata const& value)
    {
        metadata = value;
        return *this;
    }
};

struct DataDescription2
{
    vector<ClusterDescription2> clusters;
    vector<ParticleDescription2> particles;

    ENGINEINTERFACE_EXPORT DataDescription2() = default;
    DataDescription2& addClusters(list<ClusterDescription2> const& value)
    {
        clusters.insert(clusters.end(), value.begin(), value.end());
        return *this;
    }
    DataDescription2& addCluster(ClusterDescription2 const& value)
    {
        addClusters({value});
        return *this;
    }

    DataDescription2& addParticles(list<ParticleDescription2> const& value)
    {
        particles.insert(particles.end(), value.begin(), value.end());
        return *this;
    }
    DataDescription2& addParticle(ParticleDescription2 const& value)
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
    RealVector2D calcCenter() const;
    void shift(RealVector2D const& delta);
};


struct DescriptionNavigator
{
	std::unordered_set<uint64_t> cellIds;
    std::unordered_set<uint64_t> particleIds;
    std::map<uint64_t, uint64_t> clusterIdsByCellIds;
    std::map<uint64_t, int> clusterIndicesByClusterIds;
    std::map<uint64_t, int> clusterIndicesByCellIds;
    std::map<uint64_t, int> cellIndicesByCellIds;
    std::map<uint64_t, int> particleIndicesByParticleIds;

	void update(DataDescription2 const& data)
	{
		cellIds.clear();
		particleIds.clear();
		clusterIdsByCellIds.clear();
		clusterIndicesByCellIds.clear();
		clusterIndicesByClusterIds.clear();
		cellIndicesByCellIds.clear();
		particleIndicesByParticleIds.clear();

		int clusterIndex = 0;
		for (auto const &cluster : data.clusters) {
			clusterIndicesByClusterIds.insert_or_assign(cluster.id, clusterIndex);
			int cellIndex = 0;
			for (auto const &cell : cluster.cells) {
				clusterIdsByCellIds.insert_or_assign(cell.id, cluster.id);
				clusterIndicesByCellIds.insert_or_assign(cell.id, clusterIndex);
				cellIndicesByCellIds.insert_or_assign(cell.id, cellIndex);
				cellIds.insert(cell.id);
				++cellIndex;
			}
			++clusterIndex;
		}

		int particleIndex = 0;
		for (auto const &particle : data.particles) {
			particleIndicesByParticleIds.insert_or_assign(particle.id, particleIndex);
			particleIds.insert(particle.id);
			++particleIndex;
		}
	}
};
