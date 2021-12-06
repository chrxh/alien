#include "DataConverter.h"

#include <algorithm>
#include <boost/range/adaptor/map.hpp>

#include "Base/NumberGenerator.h"
#include "Base/Exceptions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/ChangeDescriptions.h"


DataConverter::DataConverter(
    DataAccessTO& dataTO,
    SimulationParameters const& parameters,
    GpuSettings const& gpuConstants)
    : _dataTO(dataTO)
    , _parameters(parameters)
    , _gpuConstants(gpuConstants)
{}

void DataConverter::updateData(DataChangeDescription const & data)
{
    unordered_map<uint64_t, int> cellIndexByIds;
    for (auto const& cell : data.cells) {
		if (cell.isAdded()) {
            addCell(cell.getValue(), cellIndexByIds);
		}
	}
    for (auto const& cell : data.cells) {
        if (cell.isAdded()) {
            if (cell->id != 0) {
                setConnections(cell.getValue(), cellIndexByIds);
            }
        }
    }
	for (auto const& particle : data.particles) {
		if (particle.isAdded()) {
			addParticle(particle.getValue());
		}
	}
}

namespace
{
    std::string convertToString(char const* data, int size) {
        std::string result(size, 0);
        for (int i = 0; i < size; ++i) {
            result[i] = data[i];
        }
        return result;
    }

    void convertToArray(std::string const& source, char* target, int size)
    {
        for (int i = 0; i < size; ++i) {
            if (i < source.size()) {
                target[i] = source.at(i);
            }
            else {
                target[i] = 0;
            }
        }
    }
}

DataDescription DataConverter::getDataDescription() const
{
	DataDescription result;

    //cells
    std::vector<ClusterDescription> clusters;
	std::unordered_set<int> freeCellIndices;
    for (int i = 0; i < *_dataTO.numCells; ++i) {
        freeCellIndices.insert(i);
    }
    std::unordered_map<int, int> cellTOIndexToCellDescIndex;
    std::unordered_map<int, int> cellTOIndexToClusterDescIndex;
    int clusterDescIndex = 0;
    while (!freeCellIndices.empty()) {
        auto freeCellIndex = *freeCellIndices.begin();
        auto createClusterData = scanAndCreateClusterDescription(freeCellIndex, freeCellIndices);
        clusters.emplace_back(createClusterData.cluster);

        //update index maps
        cellTOIndexToCellDescIndex.insert(
            createClusterData.cellTOIndexToCellDescIndex.begin(), createClusterData.cellTOIndexToCellDescIndex.end());
        for (auto const& cellTOIndex : createClusterData.cellTOIndexToCellDescIndex | boost::adaptors::map_keys) {
            cellTOIndexToClusterDescIndex.emplace(cellTOIndex, clusterDescIndex);
        }
        ++clusterDescIndex;
    }
    result.addClusters(clusters);

    //tokens
    for (int i = 0; i < *_dataTO.numTokens; ++i) {
        TokenAccessTO const& token = _dataTO.tokens[i];

  		std::string data(_parameters.tokenMemorySize, 0);
        for (int i = 0; i < _parameters.tokenMemorySize; ++i) {
            data[i] = token.memory[i];
        }
        auto clusterDescIndex = cellTOIndexToClusterDescIndex.at(token.cellIndex); 
        auto cellDescIndex = cellTOIndexToCellDescIndex.at(token.cellIndex);
        CellDescription& cell = result.clusters.at(clusterDescIndex).cells.at(cellDescIndex);
            
        cell.addToken(TokenDescription().setEnergy(token.energy).setData(data));
    }

    //particles
    std::vector<ParticleDescription> particles;
    for (int i = 0; i < *_dataTO.numParticles; ++i) {
        ParticleAccessTO const& particle = _dataTO.particles[i];
        particles.emplace_back(ParticleDescription()
                               .setId(particle.id)
                               .setPos({particle.pos.x, particle.pos.y})
                               .setVel({particle.vel.x, particle.vel.y})
                               .setEnergy(particle.energy)
                               .setMetadata(ParticleMetadata().setColor(particle.metadata.color)));
    }
    result.addParticles(particles);

    return result;
}

namespace
{
    template <typename T>
    void setInplaceDifference(std::unordered_set<T>& a, std::unordered_set<T> const& b)
    {
        for (auto const& element : b) {
            a.erase(element);
        }
    }
}

auto DataConverter::scanAndCreateClusterDescription(int startCellIndex, std::unordered_set<int>& freeCellIndices) const
    -> CreateClusterReturnData
{
    CreateClusterReturnData result; 

    std::unordered_set<int> currentCellIndices;
    currentCellIndices.insert(startCellIndex);
    std::unordered_set<int> scannedCellIndices = currentCellIndices;

    std::vector<CellDescription> cells;
    std::unordered_set<int> nextCellIndices;
    int cellDescIndex = 0;
    do {
        for (auto const& currentCellIndex : currentCellIndices) {
            cells.emplace_back(createCellDescription(currentCellIndex));
            result.cellTOIndexToCellDescIndex.emplace(currentCellIndex, cellDescIndex);
            auto const& cellTO = _dataTO.cells[currentCellIndex];
            for (int i = 0; i < cellTO.numConnections; ++i) {
                auto connectionTO = cellTO.connections[i];
                if (scannedCellIndices.find(connectionTO.cellIndex) == scannedCellIndices.end()) {
                    nextCellIndices.insert(connectionTO.cellIndex);
                    scannedCellIndices.insert(connectionTO.cellIndex);
                }
            }
            ++cellDescIndex;
        }
        currentCellIndices = nextCellIndices;
        nextCellIndices.clear();
    } while (!currentCellIndices.empty());

    setInplaceDifference(freeCellIndices, scannedCellIndices);

    result.cluster.id = NumberGenerator::getInstance().getId();
    result.cluster.addCells(cells);

    return result;
}

CellDescription DataConverter::createCellDescription(int cellIndex) const
{
    CellDescription result;

    auto const& cellTO = _dataTO.cells[cellIndex];
    result.id = cellTO.id;
    result.pos = RealVector2D(cellTO.pos.x, cellTO.pos.y);
    result.vel = RealVector2D(cellTO.vel.x, cellTO.vel.y);
    result.energy = cellTO.energy;
    result.maxConnections = cellTO.maxConnections;
    std::vector<ConnectionDescription> connections;
    for (int i = 0; i < cellTO.numConnections; ++i) {
        auto const& connectionTO = cellTO.connections[i];
        ConnectionDescription connection;
        connection.cellId = _dataTO.cells[connectionTO.cellIndex].id;
        connection.distance = connectionTO.distance;
        connection.angleFromPrevious = connectionTO.angleFromPrevious;
        connections.emplace_back(connection);
    }
    result.connections = connections;
    result.tokenBlocked = cellTO.tokenBlocked;
    result.tokenBranchNumber = cellTO.branchNumber;

    auto const& metadataTO = cellTO.metadata;
    auto metadata = CellMetadata().setColor(metadataTO.color);
    if (metadataTO.nameLen > 0) {
        auto const name = std::string(&_dataTO.stringBytes[metadataTO.nameStringIndex], metadataTO.nameLen);
        metadata.setName(name);
    }
    if (metadataTO.descriptionLen > 0) {
        auto const description =
            std::string(&_dataTO.stringBytes[metadataTO.descriptionStringIndex], metadataTO.descriptionLen);
        metadata.setDescription(description);
    }
    if (metadataTO.sourceCodeLen > 0) {
        auto const sourceCode =
            std::string(&_dataTO.stringBytes[metadataTO.sourceCodeStringIndex], metadataTO.sourceCodeLen);
        metadata.setSourceCode(sourceCode);
    }
    result.metadata = metadata;

    auto feature = CellFeatureDescription()
                       .setType(static_cast<Enums::CellFunction::Type>(cellTO.cellFunctionType))
                       .setConstData(convertToString(cellTO.staticData, cellTO.numStaticBytes))
                       .setVolatileData(convertToString(cellTO.mutableData, cellTO.numMutableBytes));
    result.cellFeature = feature;
    result.tokenUsages = cellTO.tokenUsages;

    return result;
}

void DataConverter::addParticle(ParticleDescription const & particleDesc)
{
    auto particleIndex = (*_dataTO.numParticles)++;

	ParticleAccessTO& particleTO = _dataTO.particles[particleIndex];
	particleTO.id = particleDesc.id == 0 ? NumberGenerator::getInstance().getId() : particleDesc.id;
	particleTO.pos = { particleDesc.pos.x, particleDesc.pos.y };
	particleTO.vel = { particleDesc.vel.x, particleDesc.vel.y };
	particleTO.energy = toFloat(particleDesc.energy);
    particleTO.metadata.color = particleDesc.metadata.color;
}

int DataConverter::convertStringAndReturnStringIndex(std::string const& s)
{
    auto result = *_dataTO.numStringBytes;
    int len = static_cast<int>(s.size());
    for (int i = 0; i < len; ++i) {
        _dataTO.stringBytes[result + i] = s.at(i);
    }
    (*_dataTO.numStringBytes) += len;
    return result;
}

void DataConverter::addCell(
    CellChangeDescription const& cellDesc,
    unordered_map<uint64_t, int>& cellIndexTOByIds)
{
	int cellIndex = (*_dataTO.numCells)++;
	CellAccessTO& cellTO = _dataTO.cells[cellIndex];
    cellTO.id = cellDesc.id == 0 ? NumberGenerator::getInstance().getId() : cellDesc.id;
	cellTO.pos= { cellDesc.pos->x, cellDesc.pos->y };
    cellTO.vel = {cellDesc.vel->x, cellDesc.vel->y};
    cellTO.energy = toFloat(*cellDesc.energy);
	cellTO.maxConnections = *cellDesc.maxConnections;
    cellTO.branchNumber = cellDesc.tokenBranchNumber.getOptionalValue().get_value_or(0);
    cellTO.tokenBlocked = cellDesc.tokenBlocked.getOptionalValue().get_value_or(false);
    cellTO.tokenUsages = cellDesc.tokenUsages.getOptionalValue().get_value_or(0);
    auto const& cellFunction = cellDesc.cellFeatures.getOptionalValue().get_value_or(CellFeatureDescription());
    cellTO.cellFunctionType = cellFunction.getType();
    cellTO.numStaticBytes = std::min(static_cast<int>(cellFunction.constData.size()), MAX_CELL_STATIC_BYTES);
    cellTO.numMutableBytes = std::min(static_cast<int>(cellFunction.volatileData.size()), MAX_CELL_MUTABLE_BYTES);
    convertToArray(cellFunction.constData, cellTO.staticData, MAX_CELL_STATIC_BYTES);
    convertToArray(cellFunction.volatileData, cellTO.mutableData, MAX_CELL_MUTABLE_BYTES);
    if (cellDesc.connectingCells.getOptionalValue()) {
		cellTO.numConnections = toInt(cellDesc.connectingCells->size());
	}
	else {
		cellTO.numConnections = 0;
	}
    if (cellDesc.metadata.getOptionalValue()) {
        auto& metadataTO = cellTO.metadata;
        metadataTO.color = cellDesc.metadata->color;
        metadataTO.nameLen = toInt(cellDesc.metadata->name.size());
        if (metadataTO.nameLen > 0) {
            metadataTO.nameStringIndex = convertStringAndReturnStringIndex(cellDesc.metadata->name);
        }
        metadataTO.descriptionLen = toInt(cellDesc.metadata->description.size());
        if (metadataTO.descriptionLen > 0) {
            metadataTO.descriptionStringIndex = convertStringAndReturnStringIndex(cellDesc.metadata->description);
        }
        metadataTO.sourceCodeLen = toInt(cellDesc.metadata->computerSourcecode.size());
        if (metadataTO.sourceCodeLen > 0) {
            metadataTO.sourceCodeStringIndex = convertStringAndReturnStringIndex(cellDesc.metadata->computerSourcecode);
        }
    }
    else {
        cellTO.metadata.color = 0;
        cellTO.metadata.nameLen = 0;
        cellTO.metadata.descriptionLen = 0;
        cellTO.metadata.sourceCodeLen= 0;
    }

    if (cellDesc.tokens.getOptionalValue()) {
        for (int i = 0; i < cellDesc.tokens->size(); ++i) {
            TokenDescription const& tokenDesc = cellDesc.tokens->at(i);
            int tokenIndex = (*_dataTO.numTokens)++;
            TokenAccessTO& tokenTO = _dataTO.tokens[tokenIndex];
            tokenTO.energy = toFloat(tokenDesc.energy);
            tokenTO.cellIndex = cellIndex;
            convertToArray(tokenDesc.data, tokenTO.memory, _parameters.tokenMemorySize);
        }
    }
	cellIndexTOByIds.insert_or_assign(cellTO.id, cellIndex);
}

void DataConverter::setConnections(
    CellChangeDescription const& cellToAdd, unordered_map<uint64_t, int> const& cellIndexByIds)
{
	if (cellToAdd.connectingCells.getOptionalValue()) {
		int index = 0;
        auto& cellTO = _dataTO.cells[cellIndexByIds.at(cellToAdd.id)];
        for (ConnectionChangeDescription const& connection : *cellToAdd.connectingCells) {
            cellTO.connections[index].cellIndex = cellIndexByIds.at(connection.cellId);
            cellTO.connections[index].distance = toFloat(connection.distance);
            cellTO.connections[index].angleFromPrevious = toFloat(connection.angleFromPrevious);
            ++index;
		}
        cellTO.numConnections = index;
	}
}

namespace
{
	void convert(RealVector2D const& input, float2& output)
	{
		output.x = input.x;
		output.y = input.y;
	}
}
