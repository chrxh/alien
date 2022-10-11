#include "DataConverter.h"

#include <algorithm>
#include <boost/range/adaptor/map.hpp>

#include "Base/NumberGenerator.h"
#include "Base/Exceptions.h"
#include "EngineInterface/Descriptions.h"


DataConverter::DataConverter(SimulationParameters const& parameters)
    : _parameters(parameters)
{}

ClusteredDataDescription DataConverter::convertAccessTOtoClusteredDataDescription(DataAccessTO const& dataTO, SortTokens sortTokens) const
{
	ClusteredDataDescription result;

    //cells
    std::vector<ClusterDescription> clusters;
    std::unordered_set<int> freeCellIndices;
    for (int i = 0; i < *dataTO.numCells; ++i) {
        freeCellIndices.insert(i);
    }
    std::unordered_map<int, int> cellTOIndexToCellDescIndex;
    std::unordered_map<int, int> cellTOIndexToClusterDescIndex;
    int clusterDescIndex = 0;
    while (!freeCellIndices.empty()) {
        auto freeCellIndex = *freeCellIndices.begin();
        auto createClusterData = scanAndCreateClusterDescription(dataTO, freeCellIndex, freeCellIndices);
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
    for (int i = 0; i < *dataTO.numTokens; ++i) {
        TokenAccessTO const& token = dataTO.tokens[i];

        std::string data(_parameters.tokenMemorySize, 0);
        for (int i = 0; i < _parameters.tokenMemorySize; ++i) {
            data[i] = token.memory[i];
        }
        auto clusterDescIndex = cellTOIndexToClusterDescIndex.at(token.cellIndex);
        auto cellDescIndex = cellTOIndexToCellDescIndex.at(token.cellIndex);
        CellDescription& cell = result.clusters.at(clusterDescIndex).cells.at(cellDescIndex);
        cell.addToken(TokenDescription().setEnergy(token.energy).setData(data).setSequenceNumber(token.sequenceNumber));
    }
    //sort tokens by sequence number
    if (sortTokens == SortTokens::Yes) {
        for (auto& cluster : result.clusters) {
            for (auto& cell : cluster.cells) {
                std::sort(
                    cell.tokens.begin(),
                    cell.tokens.end(),
                    [](TokenDescription const& left, TokenDescription const& right) {
                        return left.sequenceNumber < right.sequenceNumber;
                    });
            }
        }
    }

    //particles
    std::vector<ParticleDescription> particles;
    for (int i = 0; i < *dataTO.numParticles; ++i) {
        ParticleAccessTO const& particle = dataTO.particles[i];
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

DataDescription DataConverter::convertAccessTOtoDataDescription(DataAccessTO const& dataTO, SortTokens sortTokens) const
{
    DataDescription result;

    //cells
    std::vector<CellDescription> cells;
    for (int i = 0; i < *dataTO.numCells; ++i) {
        cells.emplace_back(createCellDescription(dataTO, i));
    }
    result.addCells(cells);

    //tokens
    for (int i = 0; i < *dataTO.numTokens; ++i) {
        TokenAccessTO const& token = dataTO.tokens[i];

        std::string data(_parameters.tokenMemorySize, 0);
        for (int i = 0; i < _parameters.tokenMemorySize; ++i) {
            data[i] = token.memory[i];
        }
        auto cellDescIndex = token.cellIndex;
        CellDescription& cell = result.cells.at(cellDescIndex);
        cell.addToken(TokenDescription().setEnergy(token.energy).setData(data).setSequenceNumber(token.sequenceNumber));
    }
    //sort tokens by sequence number
    if (sortTokens == SortTokens::Yes) {
        for (auto& cell: result.cells) {
            std::sort(cell.tokens.begin(), cell.tokens.end(), [](TokenDescription const& left, TokenDescription const& right) {
                return left.sequenceNumber < right.sequenceNumber;
            });
        }
    }

    //particles
    std::vector<ParticleDescription> particles;
    for (int i = 0; i < *dataTO.numParticles; ++i) {
        ParticleAccessTO const& particle = dataTO.particles[i];
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

OverlayDescription DataConverter::convertAccessTOtoOverlayDescription(DataAccessTO const& dataTO) const
{
    OverlayDescription result;
    result.elements.reserve(*dataTO.numCells + *dataTO.numParticles);
    for (int i = 0; i < *dataTO.numCells; ++i) {
        auto const& cellTO = dataTO.cells[i];
        OverlayElementDescription element;
        element.id = cellTO.id;
        element.cell = true;
        element.pos = {cellTO.pos.x, cellTO.pos.y};
        element.cellType = static_cast<Enums::CellFunction>(static_cast<unsigned int>(cellTO.cellFunctionType) % Enums::CellFunction_Count);
        element.selected = cellTO.selected;
        element.branchNumber = cellTO.branchNumber;
        result.elements.emplace_back(element);
    }

    for (int i = 0; i < *dataTO.numParticles; ++i) {
        auto const& particleTO = dataTO.particles[i];
        OverlayElementDescription element;
        element.id = particleTO.id;
        element.cell = false;
        element.pos = {particleTO.pos.x, particleTO.pos.y};
        element.selected = particleTO.selected;
        result.elements.emplace_back(element);
    }
    return result;
}

void DataConverter::convertClusteredDataDescriptionToAccessTO(DataAccessTO& result, ClusteredDataDescription const& description) const
{
    std::unordered_map<uint64_t, int> cellIndexByIds;
    for (auto const& cluster: description.clusters) {
        for (auto const& cell : cluster.cells) {
            addCell(result, cell, cellIndexByIds);
        }
    }
    for (auto const& cluster : description.clusters) {
        for (auto const& cell : cluster.cells) {
            if (cell.id != 0) {
                setConnections(result, cell, cellIndexByIds);
            }
        }
    }
    for (auto const& particle : description.particles) {
        addParticle(result, particle);
    }
}

void DataConverter::convertDataDescriptionToAccessTO(DataAccessTO& result, DataDescription const& description) const
{
    std::unordered_map<uint64_t, int> cellIndexByIds;
    for (auto const& cell : description.cells) {
        addCell(result, cell, cellIndexByIds);
    }
    for (auto const& cell : description.cells) {
        if (cell.id != 0) {
            setConnections(result, cell, cellIndexByIds);
        }
    }
    for (auto const& particle : description.particles) {
        addParticle(result, particle);
    }
}

void DataConverter::convertCellDescriptionToAccessTO(DataAccessTO& result, CellDescription const& cell) const
{
    std::unordered_map<uint64_t, int> cellIndexByIds;
    addCell(result, cell, cellIndexByIds);
}

void DataConverter::convertParticleDescriptionToAccessTO(DataAccessTO& result, ParticleDescription const& particle) const
{
    addParticle(result, particle);
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

auto DataConverter::scanAndCreateClusterDescription(
    DataAccessTO const& dataTO,
    int startCellIndex,
    std::unordered_set<int>& freeCellIndices) const
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
            cells.emplace_back(createCellDescription(dataTO, currentCellIndex));
            result.cellTOIndexToCellDescIndex.emplace(currentCellIndex, cellDescIndex);
            auto const& cellTO = dataTO.cells[currentCellIndex];
            for (int i = 0; i < cellTO.numConnections; ++i) {
                auto connectionTO = cellTO.connections[i];
                if (connectionTO.cellIndex != -1) {
                    if (scannedCellIndices.find(connectionTO.cellIndex) == scannedCellIndices.end()) {
                        nextCellIndices.insert(connectionTO.cellIndex);
                        scannedCellIndices.insert(connectionTO.cellIndex);
                    }
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

CellDescription DataConverter::createCellDescription(DataAccessTO const& dataTO, int cellIndex) const
{
    CellDescription result;

    auto const& cellTO = dataTO.cells[cellIndex];
    result.id = cellTO.id;
    result.pos = RealVector2D(cellTO.pos.x, cellTO.pos.y);
    result.vel = RealVector2D(cellTO.vel.x, cellTO.vel.y);
    result.energy = cellTO.energy;
    result.maxConnections = cellTO.maxConnections;
    std::vector<ConnectionDescription> connections;
    for (int i = 0; i < cellTO.numConnections; ++i) {
        auto const& connectionTO = cellTO.connections[i];
        ConnectionDescription connection;
        if (connectionTO.cellIndex != -1) {
            connection.cellId = dataTO.cells[connectionTO.cellIndex].id;
        } else {
            connection.cellId = 0;
        }
        connection.distance = connectionTO.distance;
        connection.angleFromPrevious = connectionTO.angleFromPrevious;
        connections.emplace_back(connection);
    }
    result.connections = connections;
    result.tokenBlocked = cellTO.tokenBlocked;
    result.tokenBranchNumber = cellTO.branchNumber;
    result.barrier = cellTO.barrier;
    result.age = cellTO.age;

    auto const& metadataTO = cellTO.metadata;
    auto metadata = CellMetadata().setColor(metadataTO.color);
    if (metadataTO.nameLen > 0) {
        auto const name = std::string(&dataTO.stringBytes[metadataTO.nameStringIndex], metadataTO.nameLen);
        metadata.setName(name);
    }
    if (metadataTO.descriptionLen > 0) {
        auto const description =
            std::string(&dataTO.stringBytes[metadataTO.descriptionStringIndex], metadataTO.descriptionLen);
        metadata.setDescription(description);
    }
    if (metadataTO.sourceCodeLen > 0) {
        auto const sourceCode =
            std::string(&dataTO.stringBytes[metadataTO.sourceCodeStringIndex], metadataTO.sourceCodeLen);
        metadata.setSourceCode(sourceCode);
    }
    result.metadata = metadata;

    auto feature = CellFeatureDescription()
                       .setType(static_cast<Enums::CellFunction>(cellTO.cellFunctionType))
                       .setStaticData(cellTO.staticData)
                       .setMutableData(cellTO.mutableData);
    result.cellFeature = feature;
    result.cellFunctionInvocations = cellTO.cellFunctionInvocations;

    return result;
}

void DataConverter::addParticle(DataAccessTO const& dataTO, ParticleDescription const& particleDesc) const
{
    auto particleIndex = (*dataTO.numParticles)++;

	ParticleAccessTO& particleTO = dataTO.particles[particleIndex];
	particleTO.id = particleDesc.id == 0 ? NumberGenerator::getInstance().getId() : particleDesc.id;
    particleTO.pos = {particleDesc.pos.x, particleDesc.pos.y};
    particleTO.vel = {particleDesc.vel.x, particleDesc.vel.y};
	particleTO.energy = toFloat(particleDesc.energy);
    particleTO.metadata.color = particleDesc.metadata.color;
}

int DataConverter::convertStringAndReturnStringIndex(DataAccessTO const& dataTO, std::string const& s) const
{
    auto result = *dataTO.numStringBytes;
    int len = static_cast<int>(s.size());
    for (int i = 0; i < len; ++i) {
        dataTO.stringBytes[result + i] = s.at(i);
    }
    (*dataTO.numStringBytes) += len;
    return result;
}

void DataConverter::addCell(
    DataAccessTO const& dataTO, CellDescription const& cellDesc, std::unordered_map<uint64_t, int>& cellIndexTOByIds) const
{
    int cellIndex = (*dataTO.numCells)++;
    CellAccessTO& cellTO = dataTO.cells[cellIndex];
    cellTO.id = cellDesc.id == 0 ? NumberGenerator::getInstance().getId() : cellDesc.id;
	cellTO.pos= { cellDesc.pos.x, cellDesc.pos.y };
    cellTO.vel = {cellDesc.vel.x, cellDesc.vel.y};
    cellTO.energy = toFloat(cellDesc.energy);
	cellTO.maxConnections = cellDesc.maxConnections;
    cellTO.branchNumber = cellDesc.tokenBranchNumber;
    cellTO.tokenBlocked = cellDesc.tokenBlocked;
    cellTO.cellFunctionInvocations = cellDesc.cellFunctionInvocations;
    auto const& cellFunction = cellDesc.cellFeature;
    cellTO.cellFunctionType = cellFunction.getType();
    memcpy(cellTO.staticData, cellFunction.staticData.data(), MAX_CELL_STATIC_BYTES);
    memcpy(cellTO.mutableData, cellFunction.mutableData.data(), MAX_CELL_MUTABLE_BYTES);
	cellTO.numConnections = 0;
    cellTO.barrier = cellDesc.barrier;
    cellTO.age = cellDesc.age;
    auto& metadataTO = cellTO.metadata;
    metadataTO.color = cellDesc.metadata.color;
    metadataTO.nameLen = toInt(cellDesc.metadata.name.size());
    if (metadataTO.nameLen > 0) {
        metadataTO.nameStringIndex = convertStringAndReturnStringIndex(dataTO, cellDesc.metadata.name);
    }
    metadataTO.descriptionLen = toInt(cellDesc.metadata.description.size());
    if (metadataTO.descriptionLen > 0) {
        metadataTO.descriptionStringIndex = convertStringAndReturnStringIndex(dataTO, cellDesc.metadata.description);
    }
    metadataTO.sourceCodeLen = toInt(cellDesc.metadata.computerSourcecode.size());
    if (metadataTO.sourceCodeLen > 0) {
        metadataTO.sourceCodeStringIndex =
            convertStringAndReturnStringIndex(dataTO, cellDesc.metadata.computerSourcecode);
    }

    for (int i = 0; i < cellDesc.tokens.size(); ++i) {
        TokenDescription const& tokenDesc = cellDesc.tokens.at(i);
        int tokenIndex = (*dataTO.numTokens)++;
        TokenAccessTO& tokenTO = dataTO.tokens[tokenIndex];
        tokenTO.energy = toFloat(tokenDesc.energy);
        tokenTO.cellIndex = cellIndex;
        convertToArray(tokenDesc.data, tokenTO.memory, _parameters.tokenMemorySize);
    }
	cellIndexTOByIds.insert_or_assign(cellTO.id, cellIndex);
}

void DataConverter::setConnections(DataAccessTO const& dataTO, CellDescription const& cellToAdd, std::unordered_map<uint64_t, int> const& cellIndexByIds) const
{
    int index = 0;
    auto& cellTO = dataTO.cells[cellIndexByIds.at(cellToAdd.id)];
    float angleOffset = 0;
    for (ConnectionDescription const& connection : cellToAdd.connections) {
        if (connection.cellId != 0) {
            cellTO.connections[index].cellIndex = cellIndexByIds.at(connection.cellId);
            cellTO.connections[index].distance = connection.distance;
            cellTO.connections[index].angleFromPrevious = connection.angleFromPrevious + angleOffset;
            ++index;
            angleOffset = 0;
        } else {
            angleOffset += connection.angleFromPrevious;
        }
    }
    if (angleOffset != 0 && index > 0) {
        cellTO.connections[0].angleFromPrevious += angleOffset;
    }
    cellTO.numConnections = index;
}

namespace
{
	void convert(RealVector2D const& input, float2& output)
	{
		output.x = input.x;
		output.y = input.y;
	}
}
