#include <algorithm>
#include <boost/range/adaptor/map.hpp>

#include "Base/NumberGenerator.h"
#include "Base/Exceptions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/ChangeDescriptions.h"
#include "EngineInterface/Physics.h"

#include "DataConverter.h"

DataConverter::DataConverter(
    DataAccessTO& dataTO,
    NumberGenerator* numberGen,
    SimulationParameters const& parameters,
    CudaConstants const& cudaConstants)
    : _dataTO(dataTO)
    , _numberGen(numberGen)
    , _parameters(parameters)
    , _cudaConstants(cudaConstants)
{}

void DataConverter::updateData(DataChangeDescription const & data)
{
	for (auto const& cell : data.cells) {
		if (cell.isDeleted()) {
			markDelCell(cell.getValue().id);
		}
		if (cell.isModified()) {
			markModifyCell(cell.getValue());
		}
	}
	for (auto const& particle : data.particles) {
		if (particle.isDeleted()) {
			markDelParticle(particle.getValue().id);
		}
		if (particle.isModified()) {
			markModifyParticle(particle.getValue());
		}
	}

	processDeletions();
	processModifications();

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
    QByteArray convertToQByteArray(char const* data, int size)
    {
        QByteArray result;
        result.reserve(size);
        for (int i = 0; i < size; ++i) {
            result.append(data[i]);
        }
        return result;
    }

    void convertToArray(QByteArray const& source, char* target, int size)
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
    std::list<ClusterDescription> clusters;
	std::set<int> freeCellIndices;
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

  		QByteArray data(_parameters.tokenMemorySize, 0);
        for (int i = 0; i < _parameters.tokenMemorySize; ++i) {
            data[i] = token.memory[i];
        }
        auto clusterDescIndex = cellTOIndexToClusterDescIndex.at(token.cellIndex); 
        auto cellDescIndex = cellTOIndexToCellDescIndex.at(token.cellIndex);
        CellDescription& cell = result.clusters->at(clusterDescIndex).cells->at(cellDescIndex);
            
        cell.addToken(TokenDescription().setEnergy(token.energy).setData(data));
    }

    //particles
    std::list<ParticleDescription> particles;
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

/*
	list<uint64_t> connectingCellIds;
	unordered_map<int, int> cellIndexByCellTOIndex;
	unordered_map<int, int> clusterIndexByCellTOIndex;
	for (int i = 0; i < *_dataTO.numClusters; ++i) {
		ClusterAccessTO const& clusterTO = _dataTO.clusters[i];

        auto metadata = ClusterMetadata();
        auto const metadataTO = clusterTO.metadata;
        if (metadataTO.nameLen > 0) {
            auto const name = QString::fromLatin1(&_dataTO.stringBytes[metadataTO.nameStringIndex], metadataTO.nameLen);
            metadata.setName(name);
        }

		auto clusterDesc = ClusterDescription().setId(clusterTO.id).setPos({ clusterTO.pos.x, clusterTO.pos.y })
			.setVel({ clusterTO.vel.x, clusterTO.vel.y })
			.setAngle(clusterTO.angle)
			.setAngularVel(clusterTO.angularVel).setMetadata(metadata);

		for (int j = 0; j < clusterTO.numCells; ++j) {
			CellAccessTO const& cellTO = _dataTO.cells[clusterTO.cellStartIndex + j];
			auto pos = cellTO.pos;
			auto id = cellTO.id;
			connectingCellIds.clear();
			for (int i = 0; i < cellTO.numConnections; ++i) {
				connectingCellIds.emplace_back(_dataTO.cells[cellTO.connectionIndices[i]].id);
			}
			cellIndexByCellTOIndex.insert_or_assign(clusterTO.cellStartIndex + j, j);
			clusterIndexByCellTOIndex.insert_or_assign(clusterTO.cellStartIndex + j, i);

            auto feature = CellFeatureDescription().setType(static_cast<Enums::CellFunction::Type>(cellTO.cellFunctionType))
                .setConstData(convertToQByteArray(cellTO.staticData, cellTO.numStaticBytes)).setVolatileData(convertToQByteArray(cellTO.mutableData, cellTO.numMutableBytes));

            auto const& metadataTO = cellTO.metadata;
            auto metadata = CellMetadata().setColor(metadataTO.color);
            if (metadataTO.nameLen > 0) {
                auto const name = QString::fromLatin1(&_dataTO.stringBytes[metadataTO.nameStringIndex], metadataTO.nameLen);
                metadata.setName(name);
            }
            if (metadataTO.descriptionLen > 0) {
                auto const description = QString::fromLatin1(&_dataTO.stringBytes[metadataTO.descriptionStringIndex], metadataTO.descriptionLen);
                metadata.setDescription(description);
            }
            if (metadataTO.sourceCodeLen > 0) {
                auto const sourceCode = QString::fromLatin1(&_dataTO.stringBytes[metadataTO.sourceCodeStringIndex], metadataTO.sourceCodeLen);
                metadata.setSourceCode(sourceCode);
            }

            clusterDesc.addCell(CellDescription()
                                    .setPos({pos.x, pos.y})
                                    .setMetadata(CellMetadata())
                                    .setEnergy(cellTO.energy)
                                    .setId(id)
                                    .setConnectingCells(connectingCellIds)
                                    .setMaxConnections(cellTO.maxConnections)
                                    .setTokenBranchNumber(0)
                                    .setMetadata(metadata)
                                    .setTokens(vector<TokenDescription>{})
                                    .setTokenBranchNumber(cellTO.branchNumber)
                                    .setFlagTokenBlocked(cellTO.tokenBlocked)
                                    .setTokenUsages(cellTO.tokenUsages)
                                    .setCellFeature(feature));
        }
		result.addCluster(clusterDesc);
	}

	for (int i = 0; i < *_dataTO.numParticles; ++i) {
		ParticleAccessTO const& particle = _dataTO.particles[i];
		result.addParticle(ParticleDescription().setId(particle.id).setPos({ particle.pos.x, particle.pos.y })
			.setVel({ particle.vel.x, particle.vel.y }).setEnergy(particle.energy).setMetadata(ParticleMetadata().setColor(particle.metadata.color)));
	}

	for (int i = 0; i < *_dataTO.numTokens; ++i) {
		TokenAccessTO const& token = _dataTO.tokens[i];
		ClusterDescription& cluster = result.clusters->at(clusterIndexByCellTOIndex.at(token.cellIndex));
		CellDescription& cell = cluster.cells->at(cellIndexByCellTOIndex.at(token.cellIndex));
		QByteArray data(_parameters.tokenMemorySize, 0);
		for (int i = 0; i < _parameters.tokenMemorySize; ++i) {
			data[i] = token.memory[i];
		}
		cell.addToken(TokenDescription().setEnergy(token.energy).setData(data));
	}

	return result;
*/
}

auto DataConverter::scanAndCreateClusterDescription(
    int startCellIndex, std::set<int>& freeCellIndices) const
    -> CreateClusterReturnData
{
    CreateClusterReturnData result; 

    std::set<int> currentCellIndices;
    currentCellIndices.insert(startCellIndex);
    std::set<int> scannedCellIndices = currentCellIndices;

    std::list<CellDescription> cells;
    std::set<int> nextCellIndices;
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

    std::set<int> newFreeCellIndices;
    std::set_difference(
        freeCellIndices.begin(),
        freeCellIndices.end(),
        scannedCellIndices.begin(),
        scannedCellIndices.end(),
        std::inserter(newFreeCellIndices, newFreeCellIndices.begin()));
    freeCellIndices = std::move(newFreeCellIndices);

    result.cluster.id = _numberGen->getId();
    result.cluster.addCells(cells);

    return result;
}

CellDescription DataConverter::createCellDescription(int cellIndex) const
{
    CellDescription result;

    auto const& cellTO = _dataTO.cells[cellIndex];
    result.id = cellTO.id;
    result.pos = QVector2D(cellTO.pos.x, cellTO.pos.y);
    result.vel = QVector2D(cellTO.vel.x, cellTO.vel.y);
    result.energy = cellTO.energy;
    result.maxConnections = cellTO.maxConnections;
    list<ConnectionDescription> connections;
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
        auto const name = QString::fromLatin1(&_dataTO.stringBytes[metadataTO.nameStringIndex], metadataTO.nameLen);
        metadata.setName(name);
    }
    if (metadataTO.descriptionLen > 0) {
        auto const description =
            QString::fromLatin1(&_dataTO.stringBytes[metadataTO.descriptionStringIndex], metadataTO.descriptionLen);
        metadata.setDescription(description);
    }
    if (metadataTO.sourceCodeLen > 0) {
        auto const sourceCode =
            QString::fromLatin1(&_dataTO.stringBytes[metadataTO.sourceCodeStringIndex], metadataTO.sourceCodeLen);
        metadata.setSourceCode(sourceCode);
    }
    result.metadata = metadata;

    auto feature = CellFeatureDescription()
                       .setType(static_cast<Enums::CellFunction::Type>(cellTO.cellFunctionType))
                       .setConstData(convertToQByteArray(cellTO.staticData, cellTO.numStaticBytes))
                       .setVolatileData(convertToQByteArray(cellTO.mutableData, cellTO.numMutableBytes));
    result.cellFeature = feature;
    result.tokenUsages = cellTO.tokenUsages;

    return result;
}

void DataConverter::addParticle(ParticleDescription const & particleDesc)
{
    auto particleIndex = (*_dataTO.numParticles)++;
    if (particleIndex >= _cudaConstants.MAX_PARTICLES) {
        throw BugReportException("Array size for particles is chosen too small.");
    }

	ParticleAccessTO& particleTO = _dataTO.particles[particleIndex];
	particleTO.id = particleDesc.id == 0 ? _numberGen->getId() : particleDesc.id;
	particleTO.pos = { particleDesc.pos->x(), particleDesc.pos->y() };
	particleTO.vel = { particleDesc.vel->x(), particleDesc.vel->y() };
	particleTO.energy = *particleDesc.energy;
    if (auto const& metadata = particleDesc.metadata) {
        particleTO.metadata.color = metadata->color;
    }
    else {
        particleTO.metadata.color = 0;
    }
}

void DataConverter::markDelCell(uint64_t cellId)
{
	_cellIdsToDelete.insert(cellId);
}

void DataConverter::markDelParticle(uint64_t particleId)
{
	_particleIdsToDelete.insert(particleId);
}

void DataConverter::markModifyCell(CellChangeDescription const & clusterDesc)
{
    _cellToModifyById.insert_or_assign(clusterDesc.id, clusterDesc);
}

void DataConverter::markModifyParticle(ParticleChangeDescription const & particleDesc)
{
	_particleToModifyById.insert_or_assign(particleDesc.id, particleDesc);
}

void DataConverter::processDeletions()
{
	if (_cellIdsToDelete.empty() && _particleIdsToDelete.empty()) {
		return;
	}

	//delete cells
    int cellIndexCopyOffset = 0;
    std::unordered_map<int, int> newByOldCellIndex;
    for (int index = 0; index < *_dataTO.numCells; ++index) {
        CellAccessTO& cell = _dataTO.cells[index];
        uint64_t cellId = cell.id;
        if (_cellIdsToDelete.find(cellId) != _cellIdsToDelete.end()) {
            ++cellIndexCopyOffset;
        } else if (cellIndexCopyOffset > 0) {
            newByOldCellIndex.insert_or_assign(index, index - cellIndexCopyOffset);
            _dataTO.cells[index - cellIndexCopyOffset] = cell;
        }
    }
    *_dataTO.numCells -= cellIndexCopyOffset;

	//delete tokens
/*
	tokenIndexCopyOffset = 0;
	for (int tokenIndex = 0; tokenIndex < *_dataTO.numTokens; ++tokenIndex) {
		TokenAccessTO& token = _dataTO.tokens[tokenIndex];
		if (newByOldCellIndex.find(token.cellIndex) != newByOldCellIndex.end()) {
			token.cellIndex = newByOldCellIndex.at(token.cellIndex);
		}
		if (tokenIndicesToDelete.find(tokenIndex) != tokenIndicesToDelete.end()) {
			++tokenIndexCopyOffset;
		}
		else if (tokenIndexCopyOffset > 0) {
			_dataTO.tokens[tokenIndex - tokenIndexCopyOffset] = token;
		}
	}
	*_dataTO.numTokens -= tokenIndexCopyOffset;
*/

	//delete particles
	int particleIndexCopyOffset = 0;
	for (int index = 0; index < *_dataTO.numParticles; ++index) {
		ParticleAccessTO& particle = _dataTO.particles[index];
		uint64_t particleId = particle.id;
		if (_particleIdsToDelete.find(particleId) != _particleIdsToDelete.end()) {
			++particleIndexCopyOffset;
		}
		else if (particleIndexCopyOffset > 0) {
			_dataTO.particles[index - particleIndexCopyOffset] = particle;
		}
	}
	*_dataTO.numParticles -= particleIndexCopyOffset;

	//adjust cell connections
	for (int cellIndex = 0; cellIndex < *_dataTO.numCells; ++cellIndex) {
		CellAccessTO& cell = _dataTO.cells[cellIndex];
		for (int connectionIndex = 0; connectionIndex < cell.numConnections; ++connectionIndex) {
            auto it = newByOldCellIndex.find(cell.connections[connectionIndex].cellIndex);
			if (it != newByOldCellIndex.end()) {
				cell.connections[connectionIndex].cellIndex = it->second;
			}
		}
	}
}

void DataConverter::processModifications()
{
	//modify cells
	for (int cellIndex = 0; cellIndex < *_dataTO.numCells; ++cellIndex) {
		CellAccessTO& cell = _dataTO.cells[cellIndex];
		uint64_t cellId = cell.id;
		if (_cellToModifyById.find(cellId) != _cellToModifyById.end()) {
			applyChangeDescription(_cellToModifyById.at(cellId), cell);
		}
	}

	//modify tokens
/*
	std::unordered_map<uint64_t, vector<TokenAccessTO>> tokenTOsByCellId;
	for (int index = 0; index < *_dataTO.numTokens; ++index) {
		auto const& tokenTO = _dataTO.tokens[index];
		auto const& cellTO = _dataTO.cells[tokenTO.cellIndex];
		tokenTOsByCellId[cellTO.id].emplace_back(tokenTO);
	}
	*_dataTO.numTokens = 0;
	for (int clusterIndex = 0; clusterIndex < *_dataTO.numClusters; ++clusterIndex) {
		auto& clusterTO = _dataTO.clusters[clusterIndex];
		clusterTO.tokenStartIndex = *_dataTO.numTokens;
		clusterTO.numTokens = 0;
		for (int cellIndex = clusterTO.cellStartIndex; cellIndex < clusterTO.cellStartIndex + clusterTO.numCells; ++cellIndex) {
			auto const& cellTO = _dataTO.cells[cellIndex];
			if (_cellToModifyById.find(cellTO.id) != _cellToModifyById.end()) {
				auto const& cell = _cellToModifyById.at(cellTO.id);
				if (boost::optional<vector<TokenDescription>> const& tokens = cell.tokens.getOptionalValue()) {
					clusterTO.numTokens += tokens->size();
					for (int sourceTokenIndex = 0; sourceTokenIndex < tokens->size(); ++sourceTokenIndex) {
						int targetTokenIndex = (*_dataTO.numTokens)++;
                        if (targetTokenIndex >= _cudaConstants.MAX_TOKENS) {
                            throw BugReportException("Array size for tokens is chosen too small.");
                        }

						auto& targetToken = _dataTO.tokens[targetTokenIndex];
						auto const& sourceToken = tokens->at(sourceTokenIndex);
						targetToken.cellIndex = cellIndex;
						targetToken.energy = *sourceToken.energy;
						convertToArray(*sourceToken.data, targetToken.memory, _parameters.tokenMemorySize);
					}
				}
			}
			else if (tokenTOsByCellId.find(cellTO.id) != tokenTOsByCellId.end()){
				auto const& tokens = tokenTOsByCellId.at(cellTO.id);
				clusterTO.numTokens += tokens.size();
				for (int sourceTokenIndex = 0; sourceTokenIndex < tokens.size(); ++sourceTokenIndex) {
					int targetTokenIndex = (*_dataTO.numTokens)++;
                    if (targetTokenIndex >= _cudaConstants.MAX_TOKENS) {
                        throw BugReportException("Array size for tokens is chosen too small.");
                    }

					auto& targetToken = _dataTO.tokens[targetTokenIndex];
					auto const& sourceToken = tokens[sourceTokenIndex];
					targetToken = sourceToken;
				}
			}
		}
	}
*/

	//modify particles
	for (int index = 0; index < *_dataTO.numParticles; ++index) {
		ParticleAccessTO& particle = _dataTO.particles[index];
		uint64_t particleId = particle.id;
		if (_particleToModifyById.find(particleId) != _particleToModifyById.end()) {
			applyChangeDescription(_particleToModifyById.at(particleId), particle);
		}
	}
}

int DataConverter::convertStringAndReturnStringIndex(QString const& s)
{
    auto const result = *_dataTO.numStringBytes;
    auto const len = s.size();
    for (int i = 0; i < len; ++i) {
        _dataTO.stringBytes[result + i] = s.at(i).toLatin1();
    }
    (*_dataTO.numStringBytes) += len;
    return result;
}

void DataConverter::addCell(
    CellChangeDescription const& cellDesc,
    unordered_map<uint64_t, int>& cellIndexTOByIds)
{
	int cellIndex = (*_dataTO.numCells)++;
    if (cellIndex >= _cudaConstants.MAX_CELLS) {
        throw BugReportException("Array size for cells is chosen too small.");
    }
	CellAccessTO& cellTO = _dataTO.cells[cellIndex];
	cellTO.id = cellDesc.id == 0 ? _numberGen->getId() : cellDesc.id;
	cellTO.pos= { cellDesc.pos->x(), cellDesc.pos->y() };
    cellTO.vel = {cellDesc.vel->x(), cellDesc.vel->y()};
    cellTO.energy = *cellDesc.energy;
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
		cellTO.numConnections = cellDesc.connectingCells->size();
	}
	else {
		cellTO.numConnections = 0;
	}
    if (cellDesc.metadata.getOptionalValue()) {
        auto& metadataTO = cellTO.metadata;
        metadataTO.color = cellDesc.metadata->color;
        metadataTO.nameLen = cellDesc.metadata->name.size();
        if (metadataTO.nameLen > 0) {
            metadataTO.nameStringIndex = convertStringAndReturnStringIndex(cellDesc.metadata->name);
        }
        metadataTO.descriptionLen = cellDesc.metadata->description.size();
        if (metadataTO.descriptionLen > 0) {
            metadataTO.descriptionStringIndex = convertStringAndReturnStringIndex(cellDesc.metadata->description);
        }
        metadataTO.sourceCodeLen = cellDesc.metadata->computerSourcecode.size();
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
            tokenTO.energy = *tokenDesc.energy;
            tokenTO.cellIndex = cellIndex;
            convertToArray(*tokenDesc.data, tokenTO.memory, _parameters.tokenMemorySize);
        }
    }
/*
	if (cellDesc.tokens) {
		for (int i = 0; i < cellDesc.tokens->size(); ++i) {
			TokenDescription const& tokenDesc = cellDesc.tokens->at(i);
			int tokenIndex = (*_dataTO.numTokens)++;
			TokenAccessTO& tokenTO = _dataTO.tokens[tokenIndex];
			tokenTO.energy = *tokenDesc.energy;
			tokenTO.cellIndex = cellIndex;
			convertToArray(*tokenDesc.data, tokenTO.memory, _parameters.tokenMemorySize);
        }
	}
*/

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
            cellTO.connections[index].distance = connection.distance;
            cellTO.connections[index].angleFromPrevious = connection.angleFromPrevious;
            ++index;
		}
        cellTO.numConnections = index;
	}
}

namespace
{
	void convert(QVector2D const& input, float2& output)
	{
		output.x = input.x();
		output.y = input.y();
	}
}

void DataConverter::applyChangeDescription(ParticleChangeDescription const& particleChanges, ParticleAccessTO& particle)
{
	if (particleChanges.pos) {
		QVector2D newPos = particleChanges.pos.getValue();
		convert(newPos, particle.pos);
	}
	if (particleChanges.vel) {
		QVector2D newVel = particleChanges.vel.getValue();
		convert(newVel, particle.vel);
	}
	if (particleChanges.energy) {
		particle.energy = particleChanges.energy.getValue();
	}
    if (particleChanges.metadata) {
        particle.metadata.color = particleChanges.metadata->color;
    }
}

void DataConverter::applyChangeDescription(CellChangeDescription const& cellChanges, CellAccessTO& cellTO)
{
    if (cellChanges.pos) {
        QVector2D newAbsPos = cellChanges.pos.getValue();
        convert(newAbsPos, cellTO.pos);
    }
    if (cellChanges.maxConnections) {
        cellTO.maxConnections = cellChanges.maxConnections.getValue();
    }
    if (cellChanges.energy) {
        cellTO.energy = cellChanges.energy.getValue();
    }
    if (cellChanges.tokenBranchNumber) {
        cellTO.branchNumber = cellChanges.tokenBranchNumber.getValue();
    }
    if (cellChanges.cellFeatures) {
        auto cellFunction = *cellChanges.cellFeatures;
        cellTO.cellFunctionType = cellFunction.getType();
        cellTO.numStaticBytes = std::min(static_cast<int>(cellFunction.constData.size()), MAX_CELL_STATIC_BYTES);
        cellTO.numMutableBytes = std::min(static_cast<int>(cellFunction.volatileData.size()), MAX_CELL_MUTABLE_BYTES);
        convertToArray(cellFunction.constData, cellTO.staticData, MAX_CELL_STATIC_BYTES);
        convertToArray(cellFunction.volatileData, cellTO.mutableData, MAX_CELL_MUTABLE_BYTES);
    }
    if (cellChanges.metadata) {
        auto& metadataTO = cellTO.metadata;
        metadataTO.color = cellChanges.metadata->color;
        metadataTO.nameLen = cellChanges.metadata->name.size();
        if (metadataTO.nameLen > 0) {
            metadataTO.nameStringIndex = convertStringAndReturnStringIndex(cellChanges.metadata->name);
        }
        metadataTO.descriptionLen = cellChanges.metadata->description.size();
        if (metadataTO.descriptionLen > 0) {
            metadataTO.descriptionStringIndex = convertStringAndReturnStringIndex(cellChanges.metadata->description);
        }
        metadataTO.sourceCodeLen = cellChanges.metadata->computerSourcecode.size();
        if (metadataTO.sourceCodeLen > 0) {
            metadataTO.sourceCodeStringIndex = convertStringAndReturnStringIndex(cellChanges.metadata->computerSourcecode);
        }
    }
    if (cellChanges.tokenUsages) {
        cellTO.tokenUsages = cellChanges.tokenUsages.getValue();
    }
}

