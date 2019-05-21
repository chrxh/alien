#include "Base/NumberGenerator.h"
#include "ModelBasic/Descriptions.h"
#include "ModelBasic/ChangeDescriptions.h"
#include "ModelBasic/Physics.h"

#include "DataConverter.h"

DataConverter::DataConverter(DataAccessTO& dataTO, NumberGenerator* numberGen, SimulationParameters const& parameters)
	: _dataTO(dataTO), _numberGen(numberGen), _parameters(parameters)
{}

void DataConverter::updateData(DataChangeDescription const & data)
{
	for (auto const& cluster : data.clusters) {
		if (cluster.isDeleted()) {
			markDelCluster(cluster.getValue().id);
		}
		if (cluster.isModified()) {
			markModifyCluster(cluster.getValue());
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

	for (auto const& cluster : data.clusters) {
		if (cluster.isAdded()) {
			addCluster(cluster.getValue());
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
            result[i] = data[i];
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
	list<uint64_t> connectingCellIds;
	unordered_map<int, int> cellIndexByCellTOIndex;
	unordered_map<int, int> clusterIndexByCellTOIndex;
	for (int i = 0; i < *_dataTO.numClusters; ++i) {
		ClusterAccessTO const& cluster = _dataTO.clusters[i];
		auto clusterDesc = ClusterDescription().setId(cluster.id).setPos({ cluster.pos.x, cluster.pos.y })
			.setVel({ cluster.vel.x, cluster.vel.y })
			.setAngle(cluster.angle)
			.setAngularVel(cluster.angularVel).setMetadata(ClusterMetadata());

		for (int j = 0; j < cluster.numCells; ++j) {
			CellAccessTO const& cellTO = _dataTO.cells[cluster.cellStartIndex + j];
			auto pos = cellTO.pos;
			auto id = cellTO.id;
			connectingCellIds.clear();
			for (int i = 0; i < cellTO.numConnections; ++i) {
				connectingCellIds.emplace_back(_dataTO.cells[cellTO.connectionIndices[i]].id);
			}
			cellIndexByCellTOIndex.insert_or_assign(cluster.cellStartIndex + j, j);
			clusterIndexByCellTOIndex.insert_or_assign(cluster.cellStartIndex + j, i);

            auto feature = CellFeatureDescription().setType(static_cast<Enums::CellFunction::Type>(cellTO.cellFunctionType))
                .setConstData(convertToQByteArray(cellTO.staticData, cellTO.numStaticBytes)).setVolatileData(convertToQByteArray(cellTO.mutableData, cellTO.numMutableBytes));

            clusterDesc.addCell(
				CellDescription().setPos({ pos.x, pos.y }).setMetadata(CellMetadata())
				.setEnergy(cellTO.energy).setId(id).setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::COMPUTER))
				.setConnectingCells(connectingCellIds).setMaxConnections(cellTO.maxConnections).setFlagTokenBlocked(false)
				.setTokenBranchNumber(0).setMetadata(CellMetadata()).setTokens(vector<TokenDescription>{}).setTokenBranchNumber(cellTO.branchNumber)
				.setFlagTokenBlocked(cellTO.tokenBlocked).setCellFeature(feature)
			);
		}
		result.addCluster(clusterDesc);
	}

	for (int i = 0; i < *_dataTO.numParticles; ++i) {
		ParticleAccessTO const& particle = _dataTO.particles[i];
		result.addParticle(ParticleDescription().setId(particle.id).setPos({ particle.pos.x, particle.pos.y })
			.setVel({ particle.vel.x, particle.vel.y }).setEnergy(particle.energy));
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
}

void DataConverter::addCluster(ClusterDescription const& clusterDesc)
{
	if (!clusterDesc.cells) {
		return;
	}

	ClusterAccessTO& clusterTO = _dataTO.clusters[(*_dataTO.numClusters)++];
	clusterTO.id = clusterDesc.id == 0 ? _numberGen->getId() : clusterDesc.id;
	QVector2D clusterPos = clusterDesc.pos ? clusterPos = *clusterDesc.pos : clusterPos = clusterDesc.getClusterPosFromCells();
	clusterTO.pos = { clusterPos.x(), clusterPos.y() };
	clusterTO.vel = { clusterDesc.vel->x(), clusterDesc.vel->y() };
	clusterTO.angle = *clusterDesc.angle;
	clusterTO.angularVel = *clusterDesc.angularVel;
	clusterTO.numCells = clusterDesc.cells ? clusterDesc.cells->size() : 0;
	clusterTO.numTokens = 0;	//will be incremented in addCell
	clusterTO.tokenStartIndex = *_dataTO.numTokens;
	unordered_map<uint64_t, int> cellIndexByIds;
	bool firstIndex = true;
	for (CellDescription const& cellDesc : *clusterDesc.cells) {
		addCell(cellDesc, clusterDesc, clusterTO, cellIndexByIds);
		if (firstIndex) {
			clusterTO.cellStartIndex = cellIndexByIds.begin()->second;
			firstIndex = false;
		}
	}
	for (CellDescription const& cellDesc : *clusterDesc.cells) {
		if (cellDesc.id != 0) {
			setConnections(cellDesc, _dataTO.cells[cellIndexByIds.at(cellDesc.id)], cellIndexByIds);
		}
	}
}

void DataConverter::addParticle(ParticleDescription const & particleDesc)
{
	ParticleAccessTO& particleTO = _dataTO.particles[(*_dataTO.numParticles)++];
	particleTO.id = particleDesc.id == 0 ? _numberGen->getId() : particleDesc.id;
	particleTO.pos = { particleDesc.pos->x(), particleDesc.pos->y() };
	particleTO.vel = { particleDesc.vel->x(), particleDesc.vel->y() };
	particleTO.energy = *particleDesc.energy;
}

void DataConverter::markDelCluster(uint64_t clusterId)
{
	_clusterIdsToDelete.insert(clusterId);
}

void DataConverter::markDelParticle(uint64_t particleId)
{
	_particleIdsToDelete.insert(particleId);
}

void DataConverter::markModifyCluster(ClusterChangeDescription const & clusterDesc)
{
	_clusterToModifyById.insert_or_assign(clusterDesc.id, clusterDesc);
	for (auto const& cellTracker : clusterDesc.cells) {
		if (cellTracker.isModified()) {
			_cellToModifyById.insert_or_assign(cellTracker->id, cellTracker.getValue());
		}
	}
}

void DataConverter::markModifyParticle(ParticleChangeDescription const & particleDesc)
{
	_particleToModifyById.insert_or_assign(particleDesc.id, particleDesc);
}


//deleting specific cells from clusters is not supported
void DataConverter::processDeletions()
{
	if (_clusterIdsToDelete.empty() && _particleIdsToDelete.empty()) {
		return;
	}

	//delete clusters
	std::unordered_set<int> cellIndicesToDelete;
	std::unordered_set<int> tokenIndicesToDelete;
	int clusterIndexCopyOffset = 0;
	int tokenIndexCopyOffset = 0;
	for (int clusterIndex = 0; clusterIndex < *_dataTO.numClusters; ++clusterIndex) {
		ClusterAccessTO& cluster = _dataTO.clusters[clusterIndex];
		uint64_t clusterId = cluster.id;
		if (_clusterIdsToDelete.find(clusterId) != _clusterIdsToDelete.end()) {
			++clusterIndexCopyOffset;
			tokenIndexCopyOffset += cluster.numTokens;
			for (int cellIndex = 0; cellIndex < cluster.numCells; ++cellIndex) {
				cellIndicesToDelete.insert(cluster.cellStartIndex + cellIndex);
			}
			for (int tokenIndex = 0; tokenIndex < cluster.numTokens; ++tokenIndex) {
				tokenIndicesToDelete.insert(cluster.tokenStartIndex + tokenIndex);
			}
		}
		else if (clusterIndexCopyOffset > 0) {
			_dataTO.clusters[clusterIndex - clusterIndexCopyOffset] = cluster;
			if (tokenIndexCopyOffset > 0) {
				cluster.tokenStartIndex -= tokenIndexCopyOffset;
			}
		}
	}
	*_dataTO.numClusters -= clusterIndexCopyOffset;

	//delete cells
	int cellIndexCopyOffset = 0;
	std::unordered_map<int, int> newByOldCellIndex;
	for (int cellIndex = 0; cellIndex < *_dataTO.numCells; ++cellIndex) {
		CellAccessTO& cell = _dataTO.cells[cellIndex];
		uint64_t cellId = cell.id;
		if (cellIndicesToDelete.find(cellIndex) != cellIndicesToDelete.end()) {
			++cellIndexCopyOffset;
		}
		else if (cellIndexCopyOffset > 0) {
			newByOldCellIndex.insert_or_assign(cellIndex, cellIndex - cellIndexCopyOffset);
			_dataTO.cells[cellIndex - cellIndexCopyOffset] = cell;
		}
	}
	*_dataTO.numCells -= cellIndexCopyOffset;

	//delete tokens
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

	//delete and modify particles
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

	//adjust cell and cluster pointers
	for (int clusterIndex = 0; clusterIndex < *_dataTO.numClusters; ++clusterIndex) {
		ClusterAccessTO& cluster = _dataTO.clusters[clusterIndex];
		auto it = newByOldCellIndex.find(cluster.cellStartIndex);
		if (it != newByOldCellIndex.end()) {
			cluster.cellStartIndex = it->second;
		}
	}
	for (int cellIndex = 0; cellIndex < *_dataTO.numCells; ++cellIndex) {
		CellAccessTO& cell = _dataTO.cells[cellIndex];
		for (int connectionIndex = 0; connectionIndex < cell.numConnections; ++connectionIndex) {
			auto it = newByOldCellIndex.find(cell.connectionIndices[connectionIndex]);
			if (it != newByOldCellIndex.end()) {
				cell.connectionIndices[connectionIndex] = it->second;
			}
		}
	}
}

void DataConverter::processModifications()
{
	//modify clusters
	for (int clusterIndex = 0; clusterIndex < *_dataTO.numClusters; ++clusterIndex) {
		ClusterAccessTO& cluster = _dataTO.clusters[clusterIndex];
		uint64_t clusterId = cluster.id;
		if (_clusterToModifyById.find(clusterId) != _clusterToModifyById.end()) {
			applyChangeDescription(_clusterToModifyById.at(clusterId), cluster);
		}
	}

	//modify cells
	for (int cellIndex = 0; cellIndex < *_dataTO.numCells; ++cellIndex) {
		CellAccessTO& cell = _dataTO.cells[cellIndex];
		uint64_t cellId = cell.id;
		if (_cellToModifyById.find(cellId) != _cellToModifyById.end()) {
			applyChangeDescription(_cellToModifyById.at(cellId), cell);
		}
	}

	//modify tokens
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
				if (auto const& tokens = cell.tokens) {
					clusterTO.numTokens += tokens->size();
					for (int sourceTokenIndex = 0; sourceTokenIndex < tokens->size(); ++sourceTokenIndex) {
						int targetTokenIndex = (*_dataTO.numTokens)++;
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
					auto& targetToken = _dataTO.tokens[targetTokenIndex];
					auto const& sourceToken = tokens[sourceTokenIndex];
					targetToken = sourceToken;
				}
			}
		}
	}

	//modify particles
	for (int index = 0; index < *_dataTO.numParticles; ++index) {
		ParticleAccessTO& particle = _dataTO.particles[index];
		uint64_t particleId = particle.id;
		if (_particleToModifyById.find(particleId) != _particleToModifyById.end()) {
			applyChangeDescription(_particleToModifyById.at(particleId), particle);
		}
	}
}

void DataConverter::addCell(CellDescription const& cellDesc, ClusterDescription const& cluster, ClusterAccessTO& clusterTO
	, unordered_map<uint64_t, int>& cellIndexTOByIds)
{
	int cellIndex = (*_dataTO.numCells)++;
	CellAccessTO& cellTO = _dataTO.cells[cellIndex];
	cellTO.id = cellDesc.id == 0 ? _numberGen->getId() : cellDesc.id;
	cellTO.pos= { cellDesc.pos->x(), cellDesc.pos->y() };
	cellTO.energy = *cellDesc.energy;
	cellTO.maxConnections = *cellDesc.maxConnections;
	cellTO.branchNumber = cellDesc.tokenBranchNumber.get_value_or(0);
	cellTO.tokenBlocked = cellDesc.tokenBlocked.get_value_or(false);
    auto const& cellFunction = cellDesc.cellFeature.get_value_or(CellFeatureDescription());
    cellTO.cellFunctionType = static_cast<int>(cellFunction.type);
    cellTO.numStaticBytes = std::min(cellFunction.constData.size(), MAX_CELL_STATIC_BYTES);
    cellTO.numMutableBytes = std::min(cellFunction.volatileData.size(), MAX_CELL_MUTABLE_BYTES);
    convertToArray(cellFunction.constData, cellTO.staticData, MAX_CELL_STATIC_BYTES);
    convertToArray(cellFunction.volatileData, cellTO.mutableData, MAX_CELL_MUTABLE_BYTES);
    if (cellDesc.connectingCells) {
		cellTO.numConnections = cellDesc.connectingCells->size();
	}
	else {
		cellTO.numConnections = 0;
	}

	if (cellDesc.tokens) {
		clusterTO.numTokens += cellDesc.tokens->size();
		for (int i = 0; i < cellDesc.tokens->size(); ++i) {
			TokenDescription const& tokenDesc = cellDesc.tokens->at(i);
			int tokenIndex = (*_dataTO.numTokens)++;
			TokenAccessTO& tokenTO = _dataTO.tokens[tokenIndex];
			tokenTO.energy = *tokenDesc.energy;
			tokenTO.cellIndex = cellIndex;
			convertToArray(*tokenDesc.data, tokenTO.memory, _parameters.tokenMemorySize);
            tokenTO.memory[0] = cellTO.branchNumber % _parameters.cellMaxTokenBranchNumber;
        }
	}

	cellIndexTOByIds.insert_or_assign(cellTO.id, cellIndex);
}

void DataConverter::setConnections(
    CellDescription const& cellToAdd, CellAccessTO& cellTO, unordered_map<uint64_t, int> const& cellIndexByIds)
{
	int index = 0;
	if (cellToAdd.connectingCells) {
		for (uint64_t connectingCellId : *cellToAdd.connectingCells) {
			cellTO.connectionIndices[index] = cellIndexByIds.at(connectingCellId);
			++index;
		}
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
}

void DataConverter::applyChangeDescription(ClusterChangeDescription const& clusterChanges, ClusterAccessTO& cluster)
{
	if (clusterChanges.pos) {
		QVector2D newPos = clusterChanges.pos.getValue();
		convert(newPos, cluster.pos);
	}
	if (clusterChanges.vel) {
		QVector2D newVel = clusterChanges.vel.getValue();
		convert(newVel, cluster.vel);
	}
	if (clusterChanges.angle) {
		cluster.angle = clusterChanges.angle.getValue();
	}
	if (clusterChanges.angularVel) {
		cluster.angularVel = clusterChanges.angularVel.getValue();
	}
}

void DataConverter::applyChangeDescription(CellChangeDescription const& cellChanges, CellAccessTO& cellTO)
{
	if (cellChanges.pos) {
		QVector2D newAbsPos = cellChanges.pos.getValue();
		convert(newAbsPos, cellTO.pos);
	}
	if (cellChanges.energy) {
		cellTO.energy = cellChanges.energy.getValue();
	}
	if (cellChanges.tokenBranchNumber) {
		cellTO.branchNumber = cellChanges.tokenBranchNumber.getValue();
	}
    if (cellChanges.cellFeatures) {
        auto cellFunction = *cellChanges.cellFeatures;
        cellTO.cellFunctionType = static_cast<int>(cellFunction.type);
        cellTO.numStaticBytes = std::min(cellFunction.constData.size(), MAX_CELL_STATIC_BYTES);
        cellTO.numMutableBytes = std::min(cellFunction.volatileData.size(), MAX_CELL_MUTABLE_BYTES);
        convertToArray(cellFunction.constData, cellTO.staticData, MAX_CELL_STATIC_BYTES);
        convertToArray(cellFunction.volatileData, cellTO.mutableData, MAX_CELL_MUTABLE_BYTES);
    }
}

