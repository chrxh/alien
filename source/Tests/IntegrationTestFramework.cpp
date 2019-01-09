#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelCpu/UnitGrid.h"
#include "ModelCpu/Unit.h"
#include "ModelCpu/UnitContext.h"
#include "ModelCpu/MapCompartment.h"
#include "ModelCpu/UnitThreadControllerImpl.h"
#include "ModelCpu/UnitThread.h"
#include "ModelBasic/SimulationAccess.h"

#include "IntegrationTestFramework.h"

IntegrationTestFramework::IntegrationTestFramework(IntVector2D const& universeSize)
	: _universeSize(universeSize)
{
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	_basicFacade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
	_cpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	_gpuFacade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
	_symbols = _basicFacade->buildDefaultSymbolTable();
	_parameters = _basicFacade->buildDefaultSimulationParameters();
}

IntegrationTestFramework::~IntegrationTestFramework()
{
}

ClusterDescription IntegrationTestFramework::createSingleCellClusterWithCompleteData(uint64_t clusterId /*= 0*/, uint64_t cellId /*= 0*/) const
{
	QByteArray code("123123123");
	QByteArray cellMemory(_parameters->cellFunctionComputerCellMemorySize, 0);
	QByteArray tokenMemory(_parameters->tokenMemorySize, 0);
	cellMemory[1] = 'a';
	cellMemory[2] = 'b';
	tokenMemory[0] = 't';
	tokenMemory[3] = 's';
	CellMetadata cellMetadata;
	cellMetadata.color = 2;
	cellMetadata.name = "name1";
	cellMetadata.computerSourcecode = "code";
	cellMetadata.description = "desc";
	ClusterMetadata clusterMetadata;
	clusterMetadata.name = "name2";

	return ClusterDescription().addCell(
		CellDescription().setCellFeature(
			CellFeatureDescription().setType(Enums::CellFunction::COMPUTER).setConstData(code).setVolatileData(cellMemory)
		).setId(cellId).setPos({ 1, 2 }).setEnergy(56).setFlagTokenBlocked(true).setMaxConnections(3).setMetadata(cellMetadata)
		.setTokenBranchNumber(2).setTokens({
			TokenDescription().setData(tokenMemory).setEnergy(89)
	})
	).setId(clusterId).setPos({ 1, 2 }).setVel({ -1, 1 }).setAngle(23).setAngularVel(1.2).setMetadata(clusterMetadata);
}

ClusterDescription IntegrationTestFramework::createHorizontalCluster(int numCells, optional<QVector2D> const& centerPos,
	optional<QVector2D> const& centerVel, optional<double> const& angularVel) const
{
	QVector2D pos = centerPos ? * centerPos : QVector2D(_numberGen->getRandomReal(0, _universeSize.x), _numberGen->getRandomReal(0, _universeSize.y));
	QVector2D vel = centerVel ? *centerVel : QVector2D(_numberGen->getRandomReal(-1, 1), _numberGen->getRandomReal(-1, 1));

	ClusterDescription cluster;
	cluster.setId(_numberGen->getId()).setPos(pos).setVel(vel).setAngle(0).setAngularVel(angularVel.get_value_or(0));
	for (int j = 0; j < numCells; ++j) {
		cluster.addCell(
			CellDescription().setEnergy(_parameters->cellFunctionConstructorOffspringCellEnergy)
			.setPos(pos + QVector2D(-static_cast<float>(numCells - 1) / 2.0 + j, 0))
			.setMaxConnections(2).setId(_numberGen->getId()).setCellFeature(CellFeatureDescription())
		);
	}
	for (int j = 0; j < numCells; ++j) {
		list<uint64_t> connectingCells;
		if (j > 0) {
			connectingCells.emplace_back(cluster.cells->at(j - 1).id);
		}
		if (j < numCells - 1) {
			connectingCells.emplace_back(cluster.cells->at(j + 1).id);
		}
		cluster.cells->at(j).setConnectingCells(connectingCells);
	}
	return cluster;
}

ClusterDescription IntegrationTestFramework::createVerticalCluster(int numCells, optional<QVector2D> const & centerPos, optional<QVector2D> const & centerVel) const
{
	QVector2D pos = centerPos ? *centerPos : QVector2D(_numberGen->getRandomReal(0, _universeSize.x), _numberGen->getRandomReal(0, _universeSize.y));
	QVector2D vel = centerVel ? *centerVel : QVector2D(_numberGen->getRandomReal(-1, 1), _numberGen->getRandomReal(-1, 1));

	ClusterDescription cluster;
	cluster.setId(_numberGen->getId()).setPos(pos).setVel(vel).setAngle(0).setAngularVel(0);
	for (int j = 0; j < numCells; ++j) {
		cluster.addCell(
			CellDescription().setEnergy(_parameters->cellFunctionConstructorOffspringCellEnergy)
			.setPos(pos + QVector2D(0, -static_cast<float>(numCells - 1) / 2.0 + j))
			.setMaxConnections(2).setId(_numberGen->getId()).setCellFeature(CellFeatureDescription())
		);
	}
	for (int j = 0; j < numCells; ++j) {
		list<uint64_t> connectingCells;
		if (j > 0) {
			connectingCells.emplace_back(cluster.cells->at(j - 1).id);
		}
		if (j < numCells - 1) {
			connectingCells.emplace_back(cluster.cells->at(j + 1).id);
		}
		cluster.cells->at(j).setConnectingCells(connectingCells);
	}
	return cluster;
}

ParticleDescription IntegrationTestFramework::createParticle() const
{
	QVector2D pos(_numberGen->getRandomReal(0, _universeSize.x), _numberGen->getRandomReal(0, _universeSize.y));
	QVector2D vel(_numberGen->getRandomReal(-0.5, 0.5), _numberGen->getRandomReal(-0.5, 0.5));
	return ParticleDescription().setEnergy(_parameters->cellMinEnergy).setPos(pos).setVel(vel).setId(_numberGen->getId());
}


template<>
bool isCompatible<QVector2D>(QVector2D vec1, QVector2D vec2)
{
	return std::abs(vec1.x() - vec2.x()) < FLOATINGPOINT_MEDIUM_PRECISION
		&& std::abs(vec1.y() - vec2.y()) < FLOATINGPOINT_MEDIUM_PRECISION;
}

template<>
bool isCompatible<double>(double a, double b)
{
	return std::abs(a - b) < FLOATINGPOINT_MEDIUM_PRECISION;
}

template<>
bool isCompatible<TokenDescription>(TokenDescription token1, TokenDescription token2)
{
	return isCompatible(token1.energy, token2.energy)
		&& isCompatible(token1.data->mid(1), token2.data->mid(1));	//do not compare first byte (overriden branch number)
}

namespace
{
	void removeZerosAtEnd(QByteArray& data)
	{
		while (true) {
			if (data.isEmpty()) {
				break;
			}
			if (data.at(data.size() - 1) == 0) {
				data.chop(1);
			}
			else {
				break;
			}
		}
	}
}

template<>
bool isCompatible<CellFeatureDescription>(CellFeatureDescription feature1, CellFeatureDescription feature2)
{
	removeZerosAtEnd(feature1.volatileData);
	removeZerosAtEnd(feature2.volatileData);
	return isCompatible(feature1.type, feature2.type)
		&& isCompatible(feature1.constData, feature2.constData)
		&& isCompatible(feature1.volatileData, feature2.volatileData)
		;
}

template<>
bool isCompatible<CellDescription>(CellDescription cell1, CellDescription cell2)
{
	return isCompatible(cell1.tokenBlocked, cell2.tokenBlocked)
		&& isCompatible(cell1.pos, cell2.pos)
		&& isCompatible(cell1.energy, cell2.energy)
		&& isCompatible(cell1.maxConnections, cell2.maxConnections)
		&& isCompatible(cell1.connectingCells, cell2.connectingCells)
		&& isCompatible(cell1.tokenBranchNumber, cell2.tokenBranchNumber)
		&& isCompatible(cell1.metadata, cell2.metadata)
		&& isCompatible(cell1.cellFeature, cell2.cellFeature)
		&& isCompatible(cell1.tokens, cell2.tokens)
		;
}

template<>
bool isCompatible<ClusterDescription>(ClusterDescription cluster1, ClusterDescription cluster2)
{
	return isCompatible(cluster1.pos, cluster2.pos)
		&& isCompatible(cluster1.vel, cluster2.vel)
		&& isCompatible(cluster1.angle, cluster2.angle)
		&& isCompatible(cluster1.angularVel, cluster2.angularVel)
		&& isCompatible(cluster1.metadata, cluster2.metadata)
		&& isCompatible(cluster1.cells, cluster2.cells);
}

template<>
bool isCompatible<ParticleDescription>(ParticleDescription particle1, ParticleDescription particle2)
{
	return isCompatible(particle1.pos, particle2.pos)
		&& isCompatible(particle1.vel, particle2.vel)
		&& isCompatible(particle1.energy, particle2.energy)
		&& isCompatible(particle1.metadata, particle2.metadata);
}

namespace
{
	void sortById(DataDescription& data)
	{
		if (data.clusters) {
			std::sort(data.clusters->begin(), data.clusters->end(), [](auto const &cluster1, auto const &cluster2) {
				return cluster1.id <= cluster2.id;
			});
			for (auto& cluster : *data.clusters) {
				std::sort(cluster.cells->begin(), cluster.cells->end(), [](auto const &cell1, auto const &cell2) {
					return cell1.id <= cell2.id;
				});
			}
		}
		if (data.particles) {
			std::sort(data.particles->begin(), data.particles->end(), [](auto const &particle1, auto const &particle2) {
				return particle1.id <= particle2.id;
			});
		}
	}
}

template<>
bool isCompatible<DataDescription>(DataDescription data1, DataDescription data2)
{
	sortById(data1);
	sortById(data2);
	return isCompatible(data1.clusters, data2.clusters)
		&& isCompatible(data1.particles, data2.particles);
}
