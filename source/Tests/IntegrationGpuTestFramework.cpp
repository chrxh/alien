#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"

#include "IntegrationGpuTestFramework.h"

IntegrationGpuTestFramework::IntegrationGpuTestFramework(IntVector2D const& universeSize)
	: IntegrationTestFramework(universeSize)
{
    ModelGpuData data;
    data.setNumThreadsPerBlock(64);
    data.setNumBlocks(64);

    data.setNumClusterPointerArrays(1);
    data.setMaxClusters(500000);
    data.setMaxCells(2000000);
    data.setMaxParticles(2000000);
    data.setMaxTokens(500000);
    data.setMaxCellPointers(2000000 * 10);
    data.setMaxClusterPointers(500000 * 10);
    data.setMaxParticlePointers(2000000 * 10);
    data.setMaxTokenPointers(500000 * 10);

    data.setRandomNumberBlockSize(31231257);
    data.setProtectionTimesteps(14);

	_controller = _gpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, data, 0);
	_context = _controller->getContext();
	_spaceProp = _context->getSpaceProperties();
	_access = _gpuFacade->buildSimulationAccess();
	_parameters = _context->getSimulationParameters();
    _numberGen = _context->getNumberGenerator();
	_access->init(_controller);

    _descHelper = _basicFacade->buildDescriptionHelper();
    _descHelper->init(_context);
}

IntegrationGpuTestFramework::~IntegrationGpuTestFramework()
{
	delete _access;
	delete _controller;
    delete _descHelper;
}

void IntegrationGpuTestFramework::checkEnergy(DataDescription const& origData, DataDescription const& newData) const
{
	auto energyBefore = calcEnergy(origData);
	auto energyAfter = calcEnergy(newData);

	EXPECT_TRUE(isCompatible(energyBefore, energyAfter));
}

void IntegrationGpuTestFramework::checkDistancesToConnectingCells(DataDescription const & data) const
{
	if (!data.clusters) {
		return;
	}
	auto cellMaxDistance = _parameters.cellMaxDistance + FLOATINGPOINT_MEDIUM_PRECISION;
	auto cellByCellId = IntegrationTestHelper::getCellByCellId(data);
	for (ClusterDescription const& cluster : *data.clusters) {
		for (CellDescription const& cell : *cluster.cells) {
			if (!cell.connectingCells) {
				continue;
			}
			for (auto const& connectingCellId : *cell.connectingCells) {
				CellDescription const& connectingCell = cellByCellId.at(connectingCellId);
				ASSERT_GE(cellMaxDistance, (*connectingCell.pos - *cell.pos).length());
			}
		}
	}
}

void IntegrationGpuTestFramework::checkKineticEnergy(DataDescription const & origData, DataDescription const & newData) const
{
	auto energyBefore = calcKineticEnergy(origData);
	auto energyAfter = calcKineticEnergy(newData);

	EXPECT_TRUE(isCompatible(energyBefore, energyAfter));
}

Physics::Velocities IntegrationGpuTestFramework::calcVelocitiesOfClusterPart(ClusterDescription const& cluster, set<uint64_t> const& cellIds) const
{
	CHECK(!cellIds.empty());
	vector<QVector2D> relPositionOfMasses;
	for (CellDescription const& cell : *cluster.cells) {
		if (cellIds.find(cell.id) != cellIds.end()) {
			relPositionOfMasses.emplace_back(*cell.pos - *cluster.pos);
		}
	}
	return Physics::velocitiesOfCenter({ *cluster.vel, *cluster.angularVel }, relPositionOfMasses);
}

Physics::Velocities IntegrationGpuTestFramework::calcVelocitiesOfFusion(ClusterDescription const & cluster1, ClusterDescription const & cluster2) const
{
	vector<QVector2D> relPositionOfMasses1;
	std::transform(
		cluster1.cells->begin(), cluster1.cells->end(),
		std::inserter(relPositionOfMasses1, relPositionOfMasses1.begin()),
		[cluster1](auto const& cell) { return *cell.pos - *cluster1.pos; });

	vector<QVector2D> relPositionOfMasses2;
	std::transform(
		cluster2.cells->begin(), cluster2.cells->end(),
		std::inserter(relPositionOfMasses2, relPositionOfMasses2.begin()),
		[cluster2](auto const& cell) { return *cell.pos - *cluster2.pos; });

	return Physics::fusion(
		*cluster1.pos, { *cluster1.vel, *cluster1.angularVel }, relPositionOfMasses1,
		*cluster2.pos, { *cluster2.vel, *cluster2.angularVel }, relPositionOfMasses2);
}

double IntegrationGpuTestFramework::calcEnergy(DataDescription const & data) const
{
	auto result = 0.0;
	if (data.clusters) {
		for (auto const& cluster : *data.clusters) {
			result += calcEnergy(cluster);
		}
	}
	if (data.particles) {
		for (auto const& particle : *data.particles) {
			result += *particle.energy;
		}
	}

	return result;
}

double IntegrationGpuTestFramework::calcEnergy(ClusterDescription const & cluster) const
{
	auto result = calcKineticEnergy(cluster);
	if (cluster.cells) {
        for (CellDescription const& cell : *cluster.cells) {
            result += *cell.energy;
            if (cell.tokens) {
                for (TokenDescription const& token : *cell.tokens) {
                    result += *token.energy;
                }
            }
        }
	}
	return result;
}

double IntegrationGpuTestFramework::calcKineticEnergy(DataDescription const & data) const
{
	auto result = 0.0;
	if (data.clusters) {
		for (auto const& cluster : *data.clusters) {
			result += calcKineticEnergy(cluster);
		}
	}
	return result;
}

double IntegrationGpuTestFramework::calcKineticEnergy(ClusterDescription const& cluster) const
{
	auto mass = cluster.cells->size();
	auto vel = *cluster.vel;

	vector<QVector2D> relPositions;
	std::transform(cluster.cells->begin(), cluster.cells->end(), std::inserter(relPositions, relPositions.begin()),
		[&cluster](CellDescription const& cell) { return *cell.pos - *cluster.pos; });

	auto angularMass = Physics::angularMass(relPositions);
	auto angularVel = *cluster.angularVel;
	return Physics::kineticEnergy(mass, vel, angularMass, angularVel);
}

void IntegrationGpuTestFramework::setMaxConnections(ClusterDescription& cluster, int maxConnections) const
{
	for (CellDescription& cell : *cluster.cells) {
		cell.setMaxConnections(maxConnections);
	}
}

void IntegrationGpuTestFramework::setCenterPos(ClusterDescription& cluster, QVector2D const& centerPos) const
{
    auto diff = centerPos - *cluster.pos;
    cluster.pos = centerPos;
    for (auto& cell : *cluster.cells) {
        cell.pos = *cell.pos + diff;
    }
}
