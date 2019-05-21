#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"

#include "SimulationGpuTestFramework.h"

SimulationGpuTestFramework::SimulationGpuTestFramework(IntVector2D const& universeSize)
	: IntegrationTestFramework(universeSize)
{
	_controller = _gpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelGpuData(), 0);
	_context = _controller->getContext();
	_spaceProp = _context->getSpaceProperties();
	_access = _gpuFacade->buildSimulationAccess();
	_parameters = _context->getSimulationParameters();
	_numberGen = _context->getNumberGenerator();
	_access->init(_controller);
}

SimulationGpuTestFramework::~SimulationGpuTestFramework()
{
	delete _access;
	delete _controller;
}

void SimulationGpuTestFramework::checkEnergy(DataDescription const& origData, DataDescription const& newData) const
{
	auto energyBefore = calcEnergy(origData);
	auto energyAfter = calcEnergy(newData);

	EXPECT_TRUE(isCompatible(energyBefore, energyAfter));
}

void SimulationGpuTestFramework::checkDistancesToConnectingCells(DataDescription const & data) const
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

void SimulationGpuTestFramework::checkKineticEnergy(DataDescription const & origData, DataDescription const & newData) const
{
	auto energyBefore = calcKineticEnergy(origData);
	auto energyAfter = calcKineticEnergy(newData);

	EXPECT_TRUE(isCompatible(energyBefore, energyAfter));
}

Physics::Velocities SimulationGpuTestFramework::calcVelocitiesOfClusterPart(ClusterDescription const& cluster, set<uint64_t> const& cellIds) const
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

Physics::Velocities SimulationGpuTestFramework::calcVelocitiesOfFusion(ClusterDescription const & cluster1, ClusterDescription const & cluster2) const
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

double SimulationGpuTestFramework::calcEnergy(DataDescription const & data) const
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

double SimulationGpuTestFramework::calcEnergy(ClusterDescription const & cluster) const
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

double SimulationGpuTestFramework::calcKineticEnergy(DataDescription const & data) const
{
	auto result = 0.0;
	if (data.clusters) {
		for (auto const& cluster : *data.clusters) {
			result += calcKineticEnergy(cluster);
		}
	}
	return result;
}

double SimulationGpuTestFramework::calcKineticEnergy(ClusterDescription const& cluster) const
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

void SimulationGpuTestFramework::setMaxConnections(ClusterDescription& cluster, int maxConnections) const
{
	for (CellDescription& cell : *cluster.cells) {
		cell.setMaxConnections(maxConnections);
	}
}

