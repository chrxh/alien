#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Model/ModelBuilderFacade.h"
#include "Model/Settings.h"
#include "Model/SimulationController.h"
#include "Model/Context/SimulationContext.h"
#include "Model/Context/SimulationParameters.h"
#include "Model/AccessPorts/SimulationAccess.h"

#include "tests/Predicates.h"

class DataDescriptionTransferTest : public ::testing::Test
{
public:
	DataDescriptionTransferTest();
	~DataDescriptionTransferTest();

protected:
	ClusterDescription createClusterDescription(int numCells) const;

	ModelBuilderFacade* _facade = nullptr;
	SimulationController* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SimulationParameters* _parameters = nullptr;
	NumberGenerator* _numberGen = nullptr;
	SimulationAccess* _access = nullptr;
	IntVector2D _gridSize{ 6, 6 };
	IntVector2D _universeSize{ 600, 300 };
};

DataDescriptionTransferTest::DataDescriptionTransferTest()
{
	_facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto symbols = _facade->buildDefaultSymbolTable();
	_parameters = _facade->buildDefaultSimulationParameters();
	_controller = _facade->buildSimulationController(1, _gridSize, _universeSize, symbols, _parameters);
	_context = static_cast<SimulationContext*>(_controller->getContext());
	_numberGen = factory->buildRandomNumberGenerator();
	_numberGen->init(123123, 0);

	_access = _facade->buildSimulationAccess(_context);
}

DataDescriptionTransferTest::~DataDescriptionTransferTest()
{
	delete _access;
	delete _controller;
	delete _numberGen;
}

ClusterDescription DataDescriptionTransferTest::createClusterDescription(int numCells) const
{
	ClusterDescription cluster;
	QVector2D pos(_numberGen->getRandomReal(0, 499), _numberGen->getRandomReal(0, 299));
	cluster.setId(_numberGen->getTag()).setPos(pos).setVel(QVector2D(_numberGen->getRandomReal(-1, 1), _numberGen->getRandomReal(-1, 1)));
	for (int j = 0; j < numCells; ++j) {
		cluster.addCell(
			CellDescription().setEnergy(_parameters->cellCreationEnergy).setPos(pos + QVector2D(-static_cast<float>(numCells - 1) / 2.0 + j, 0))
			.setMaxConnections(2).setId(_numberGen->getTag())
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

namespace
{
	template<typename T>
	bool isCompatible(T const& a, T const& b)
	{
		return a == b;
	}

	template<>
	bool isCompatible<QVector2D>(QVector2D const& vec1, QVector2D const& vec2)
	{
		return std::abs(vec1.x() - vec2.x()) < FLOATINGPOINT_MEDIUM_PRECISION
			&& std::abs(vec1.y() - vec2.y()) < FLOATINGPOINT_MEDIUM_PRECISION;
	}

	template<>
	bool isCompatible<double>(double const& a, double const& b)
	{
		return std::abs(a - b) < FLOATINGPOINT_MEDIUM_PRECISION;
	}

	template<typename T>
	bool isCompatible(optional<T> const& a, optional<T> const& b)
	{
		if (!a || !b) {
			return true;
		}
		return isCompatible(*a, *b);
	}

	template<typename T>
	bool isCompatible(vector<T> const& a, vector<T> const& b)
	{
		if (a.size() != b.size()) {
			false;
		}
		for (int i = 0; i < a.size(); ++i) {
			if (!isCompatible(a.at(i), b.at(i))) {
				return false;
			}
		}
		return true;
	}

	template<>
	bool isCompatible<TokenDescription>(TokenDescription const & token1, TokenDescription const & token2)
	{
		return isCompatible(token1.energy, token2.energy)
			&& isCompatible(token1.data, token2.data);
	}

	template<>
	bool isCompatible<CellDescription>(CellDescription const & cell1, CellDescription const & cell2)
	{
		return isCompatible(cell1.pos, cell2.pos)
			&& isCompatible(cell1.energy, cell2.energy)
			&& isCompatible(cell1.maxConnections, cell2.maxConnections)
			&& isCompatible(cell1.connectingCells, cell2.connectingCells)
			&& isCompatible(cell1.tokenBlocked, cell2.tokenBlocked)
			&& isCompatible(cell1.tokenBranchNumber, cell2.tokenBranchNumber)
			&& isCompatible(cell1.metadata, cell2.metadata)
			&& isCompatible(cell1.cellFunction, cell2.cellFunction)
			&& isCompatible(cell1.tokens, cell2.tokens)
			;
	}

	template<>
	bool isCompatible<ClusterDescription>(ClusterDescription const & cluster1, ClusterDescription const & cluster2)
	{
		return isCompatible(cluster1.pos, cluster2.pos)
			&& isCompatible(cluster1.vel, cluster2.vel)
			&& isCompatible(cluster1.angle, cluster2.angle)
			&& isCompatible(cluster1.angularVel, cluster2.angularVel)
			&& isCompatible(cluster1.metadata, cluster2.metadata)
			&& isCompatible(cluster1.cells, cluster2.cells);
	}

	template<>
	bool isCompatible<ParticleDescription>(ParticleDescription const & particle1, ParticleDescription const & particle2)
	{
		return isCompatible(particle1.pos, particle2.pos)
			&& isCompatible(particle1.vel, particle2.vel)
			&& isCompatible(particle1.energy, particle2.energy)
			&& isCompatible(particle1.metadata, particle2.metadata);
	}

	template<>
	bool isCompatible<DataDescription>(DataDescription const & data1, DataDescription const & data2)
	{
		return isCompatible(data1.clusters, data2.clusters)
			&& isCompatible(data1.particles, data2.particles);
	}
}

TEST_F(DataDescriptionTransferTest, testAddRandomData)
{

	DataDescription dataBefore;
	int numClusters = 100;
	for (int i = 1; i <= numClusters; ++i) {
		QVector2D pos(_numberGen->getRandomReal(0, 499), _numberGen->getRandomReal(0, 299));
		dataBefore.addCluster(createClusterDescription(i));
	}
	_access->updateData(dataBefore);

	IntRect rect = { { 0, 0 }, { _universeSize.x - 1, _universeSize.y - 1 } };
	ResolveDescription resolveDesc;
	_access->requireData(rect, resolveDesc);
	DataDescription dataAfter = _access->retrieveData();

	ASSERT_TRUE(dataBefore.clusters->size() == dataAfter.clusters->size());
	std::sort(dataBefore.clusters->begin(), dataBefore.clusters->end(), [](auto const &cluster1, auto const &cluster2) {
		return cluster1.cells->size() <= cluster2.cells->size();
	});
	std::sort(dataAfter.clusters->begin(), dataAfter.clusters->end(), [](auto const &cluster1, auto const &cluster2) {
		return cluster1.cells->size() <= cluster2.cells->size();
	});
	for (int i = 0; i < dataAfter.clusters->size(); ++i) {
		auto& cluster1 = dataAfter.clusters->at(i);
		auto& cluster2 = dataBefore.clusters->at(i);
		std::sort(cluster1.cells->begin(), cluster1.cells->end(), [](auto const &cell1, auto const &cell2) {
			return cell1.pos->x() <= cell2.pos->x();
		});
		std::sort(cluster2.cells->begin(), cluster2.cells->end(), [](auto const &cell1, auto const &cell2) {
			return cell1.pos->x() <= cell2.pos->x();
		});
	}
	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

TEST_F(DataDescriptionTransferTest, testAddAndDeleteRandomData)
{

	DataDescription dataBefore;
	int numClusters = 100;
	for (int i = 1; i <= numClusters; ++i) {
		QVector2D pos(_numberGen->getRandomReal(0, 499), _numberGen->getRandomReal(0, 299));
		dataBefore.addCluster(createClusterDescription(i));
	}
	_access->updateData(dataBefore);

	DataChangeDescription dataChange;
	for (int i = 0; i <= 49; ++i) {
		uint64_t id = dataBefore.clusters->at(i).id;
		auto pos = *dataBefore.clusters->at(i).pos;
		dataChange.addDeletedCluster(ClusterChangeDescription().setId(id).setPos(pos));
	}
	_access->updateData(dataChange);

	dataBefore.clusters->erase(dataBefore.clusters->begin(), dataBefore.clusters->begin() + 50);

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	ResolveDescription resolveDesc;
	_access->requireData(rect, resolveDesc);
	DataDescription dataAfter = _access->retrieveData();

	ASSERT_TRUE(dataBefore.clusters->size() == dataAfter.clusters->size());
	std::sort(dataBefore.clusters->begin(), dataBefore.clusters->end(), [](auto const &cluster1, auto const &cluster2) {
		return cluster1.cells->size() <= cluster2.cells->size();
	});
	std::sort(dataAfter.clusters->begin(), dataAfter.clusters->end(), [](auto const &cluster1, auto const &cluster2) {
		return cluster1.cells->size() <= cluster2.cells->size();
	});
	for (int i = 0; i < dataAfter.clusters->size(); ++i) {
		auto& cluster1 = dataAfter.clusters->at(i);
		auto& cluster2 = dataBefore.clusters->at(i);
		std::sort(cluster1.cells->begin(), cluster1.cells->end(), [](auto const &cell1, auto const &cell2) {
			return cell1.pos->x() <= cell2.pos->x();
		});
		std::sort(cluster2.cells->begin(), cluster2.cells->end(), [](auto const &cell1, auto const &cell2) {
			return cell1.pos->x() <= cell2.pos->x();
		});
	}
	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}
