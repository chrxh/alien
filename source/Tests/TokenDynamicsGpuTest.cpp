#include "SimulationGpuTest.h"

class TokenDynamicsGpuTest
	: public SimulationGpuTest
{
public:
	virtual ~TokenDynamicsGpuTest() = default;
};

/**
* Situation: - one horizontal cluster with 10 cells and ascending branch numbers
*			 - first cell has a token
*			 - simulating 9 time steps
* Expected result: token should be on the last cell
*/
TEST_F(TokenDynamicsGpuTest, testTokenMovement_fittingBranchNumbers)
{
	DataDescription origData;
	auto const& cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

	auto cluster = createHorizontalCluster(10, QVector2D{}, QVector2D{}, 0);
	for (int i = 0; i < 10; ++i) {
		auto& cell = cluster.cells->at(i);
		cell.tokenBranchNumber = i % cellMaxTokenBranchNumber;
	}
	auto& firstCell = cluster.cells->at(0);
	firstCell.addToken(createSimpleToken());
	origData.addCluster(cluster);
	
	uint64_t lastCellId = cluster.cells->at(9).id;

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(9, _controller);

	DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_EQ(1, newData.clusters->size());
	auto newCluster = newData.clusters->at(0);

	EXPECT_EQ(10, newCluster.cells->size());

	for (int i = 0; i < 10; ++i) {
		auto const& newCell = newCluster.cells->at(i);
		if (newCell.id == lastCellId) {
			EXPECT_EQ(1, newCell.tokens->size());
		}
		else {
			if (newCell.tokens) {
				EXPECT_TRUE(newCell.tokens->empty());
			}
		}
	}
}

/**
* Situation: - one horizontal cluster with 10 cells and equal branch numbers
*			 - first cell has a token
*			 - simulating one time step
* Expected result: no token should be on the cells
*/
TEST_F(TokenDynamicsGpuTest, testTokenMovement_unfittingBranchNumbers)
{
	DataDescription origData;
	auto const& cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

	auto cluster = createHorizontalCluster(10, QVector2D{}, QVector2D{}, 0);
	for (int i = 0; i < 10; ++i) {
		auto& cell = cluster.cells->at(i);
		cell.tokenBranchNumber = 0;
	}
	auto& firstCell = cluster.cells->at(0);
	firstCell.addToken(TokenDescription().setEnergy(30).setData(QByteArray(_parameters.tokenMemorySize, 0)));
	origData.addCluster(cluster);

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(1, _controller);

	DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_EQ(1, newData.clusters->size());
	auto newCluster = newData.clusters->at(0);

	EXPECT_EQ(10, newCluster.cells->size());

	for (int i = 0; i < 10; ++i) {
		auto const& newCell = newCluster.cells->at(i);
		if (newCell.tokens) {
			EXPECT_TRUE(newCell.tokens->empty());
		}
	}
}

//TODO: Tests mit Token und Decomposition/Fusion

