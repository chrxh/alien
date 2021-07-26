#include "Base/ServiceLocator.h"

#include "EngineInterface/Serializer.h"
#include "EngineInterface/SerializationHelper.h"

#include "IntegrationGpuTestFramework.h"

class TokenMovementGpuTests
	: public IntegrationGpuTestFramework
{
public:
	virtual ~TokenMovementGpuTests() = default;

protected:
    virtual void SetUp()
    {
        _parameters.radiationProb = 0;           //exclude radiation
        _parameters.cellTransformationProb = 0;  //excluding transformation of particle to cell
        _parameters.cellFusionVelocity = 0.4;
        _context->setSimulationParameters(_parameters);
    }

/*
    ClusterDescription createStickyRotatingTokenCluster(QVector2D const& pos, QVector2D const& vel);

    const float tokenTransferEnergyAmount = 10.0;
*/
};

/*
ClusterDescription TokenSpreadingGpuTests::createStickyRotatingTokenCluster(
    QVector2D const& pos,
    QVector2D const& vel)
{
    auto token = createSimpleToken();
    auto cluster = createRectangularCluster({2, 2}, pos, vel);
    for (auto& cell : *cluster.cells) {
        cell.maxConnections = 4;
    }
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    auto& thirdCell = cluster.cells->at(3);
    auto& fourthCell = cluster.cells->at(2);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    thirdCell.tokenBranchNumber = 2;
    fourthCell.tokenBranchNumber = 3;
    firstCell.addToken(token);
    return cluster;
}
*/


/**
 * Situation: - one horizontal cluster with 10 cells and ascending branch numbers
 *			 - first cell has a token
 *			 - simulating 9 time steps
 * Expected result: token should be on the last cell
 */
TEST_F(TokenMovementGpuTests, testMovementWithFittingBranchNumbers)
{
	DataDescription origData;
	auto const& cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

	auto cluster = _factory->createRect(
        DescriptionFactory::CreateRectParameters().size({10, 1}).centerPosition(QVector2D(10, 10)),
        _context->getNumberGenerator());
	for (int i = 0; i < 10; ++i) {
		auto& cell = cluster.cells->at(i);
		cell.tokenBranchNumber = 1 + i % cellMaxTokenBranchNumber;
	}
	auto& firstCell = cluster.cells->at(0);
	auto token = createSimpleToken();
    (*token.data)[0] = 1;
	firstCell.addToken(token);
	origData.addCluster(cluster);
	
	uint64_t lastCellId = cluster.cells->at(9).id;

	IntegrationTestHelper::updateData(_access, _context, origData);
	IntegrationTestHelper::runSimulation(9, _controller);

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_EQ(1, dataAfter.clusters->size());
	auto const& clusterAfter = dataAfter.clusters->at(0);

	EXPECT_EQ(10, clusterAfter.cells->size());

	for (auto const& cellAfter : *clusterAfter.cells) {
        if (cellAfter.id == lastCellId) {
            ASSERT_EQ(1, cellAfter.tokens->size());
            auto const& tokenAfter = cellAfter.tokens->at(0);
            EXPECT_EQ(*token.energy, *tokenAfter.energy);
        } else if (cellAfter.tokens) {
            EXPECT_TRUE(cellAfter.tokens->empty());
		}
	}
}

/**
 * Situation:
 * - one horizontal cluster with 10 cells and equal branch numbers 
 * - first cell has a token
 * - simulating one time step
 * Expected result: no token should be on the cells
 */
TEST_F(TokenMovementGpuTests, testMovementWithUnfittingBranchNumbers)
{
    DataDescription dataBefore;
    auto const& cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

    auto cluster = _factory->createRect(
        DescriptionFactory::CreateRectParameters().size({10, 1}).centerPosition(QVector2D(10, 10)),
        _context->getNumberGenerator());
    auto& firstCell = cluster.cells->at(0);
    auto token = createSimpleToken();
    (*token.data)[0] = 1;
    firstCell.addToken(token);
    dataBefore.addCluster(cluster);

    uint64_t lastCellId = cluster.cells->at(9).id;

    IntegrationTestHelper::updateData(_access, _context, dataBefore);
    IntegrationTestHelper::runSimulation(9, _controller);

    DataDescription dataAfter = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    ASSERT_EQ(1, dataAfter.clusters->size());
    auto const& clusterAfter = dataAfter.clusters->at(0);

    EXPECT_EQ(10, clusterAfter.cells->size());

    for (auto const& cellAfter : *clusterAfter.cells) {
        EXPECT_TRUE(cellAfter.tokens->empty());
    }
}

/**
 * Situation:
 * - one horizontal cluster with 10 cells and ascending branch numbers
 * - first cell has a token
 * - last cell has flag tokenBlocked
 * - simulating 9 time steps
 * Expected result: no token should be on the cells
 */
TEST_F(TokenMovementGpuTests, testMovementBlocked)
{
    DataDescription origData;
    auto const& cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

    auto cluster = _factory->createRect(
        DescriptionFactory::CreateRectParameters().size({10, 1}).centerPosition(QVector2D(10, 10)),
        _context->getNumberGenerator());
    for (int i = 0; i < 10; ++i) {
        auto& cell = cluster.cells->at(i);
        cell.tokenBranchNumber = 1 + i % cellMaxTokenBranchNumber;
    }
    auto& firstCell = cluster.cells->at(0);
    firstCell.addToken(createSimpleToken());

    auto& lastCell = cluster.cells->at(9);
    lastCell.tokenBlocked = true;
    origData.addCluster(cluster);

    uint64_t lastCellId = cluster.cells->at(9).id;

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(9, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    ASSERT_EQ(1, newData.clusters->size());
    auto newCluster = newData.clusters->at(0);

    EXPECT_EQ(10, newCluster.cells->size());
    for (auto const& newCell : *newCluster.cells) {
        if (newCell.tokens) {
            EXPECT_TRUE(newCell.tokens->empty());
        }
    }
}

TEST_F(TokenMovementGpuTests, testMovementWithUnfittingBranchNumbers_negativeValue)
{
    DataDescription origData;
    auto const& cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

    auto cluster = _factory->createRect(
        DescriptionFactory::CreateRectParameters().size({2, 1}).centerPosition(QVector2D(10, 10)),
        _context->getNumberGenerator());
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 5;
    secondCell.tokenBranchNumber = 4;

    QByteArray memory(_parameters.tokenMemorySize, 0);
    memory[0] = 0xa9;
    firstCell.addToken(TokenDescription().setEnergy(30).setData(memory));
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    auto const newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    auto const newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const newSecondCell = newCellByCellId.at(secondCell.id);
    EXPECT_EQ(0, newSecondCell.tokens->size());
}

/**
 * Situation:
 * - one horizontal cluster with 3 cells and branch numbers(0, 1, 0)
 * - first cell has 1 token
 * - third cell has 1 token
 * - simulating one time step
 * Expected result:
 * second cell should have 2 tokens
 */
TEST_F(TokenMovementGpuTests, testMovementWithEncounter)
{
    DataDescription origData;
    auto const& cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

    auto cluster = _factory->createRect(
        DescriptionFactory::CreateRectParameters().size({3, 1}).centerPosition(QVector2D(10, 10)),
        _context->getNumberGenerator());
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    auto& thirdCell = cluster.cells->at(2);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    thirdCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    firstCell.addToken(token);
    thirdCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    ASSERT_EQ(1, newData.clusters->size());
    auto newCluster = newData.clusters->at(0);

    EXPECT_EQ(3, newCluster.cells->size());

    for (auto const& newCell : *newCluster.cells) {
        if (newCell.id == secondCellId) {
            ASSERT_EQ(2, newCell.tokens->size());
            for (auto const& newToken : *newCell.tokens) {
                EXPECT_EQ(*token.energy, *newToken.energy);
            }
        } else if (newCell.tokens) {
            EXPECT_TRUE(newCell.tokens->empty());
        }
    }
}

/**
 * Situation:
 * - one horizontal cluster with 3 cells and branch numbers(1, 0, 1)
 * - second cell has a token
 * - simulating one time step
 * Expected result:
 * there should be two tokens: on the first and last cell
 */
TEST_F(TokenMovementGpuTests, testForking)
{
    DataDescription origData;
    auto cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

    auto cluster = _factory->createRect(
        DescriptionFactory::CreateRectParameters().size({3, 1}).centerPosition(QVector2D(10, 10)),
        _context->getNumberGenerator());
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    auto& thirdCell = cluster.cells->at(2);
    firstCell.tokenBranchNumber = 1;
    secondCell.tokenBranchNumber = 0;
    thirdCell.tokenBranchNumber = 1;
    secondCell.addToken(createSimpleToken());
    origData.addCluster(cluster);

    uint64_t firstCellId = firstCell.id;
    uint64_t thirdCellId = thirdCell.id;

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    ASSERT_EQ(1, newData.clusters->size());
    auto newCluster = newData.clusters->at(0);

    EXPECT_EQ(3, newCluster.cells->size());

    for (auto const& newCell : *newCluster.cells) {
        if (newCell.id == firstCellId || newCell.id == thirdCellId) {
            EXPECT_EQ(1, newCell.tokens->size());
        } else if (newCell.tokens) {
            EXPECT_TRUE(newCell.tokens->empty());
        }
    }
}

/*
/ **
* Situation: - 50 horizontal cluster with 100 cells each and ascending branch numbers
*			 - first cell on each cluster has cellMaxToken-many tokens
*			 - simulating 99 time steps
* Expected result: cellMaxToken-many tokens should be on the last cell of each cluster
* /
TEST_F(TokenSpreadingGpuTests, testMovementWithFittingBranchNumbers_manyLargeClusters)
{
	DataDescription origData;
	auto cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;
	auto cellMaxToken = _parameters.cellMaxToken;

    auto token = createSimpleToken();
	for (int clusterIndex = 0; clusterIndex < 50; ++clusterIndex) {
		auto cluster = createHorizontalCluster(100, QVector2D{0, static_cast<float>(clusterIndex) }, QVector2D{}, 0);
		for (int i = 0; i < 100; ++i) {
			auto& cell = cluster.cells->at(i);
			cell.tokenBranchNumber = i % cellMaxTokenBranchNumber;
		}
		auto& firstCell = cluster.cells->at(0);
		for (int i = 0; i < cellMaxToken; ++i) {
			firstCell.addToken(token);
		}
		origData.addCluster(cluster);
	}

	unordered_set<uint64_t> lastCellIds;
	for (int clusterIndex = 0; clusterIndex < 50; ++clusterIndex) {
		lastCellIds.insert(origData.clusters->at(clusterIndex).cells->at(99).id);
	}

	IntegrationTestHelper::updateData(_access, _context, origData);
	IntegrationTestHelper::runSimulation(99, _controller);

	DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_EQ(50, newData.clusters->size());

	for (auto const& newCluster : *newData.clusters) {
		EXPECT_EQ(100, newCluster.cells->size());

		for (auto const& newCell : *newCluster.cells) {
			if (lastCellIds.find(newCell.id) != lastCellIds.end()) {
				EXPECT_EQ(cellMaxToken, newCell.tokens->size());
                for (auto const& newToken : *newCell.tokens) {
                    EXPECT_EQ(*token.energy, *newToken.energy);
                }
			}
			else if (newCell.tokens) {
				EXPECT_TRUE(newCell.tokens->empty());
			}
		}
	}

    check(origData, newData);
}


/ **
* Situation: - one horizontal cluster with 3 cells and branch numbers (1, 0, 1)
*			 - second cell has a token
*			 - simulating one time step
* Expected result: there should be two tokens: on the first and last cell
* /
TEST_F(TokenSpreadingGpuTests, testForking)
{
	DataDescription origData;
	auto cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

	auto cluster = createHorizontalCluster(3, QVector2D{}, QVector2D{}, 0);
	auto& firstCell = cluster.cells->at(0);
	auto& secondCell = cluster.cells->at(1);
	auto& thirdCell = cluster.cells->at(2);
	firstCell.tokenBranchNumber = 1;
	secondCell.tokenBranchNumber = 0;
	thirdCell.tokenBranchNumber = 1;
	secondCell.addToken(createSimpleToken());
	origData.addCluster(cluster);

	uint64_t firstCellId = firstCell.id;
	uint64_t thirdCellId = thirdCell.id;

	IntegrationTestHelper::updateData(_access, _context, origData);
	IntegrationTestHelper::runSimulation(1, _controller);

	DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_EQ(1, newData.clusters->size());
	auto newCluster = newData.clusters->at(0);

	EXPECT_EQ(3, newCluster.cells->size());

	for (auto const& newCell : *newCluster.cells) {
		if (newCell.id == firstCellId || newCell.id == thirdCellId) {
			EXPECT_EQ(1, newCell.tokens->size());
		}
		else if (newCell.tokens) {
			EXPECT_TRUE(newCell.tokens->empty());
		}
	}

    check(origData, newData);
}

/ **
* Situation: - one horizontal cluster with 3 cells and branch numbers (1, 0, 1)
*			 - second cell has a token
*            - cells have low internal energy 
*			 - simulating one time step
* Expected result: there should be two tokens:
*                  on the first and last cell with half of the energy as of the initial token
* /
TEST_F(TokenSpreadingGpuTests, testForking_lowCellEnergies)
{
    DataDescription origData;
    auto token = createSimpleToken();
    auto cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;
    auto lowCellEnergy = _parameters.cellMinEnergy + *token.energy / 2 - 1.0;

    auto cluster = createHorizontalCluster(3, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    auto& thirdCell = cluster.cells->at(2);
    firstCell.tokenBranchNumber = 1;
    secondCell.tokenBranchNumber = 0;
    thirdCell.tokenBranchNumber = 1;
    firstCell.energy = lowCellEnergy;
    secondCell.energy = lowCellEnergy;
    thirdCell.energy = lowCellEnergy;
    secondCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t firstCellId = firstCell.id;
    uint64_t thirdCellId = thirdCell.id;

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());
    auto newCluster = newData.clusters->at(0);

    EXPECT_EQ(3, newCluster.cells->size());

    for (auto const& newCell : *newCluster.cells) {
        if (newCell.id == firstCellId || newCell.id == thirdCellId) {
            EXPECT_EQ(1, newCell.tokens->size());
            auto newToken = newCell.tokens->at(0);
            EXPECT_EQ(*token.energy / 2, *newToken.energy);
        }
        else if (newCell.tokens) {
            EXPECT_TRUE(newCell.tokens->empty());
        }
    }

    check(origData, newData);
}

/ **
* Situation: - one horizontal cluster with 5 cells and branch numbers (0, 1, 2, 1, 0)
*			 - first and last cell has a token
*			 - middle cell has too low energy
*			 - simulating 1 time step
* Expected result: cluster decomposes into two cluster, each still has a token
* /
TEST_F(TokenSpreadingGpuTests, testMovementDuringDecomposition)
{
	DataDescription origData;
	auto const& cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

	auto lowEnergy = _parameters.cellMinEnergy / 2.0;

	auto cluster = createHorizontalCluster(5, QVector2D{}, QVector2D{}, 0);
	cluster.cells->at(0).tokenBranchNumber = 0;
	cluster.cells->at(1).tokenBranchNumber = 1;
	cluster.cells->at(2).tokenBranchNumber = 2;
	cluster.cells->at(3).tokenBranchNumber = 1;
	cluster.cells->at(4).tokenBranchNumber = 0;
	cluster.cells->at(0).addToken(createSimpleToken());
	cluster.cells->at(4).addToken(createSimpleToken());
	cluster.cells->at(2).energy = lowEnergy;
	origData.addCluster(cluster);

	auto& secondCellId = cluster.cells->at(1).id;
	auto& fourthCellId = cluster.cells->at(3).id;

	IntegrationTestHelper::updateData(_access, _context, origData);
	IntegrationTestHelper::runSimulation(1, _controller);

	DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_EQ(2, newData.clusters->size());
	auto clusterById = IntegrationTestHelper::getClusterByCellId(newData);
	{
		auto cluster = clusterById.at(secondCellId);
		EXPECT_EQ(2, cluster.cells->size());
		for (auto const& cell : *cluster.cells) {
			if (cell.id == secondCellId) {
				EXPECT_EQ(1, cell.tokens->size());
			}
			else if (cell.tokens) {
				EXPECT_TRUE(cell.tokens->empty());
			}
		}
	}
	{
		auto cluster = clusterById.at(fourthCellId);
		EXPECT_EQ(2, cluster.cells->size());
		for (auto const& cell : *cluster.cells) {
			if (cell.id == fourthCellId) {
				EXPECT_EQ(1, cell.tokens->size());
			}
			else if (cell.tokens) {
				EXPECT_TRUE(cell.tokens->empty());
			}
		}
	}

    check(origData, newData);
}

TEST_F(TokenSpreadingGpuTests, testCreationAfterFusion)
{
    DataDescription origData;
    auto const velocity = 0.6f;

    auto firstCluster = createHorizontalCluster(2, QVector2D{ 100, 100.5 }, QVector2D{ 0, 0 }, 0.0);
    firstCluster.cells->at(0).tokenBranchNumber = 0;
    firstCluster.cells->at(1).tokenBranchNumber = 1;
    setMaxConnections(firstCluster, 2);
    origData.addCluster(firstCluster);

    auto secondCluster = createHorizontalCluster(2, QVector2D{ 102, 100.5 }, QVector2D{ -velocity, 0 }, 0.0);
    secondCluster.cells->at(0).tokenBranchNumber = 0;
    secondCluster.cells->at(1).tokenBranchNumber = 1;
    setMaxConnections(secondCluster, 2);
    origData.addCluster(secondCluster);

    auto secondCellId = firstCluster.cells->at(1).id;
    auto thirdCellId = secondCluster.cells->at(0).id;

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());
    auto newCluster = newData.clusters->at(0);
    EXPECT_EQ(4, newCluster.cells->size());
    for (auto const& newCell : *newCluster.cells) {
        if (newCell.id == secondCellId) {
            EXPECT_EQ(1, newCell.tokens->size());
        }
        else if (newCell.id == thirdCellId) {
            EXPECT_EQ(1, newCell.tokens->size());
        }
        else if (newCell.tokens) {
            EXPECT_TRUE(newCell.tokens->empty());
        }
    }
}

TEST_F(TokenSpreadingGpuTests, testCreationAfterSecondFusion)
{
    DataDescription origData;
    auto const velocity = 0.6f;

    auto firstCluster = createHorizontalCluster(2, QVector2D{ 96, 100.5 }, QVector2D{ velocity, 0 }, 0.0);
    firstCluster.cells->at(0).tokenBranchNumber = 0;
    firstCluster.cells->at(1).tokenBranchNumber = 1;
    setMaxConnections(firstCluster, 2);
    origData.addCluster(firstCluster);

    auto secondCluster = createHorizontalCluster(2, QVector2D{ 100, 100.5 }, QVector2D{ 0, 0 }, 0.0);
    secondCluster.cells->at(0).tokenBranchNumber = 0;
    secondCluster.cells->at(1).tokenBranchNumber = 1;
    setMaxConnections(secondCluster, 2);
    origData.addCluster(secondCluster);

    auto thirdCluster = createHorizontalCluster(2, QVector2D{ 102, 100.5 }, QVector2D{ -velocity, 0 }, 0.0);
    thirdCluster.cells->at(0).tokenBranchNumber = 0;
    thirdCluster.cells->at(1).tokenBranchNumber = 1;
    setMaxConnections(thirdCluster, 2);
    origData.addCluster(thirdCluster);

    auto secondCellId = firstCluster.cells->at(1).id;
    auto thirdCellId = secondCluster.cells->at(0).id;

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(3, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());
    auto newCluster = newData.clusters->at(0);
    EXPECT_EQ(6, newCluster.cells->size());
    for (auto const& newCell : *newCluster.cells) {
        if (newCell.id == secondCellId) {
            EXPECT_EQ(1, newCell.tokens->size());
        }
        else if (newCell.id == thirdCellId) {
            EXPECT_EQ(1, newCell.tokens->size());
        }
        else if (newCell.tokens) {
            EXPECT_TRUE(newCell.tokens->empty());
        }
    }
}


/ **
* Situation: - two horizontal clusters with each 2 cells and branch numbers (0, 1)
*			 - each cluster has token on its first cell
*			 - clusters are colliding for fusion
*			 - simulating 1 time step
* Expected result:
*			 - one cluster with branch numbers (0, 1, 0, 1)
*			 - first cell has no tokens
*			 - second cell has one token from spreading + one token from fusion
*			 - third cell has one token from fusion
*			 - fourth cell has one token from spreading
* /
TEST_F(TokenSpreadingGpuTests, testMovementDuringFusion)
{
	DataDescription origData;
	auto const velocity = 0.6f;

	auto firstCluster = createHorizontalCluster(2, QVector2D{ 100, 100.5 }, QVector2D{ 0, 0 }, 0.0);
	firstCluster.cells->at(0).tokenBranchNumber = 0;
	firstCluster.cells->at(1).tokenBranchNumber = 1;
	firstCluster.cells->at(0).addToken(createSimpleToken());
	setMaxConnections(firstCluster, 2);
	origData.addCluster(firstCluster);

	auto secondCluster = createHorizontalCluster(2, QVector2D{ 102, 100.5 }, QVector2D{ -velocity, 0 }, 0.0);
	secondCluster.cells->at(0).tokenBranchNumber = 0;
	secondCluster.cells->at(1).tokenBranchNumber = 1;
	secondCluster.cells->at(0).addToken(createSimpleToken());
	setMaxConnections(secondCluster, 2);
	origData.addCluster(secondCluster);

    auto firstCellId = firstCluster.cells->at(0).id;
    auto secondCellId = firstCluster.cells->at(1).id;
    auto thirdCellId = secondCluster.cells->at(0).id;
    auto fourthCellId = secondCluster.cells->at(1).id;

	IntegrationTestHelper::updateData(_access, _context, origData);
	IntegrationTestHelper::runSimulation(1, _controller);

	DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);

	ASSERT_EQ(1, newData.clusters->size());
	auto newCluster = newData.clusters->at(0);
	EXPECT_EQ(4, newCluster.cells->size());
    
    auto const newFirstCell = newCellByCellId.at(firstCellId);
    auto const newSecondCell = newCellByCellId.at(secondCellId);
    auto const newThirdCell = newCellByCellId.at(thirdCellId);
    auto const newFourthCell = newCellByCellId.at(fourthCellId);
    EXPECT_TRUE(!newFirstCell.tokens || newFirstCell.tokens->empty());
    EXPECT_EQ(2, newSecondCell.tokens->size());
    EXPECT_EQ(1, newThirdCell.tokens->size());
    EXPECT_EQ(1, newFourthCell.tokens->size());
}

/ **
* Situation: - one horizontal cluster with 3 cells and branch numbers (0, 1, 0)
*			 - first cell has cellMaxToken tokens
*			 - third cell has 1 token
*			 - simulating one time step
* Expected result: second cell should have cellMaxToken tokens
* /
TEST_F(TokenSpreadingGpuTests, testMovementWithTooManyTokens)
{
	DataDescription origData;
	auto cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

	auto cluster = createHorizontalCluster(3, QVector2D{}, QVector2D{}, 0);
	auto& firstCell = cluster.cells->at(0);
	auto& secondCell = cluster.cells->at(1);
	auto& thirdCell = cluster.cells->at(2);
	firstCell.tokenBranchNumber = 0;
	secondCell.tokenBranchNumber = 1;
	thirdCell.tokenBranchNumber = 0;
	for (int i = 0; i < _parameters.cellMaxToken; ++i) {
		firstCell.addToken(createSimpleToken());
	}
	thirdCell.addToken(createSimpleToken());
	origData.addCluster(cluster);

	uint64_t secondCellId = secondCell.id;

	IntegrationTestHelper::updateData(_access, _context, origData);
	IntegrationTestHelper::runSimulation(1, _controller);

	DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_EQ(1, newData.clusters->size());
	auto newCluster = newData.clusters->at(0);

	EXPECT_EQ(3, newCluster.cells->size());

	for (auto const& newCell : *newCluster.cells) {
		if (newCell.id == secondCellId) {
			EXPECT_EQ(_parameters.cellMaxToken, newCell.tokens->size());
		}
		else if (newCell.tokens) {
			EXPECT_TRUE(newCell.tokens->empty());
		}
	}

    check(origData, newData);
}

/ **
* Situation: - one horizontal cluster with 2 cells and ascending branch numbers
*			 - first cell has a token
*            - first cell has less energy than second cell
*			 - simulating one time step
*            - no radiation
* Expected result: both cell should have same energy, energy balance fulfilled
* /
TEST_F(TokenSpreadingGpuTests, testMovementAveragingCellEnergies)
{
    DataDescription origData;
    auto cellMinEnergy = _parameters.cellMinEnergy;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    firstCell.energy = cellMinEnergy * 2;
    secondCell.tokenBranchNumber = 1;
    secondCell.energy = cellMinEnergy * 4;
    auto token = createSimpleToken();
    firstCell.addToken(token);
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());
    auto const& newCluster = newData.clusters->at(0);

    EXPECT_EQ(2, newCluster.cells->size());

    for (auto const& newCell : *newCluster.cells) {
        EXPECT_EQ(cellMinEnergy * 3, *newCell.energy);
    }
    check(origData, newData);
}

/ **
* Situation: - one rectangular cluster with 100x100 cells and random branch numbers
*			 - each cell has random number of tokens
*			 - simulating 100 time steps
* Expected result: 100x100 cluster should still be there, energy balance fulfilled
* /
TEST_F(TokenSpreadingGpuTests, testMassiveMovements)
{
    auto cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;
    auto cellMaxToken = _parameters.cellMaxToken;
    auto cellMinEnergy = _parameters.cellMinEnergy;

    DataDescription origData;
    auto cluster = createRectangularCluster({ 100, 100 }, QVector2D{}, QVector2D{});
    auto token = createSimpleToken();
    for (auto& cell : *cluster.cells) {
        cell.tokenBranchNumber = _numberGen->getRandomInt(cellMaxTokenBranchNumber);
        cell.energy = cellMinEnergy * _numberGen->getRandomReal(1.0, 3.0);
        int numToken = _numberGen->getRandomInt(cellMaxToken);
        for (int i = 0; i < numToken; ++i) {
            cell.addToken(token);
        }
    }
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(100, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());
    auto const& newCluster = newData.clusters->at(0);

    EXPECT_EQ(100*100, newCluster.cells->size());
    check(origData, newData);
}

/ **
* Situation: - horizontal cluster with 3 cells and ascending branch numbers
*  			 - first cell has a token
*            - first and second cell have low energy
*			 - simulating 1 time step
* Expected result: low energy cells including token should be destroyed
* /
TEST_F(TokenSpreadingGpuTests, testMovementOnDestroyedCell_lowEnergy)
{
    auto cellMinEnergy = _parameters.cellMinEnergy;

    DataDescription origData;

    auto cluster = createHorizontalCluster(3, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    firstCell.energy = cellMinEnergy / 2;
    secondCell.energy = cellMinEnergy / 2;
    auto token = createSimpleToken();
    firstCell.addToken(token);
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());

    auto const& newCluster = newData.clusters->at(0);
    EXPECT_EQ(1, newCluster.cells->size());

    auto const& newCell = newCluster.cells->at(0);
    EXPECT_TRUE(newCell.tokens->empty());

    check(origData, newData);
}

/ **
* Situation: - horizontal cluster with 3 cells and ascending branch numbers
*   			 - first cell has a token
*            - further horizontal cluster with 5 cells which overlaps
*			 - simulating 1 time step
* Expected result: overlapping cells of the smaller cluster should be destroyed and token should be moved
* /
TEST_F(TokenSpreadingGpuTests, testMovementOnDestroyedCell_closeCell)
{
    auto lowDistance = _parameters.cellMinDistance / 2;
    uint64_t firstCellOfFirstClusterId;
    uint64_t firstCellOfSecondClusterId;
    uint64_t secondCellOfSecondClusterId;
    uint64_t thirdCellOfSecondClusterId;
    auto token = createSimpleToken();

    DataDescription origData;
    {
        auto cluster = createHorizontalCluster(3, QVector2D{0.1f, 0.1f}, QVector2D{}, 0);
        auto& firstCell = cluster.cells->at(0);
        auto& secondCell = cluster.cells->at(1);
        firstCell.tokenBranchNumber = 0;
        secondCell.tokenBranchNumber = 1;
        firstCell.addToken(token);
        origData.addCluster(cluster);
        firstCellOfFirstClusterId = firstCell.id;
    }
    {
        auto cluster = createHorizontalCluster(5, QVector2D{ 2.1f, 0.1f + static_cast<float>(lowDistance) }, QVector2D{}, 0);
        auto& firstCell = cluster.cells->at(0);
        auto& secondCell = cluster.cells->at(1);
        auto& thirdCell = cluster.cells->at(2);
        firstCell.tokenBranchNumber = 0;
        secondCell.tokenBranchNumber = 1;
        thirdCell.tokenBranchNumber = 2;
        firstCell.addToken(token);
        (*token.data)[Enums::Branching::TOKEN_BRANCH_NUMBER] = 1;
        secondCell.addToken(token);
        origData.addCluster(cluster);
        firstCellOfSecondClusterId = firstCell.id;
        secondCellOfSecondClusterId = secondCell.id;
        thirdCellOfSecondClusterId = thirdCell.id;
    }

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(2, newData.clusters->size());

    auto newClusterById = IntegrationTestHelper::getClusterByCellId(newData);
    {
        auto const& newCluster = newClusterById.at(firstCellOfFirstClusterId);
        EXPECT_EQ(1, newCluster.cells->size());
        auto const& newCell = newCluster.cells->at(0);
    }
    {
        auto const& newCluster = newClusterById.at(firstCellOfSecondClusterId);
        EXPECT_EQ(5, newCluster.cells->size());
        for (auto const& newCell : *newCluster.cells) {
            if (newCell.id == secondCellOfSecondClusterId || newCell.id == thirdCellOfSecondClusterId) {
                EXPECT_EQ(1, newCell.tokens->size());
            }
            else if (newCell.tokens) {
                EXPECT_TRUE(newCell.tokens->empty());
            }
        }

    }
    check(origData, newData);
}

/ **
* Situation: - many 2x2 clusters with circular ascending branch numbers
*  			 - first cell of each cluster has a token
*			 - simulating 100 time steps
* Expected result: no crash
* /
TEST_F(TokenSpreadingGpuTests, regressionTestManyStickyRotatingTokenClusters)
{
    auto const highVel = _parameters.cellFusionVelocity*2;

    _parameters.cellMaxTokenBranchNumber = 4;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 25; ++i) {
        auto cluster = createStickyRotatingTokenCluster(
            QVector2D{static_cast<float>(_numberGen->getRandomReal(-10, 10)),
                      static_cast<float>(_numberGen->getRandomReal(-10, 10))},
            QVector2D{static_cast<float>(_numberGen->getRandomReal(-highVel, highVel)),
                      static_cast<float>(_numberGen->getRandomReal(-highVel, highVel))});
        origData.addCluster(cluster);
    }

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(100, _controller);
}

/ **
* Situation: - one horizontal cluster with 2 cells and ascending branch numbers
*			 - first cell has a token with low energy
*			 - simulating 1 time step
* Expected result: energy balance fulfilled
* /
TEST_F(TokenSpreadingGpuTests, regressionTestLowTokenEnergy)
{
    auto const& cellMaxTokenBranchNumber = _parameters.cellMaxTokenBranchNumber;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    token.energy = _parameters.tokenMinEnergy / 2;
    (*token.data)[0] = 1;
    firstCell.addToken(token);

    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;

    DataDescription origData;
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());
    auto const& newCluster = newData.clusters->at(0);

    EXPECT_EQ(2, newCluster.cells->size());

    for (auto const& newCell : *newCluster.cells) {
        EXPECT_TRUE(!newCell.tokens || newCell.tokens->empty());
    }

    check(origData, newData);
}

/ **
* Situation: - one horizontal cluster with ascending branch numbers
*            - first cell has token
*			 - first and second cell have low energy
*            - cluster approaches other cluster with fusion velocity
*			 - simulating 1 time step
* Expected result: energy balance fulfilled
* /
TEST_F(TokenSpreadingGpuTests, regressionTestMovementOnLowEnergyCellWithSimultaneousFusion)
{
    auto const cellMinEnergy = _parameters.cellMinEnergy;
    auto const fusionVel = _parameters.cellFusionVelocity * 1.5f;

    auto cluster =
        createHorizontalCluster(3, QVector2D{}, QVector2D{fusionVel, 0}, 0, IntegrationTestFramework::Boundary::Sticky);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    firstCell.energy = cellMinEnergy / 2;
    secondCell.energy = cellMinEnergy / 2;
    auto token = createSimpleToken();
    firstCell.addToken(token);

    auto otherCluster =
        createHorizontalCluster(3, QVector2D{3, 0}, QVector2D{}, 0, IntegrationTestFramework::Boundary::Sticky);

    DataDescription origData;
    origData.addCluster(cluster);
    origData.addCluster(otherCluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());
    check(origData, newData);
}

TEST_F(TokenSpreadingGpuTests, testCellDecayDueToTokenUsage)
{
    _parameters.cellMinTokenUsages = 1;
    _parameters.cellTokenUsageDecayProb = 1;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    auto cluster = createHorizontalCluster(3, QVector2D{}, QVector2D{}, 0);
    int index = 0;
    auto token = createSimpleToken();
    for (auto& cell : *cluster.cells) {
        cell.tokenBranchNumber = ++index;
        (*token.data)[Enums::Branching::TOKEN_BRANCH_NUMBER] = index;
        cell.addToken(token);
    }
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(2, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    check(origData, newData);

    ASSERT_EQ(1, newData.clusters->size());

    auto const& newCluster = newData.clusters->at(0);
    EXPECT_GE(2, newCluster.cells->size());

}
*/
