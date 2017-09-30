#include <gtest/gtest.h>

#include "Model/ChangeDescriptions.h"

#include "tests/Predicates.h"

class ChangeDescriptionsTest : public ::testing::Test
{
public:
	ChangeDescriptionsTest() = default;
	~ChangeDescriptionsTest() = default;

};


TEST_F(ChangeDescriptionsTest, testCreateCellChangeDescriptionFromCellDescriptions)
{
	const uint64_t id = 201;
	const double energy1 = 200;
	const list<uint64_t> connectingCells1 = { 204, 503 };
	const int tokenBranchNumber1 = 5;
	const CellMetadata metadata1 = CellMetadata().setName("test");
	const CellFeatureDescription cellFunction1 = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
	const vector<TokenDescription> tokens1 = { TokenDescription().setEnergy(200) };

	CellDescription desc1 = CellDescription().setId(id).setCellFeature(cellFunction1)
		.setEnergy(energy1).setConnectingCells(connectingCells1).setTokenBranchNumber(tokenBranchNumber1)
		.setMetadata(metadata1).setCellFeature(cellFunction1).setTokens(tokens1);

	const double energy2 = 200;
	const list<uint64_t> connectingCells2 = { 104, 503 };
	const int maxConnections2 = 3;
	const CellMetadata metadata2 = CellMetadata().setName("test2");
	const CellFeatureDescription cellFunction2 = CellFeatureDescription().setType(Enums::CellFunction::COMPUTER);
	const vector<TokenDescription> tokens2 = { TokenDescription().setEnergy(300) };

	CellDescription desc2 = CellDescription().setId(id).setCellFeature(cellFunction2)
		.setEnergy(energy2).setConnectingCells(connectingCells2).setMaxConnections(maxConnections2)
		.setMetadata(metadata2).setCellFeature(cellFunction2).setTokens(tokens2);

	CellChangeDescription change(desc1, desc2);

	ASSERT_EQ(id, change.id);
	ASSERT_EQ(boost::none, change.energy);
	ASSERT_EQ(connectingCells2, *change.connectingCells);
	ASSERT_EQ(boost::none, change.tokenBranchNumber);
	ASSERT_EQ(maxConnections2, *change.maxConnections);
	ASSERT_EQ(metadata2, *change.metadata);
	ASSERT_EQ(cellFunction2, *change.cellFunction);
	ASSERT_EQ(tokens2, *change.tokens);
}

TEST_F(ChangeDescriptionsTest, testCreateClusterChangeDescriptionFromClusterDescriptions)
{
	const uint64_t id = 201;
	const QVector2D pos1(100, 20.4);
	const double angle1 = 104.0;
	const double angularVel1 = 1.0;

	const uint64_t cellId1 = 201;
	const uint64_t cellId2 = 202;
	const uint64_t cellId3 = 203;
	const double energyCell1 = 100;
	const int maxConnectionsCell2 = 5;
	const list<CellDescription> cells1 = {
		CellDescription().setId(cellId1).setPos({ 0, 0 }).setEnergy(energyCell1),
		CellDescription().setId(cellId2).setPos({ 1, 0 }).setMaxConnections(maxConnectionsCell2),
		CellDescription().setId(cellId3).setPos({ 2, 0 })
	};

	ClusterDescription desc1 = ClusterDescription().setId(id).setPos(pos1).setAngle(angle1).setAngularVel(angularVel1).addCells(cells1);

	const QVector2D vel2(-1, -2.2);
	const double angle2 = 104.0;
	const double angularVel2 = -1.9;

	const uint64_t cellId4 = 204;
	const double newEnergyCell1 = 140;
	const int maxConnectionsCell4 = 5;
	const list<CellDescription> cells2 = {
		CellDescription().setId(cellId1).setPos({ 0, 0 }).setEnergy(newEnergyCell1),
		CellDescription().setId(cellId3).setPos({ 2, 0 }),
		CellDescription().setId(cellId4).setPos({ 3, 0 }).setMaxConnections(maxConnectionsCell4)
	};
	ClusterDescription desc2 = ClusterDescription().setId(id).setVel(vel2).setAngle(angle2).setAngularVel(angularVel2).addCells(cells2);

	ClusterChangeDescription change(desc1, desc2);

	ASSERT_EQ(id, change.id);
	ASSERT_EQ(boost::none, change.pos);
	ASSERT_EQ(vel2, *change.vel);
	ASSERT_EQ(boost::none, change.angle);
	ASSERT_EQ(angularVel2, *change.angularVel);
	ASSERT_EQ(3, change.cells.size());

	unordered_map<uint64_t, int> cellIndicesByIds;
	for (int index = 0; index < change.cells.size(); ++index) {
		cellIndicesByIds.insert_or_assign(change.cells.at(index)->id, index);
	}

	int index1 = cellIndicesByIds.at(cellId1);
	int index2 = cellIndicesByIds.at(cellId2);
	int index4 = cellIndicesByIds.at(cellId4);
	ASSERT_TRUE(change.cells.at(index1).isModified());
	ASSERT_TRUE(change.cells.at(index2).isDeleted());
	ASSERT_TRUE(change.cells.at(index4).isAdded());

	CellChangeDescription cell1 = change.cells.at(index1).getValue();
	CellChangeDescription cell4 = change.cells.at(index4).getValue();
	ASSERT_EQ(newEnergyCell1, *cell1.energy);
	ASSERT_EQ(maxConnectionsCell4, *cell4.maxConnections);
}
