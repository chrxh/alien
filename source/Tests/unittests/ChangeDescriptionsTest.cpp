#include <gtest/gtest.h>

#include "Model/Entities/ChangeDescriptions.h"

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
	const CellFunctionDescription cellFunction1 = CellFunctionDescription().setType(Enums::CellFunction::CONSTRUCTOR);
	const vector<TokenDescription> tokens1 = { TokenDescription().setEnergy(200) };

	CellDescription desc1 = CellDescription().setCellFunction(cellFunction1).setId(id)
		.setEnergy(energy1).setConnectingCells(connectingCells1).setTokenBranchNumber(tokenBranchNumber1)
		.setMetadata(metadata1).setCellFunction(cellFunction1).setTokens(tokens1);

	const double energy2 = 200;
	const list<uint64_t> connectingCells2 = { 104, 503 };
	const int maxConnections2 = 3;
	const CellMetadata metadata2 = CellMetadata().setName("test2");
	const CellFunctionDescription cellFunction2 = CellFunctionDescription().setType(Enums::CellFunction::COMPUTER);
	const vector<TokenDescription> tokens2 = { TokenDescription().setEnergy(300) };

	CellDescription desc2 = CellDescription().setCellFunction(cellFunction2).setId(id)
		.setEnergy(energy2).setConnectingCells(connectingCells2).setMaxConnections(maxConnections2)
		.setMetadata(metadata2).setCellFunction(cellFunction2).setTokens(tokens2);

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
}
