#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/ShapeGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

#include "IntegrationTestFramework.h"

class ConstructorTests : public IntegrationTestFramework
{
public:
    ConstructorTests()
        : IntegrationTestFramework()
    {
        _descriptionTestDataFactory = &DescriptionTestDataFactory::get();
    }

    ~ConstructorTests() = default;

protected:
    float getConstructorEnergy() const
    {
        return _parameters.normalCellEnergy.value[0] * 3.5f;  // Provide enough energy
    }
    float getOffspringDistance() const { return 1.0f + _parameters.constructorAdditionalOffspringDistance; }

    DescriptionTestDataFactory* _descriptionTestDataFactory;
};

TEST_F(ConstructorTests, alreadyFinished)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(1))),
            ObjectDescription().id(1).pos({100.0f, 101.0f}),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_F(ConstructorTests, emptyGenome)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(0))),
        }, CreatureDescription().id(0), GenomeDescription());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_F(ConstructorTests, emptyGene)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(0))),
        }, CreatureDescription().id(0), GenomeDescription().genes({GeneDescription().separation(true)}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_F(ConstructorTests, nodeIndexOutOfRange)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(0).currentNodeIndex(1))),
        }, CreatureDescription().id(0), GenomeDescription().genes({GeneDescription().separation(true)}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_F(ConstructorTests, geneIndexOutOfRange)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(1).currentBranch(0).currentNodeIndex(0))),
        }, CreatureDescription().id(0), GenomeDescription().genes({GeneDescription().separation(true)}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_F(ConstructorTests, insufficientEnergy)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().cellType(ConstructorDescription().geneIndex(0).currentBranch(0).currentNodeIndex(0))),
        }, CreatureDescription().id(0), GenomeDescription().genes({GeneDescription().separation(true).nodes({NodeDescription()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_F(ConstructorTests, manuallyTriggered_withSignal_failed)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().cellType(ConstructorDescription().autoTriggerInterval(std::nullopt).geneIndex(0).currentBranch(0).currentNodeIndex(0))),  // Not enough energy
            ObjectDescription().id(1).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
        }, CreatureDescription().id(0), GenomeDescription().genes({GeneDescription().separation(true).nodes({NodeDescription()})}));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(2, creature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    ASSERT_TRUE(hostCell.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(0.0f, hostCell.getCellRef()._signal._channels[Channels::ConstructorSuccess]));
}

TEST_F(ConstructorTests, manuallyTriggered_withSignal_success)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().autoTriggerInterval(std::nullopt).geneIndex(0).currentBranch(0).currentNodeIndex(0))),
            ObjectDescription().id(1).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
        }, CreatureDescription().id(0), GenomeDescription().genes({GeneDescription().separation(false).nodes({NodeDescription()})}));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(3, creature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
    ASSERT_TRUE(hostCell.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(1.0f, hostCell.getCellRef()._signal._channels[Channels::ConstructorSuccess]));
}

TEST_F(ConstructorTests, manuallyTriggered_withoutSignal)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().autoTriggerInterval(std::nullopt).geneIndex(0).currentBranch(0).currentNodeIndex(0))),
            ObjectDescription().id(1).pos({101.0f, 100.0f}),
        }, CreatureDescription().id(0), GenomeDescription().genes({GeneDescription().separation(false).nodes({NodeDescription()})}));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(2, creature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_F(ConstructorTests, lastConstructedCellNotFound)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(0).lastConstructedCellId(1))),
        }, CreatureDescription().id(0), GenomeDescription().genes({GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(2, hostCreature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_F(ConstructorTests, insufficientSpace)
{
    Description data;
    data.addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentConcatenation(0).currentNodeIndex(1).lastConstructedCellId(1))),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),
        }));
    data.addCreature({
            ObjectDescription().id(1).pos({100.5f, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(1), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(1, hostCreature._numObjects);

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    ASSERT_EQ(1, newCreature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_F(ConstructorTests, crossingLinks)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription()}),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(10).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(3).lastConstructedCellId(3))),
            ObjectDescription().id(3).pos({10.0f, 10.0f + getOffspringDistance()}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(2)),
            ObjectDescription().id(2).pos({9.0f, 9.0f + getOffspringDistance()}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(1)),
            ObjectDescription().id(1).pos({11.0f, 9.0f + getOffspringDistance()}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(0)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 1);
    data.addConnection(3, 10);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    // No cell constructed
    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(creature._id).size());
}

using NodeParameter = DescriptionTestDataFactory::NodeParameter;
class ConstructorTests_AllNodeTypes
    : public ConstructorTests
    , public testing::WithParamInterface<NodeParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_AllNodeTypes,
    ConstructorTests_AllNodeTypes,
    ::testing::ValuesIn(DescriptionTestDataFactory::get().getAllNodeParameters()));

TEST_P(ConstructorTests_AllNodeTypes, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0)
{
    auto nodeParameter = GetParam();
    auto constexpr FrontAngleId = 5;

    auto randomNode = _descriptionTestDataFactory->createNonDefaultNodeDescription(nodeParameter);

    auto data = Description().addCreature({ObjectDescription().pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription()).frontAngleId(FrontAngleId))}, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({randomNode}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numObjects);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numObjects);

    auto hostCell = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newCell = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(newCell.getCellRef()._headCell);
    EXPECT_EQ(FrontAngleId, newCell.getCellRef()._frontAngleId);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(_descriptionTestDataFactory->compare(newCell, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_FALSE(hostCell.getCellRef()._signalState == SignalState_Active);
}

TEST_P(ConstructorTests_AllNodeTypes, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0__preview)
{
    auto nodeParameter = GetParam();
    auto randomNode = _descriptionTestDataFactory->createNonDefaultNodeDescription(nodeParameter);

    auto data = Description().addCreature({ObjectDescription().pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription()))}, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({randomNode}),
        }));

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(1);

    auto actualData = _simulationFacade->getPreviewData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numObjects);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numObjects);

    auto hostCell = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newCell = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(Math::length(hostCell._pos - newCell._pos) > 50.0f);  // Preview specific: Move seed far away from construction
    EXPECT_TRUE(_descriptionTestDataFactory->compare(newCell, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0__preview_detail)
{
    auto randomNode = _descriptionTestDataFactory->createNonDefaultNodeDescription(NodeParameter{CellTypeGenome_Base});

    auto data = Description().addCreature({ObjectDescription().pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription()))}, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({randomNode}),
        }));

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(1, true);

    auto actualData = _simulationFacade->getPreviewData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numObjects);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numObjects);

    auto hostCell = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newCell = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(Math::length(hostCell._pos - newCell._pos) > 50.0f);  // Preview specific: Move seed far away from construction
    EXPECT_TRUE(_descriptionTestDataFactory->compare(newCell, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_1)
{
    auto data = Description().addCreature({ObjectDescription().pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(1)))}, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true),
            GeneDescription().separation(true).nodes({NodeDescription()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numObjects);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numObjects);

    auto hostCell = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newCell = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_2_3__concatenation_0_1__branch_0_0__frontAngle_upperSide)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription(), NodeDescription().cellType(ConstructorGenomeDescription())}),
    });
    auto data = Description()
                    .addCreature({
                            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(2).geneIndex(0).lastConstructedCellId(2))),
                        }, CreatureDescription().id(0), genome)
                    .addCreature({
                            ObjectDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
                            ObjectDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).type(CellDescription().cellState(CellState_Constructing)),
                        }, CreatureDescription().id(1), genome);
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(3, actualData.getObjectsForCreature(newCreature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3});
    auto prevConstructedCell = actualData.getObjectRef(2);
    auto prevPrevConstructedCell = actualData.getObjectRef(3);

    EXPECT_EQ(CellState_Ready, actualConstructedCell.getCellRef()._cellState);

    EXPECT_EQ(CellState_Ready, prevConstructedCell.getCellRef()._cellState);

    EXPECT_EQ(CellState_Ready, prevPrevConstructedCell.getCellRef()._cellState);
}

TEST_F(ConstructorTests, creature_1__node_2_3__concatenation_0_1__branch_0_0__frontAngle_lowerSide)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription(), NodeDescription().cellType(ConstructorGenomeDescription())}),
    });
    auto data = Description()
                    .addCreature({
                            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(2).geneIndex(0).lastConstructedCellId(2))),
                        }, CreatureDescription().id(0), genome)
                    .addCreature({
                            ObjectDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
                            ObjectDescription().id(3).pos({10.0f + getOffspringDistance(), 11.0f}).type(CellDescription().cellState(CellState_Constructing)),
                        }, CreatureDescription().id(1), genome);
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(3, actualData.getObjectsForCreature(newCreature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3});
    auto prevConstructedCell = actualData.getObjectRef(2);
    auto prevPrevConstructedCell = actualData.getObjectRef(3);

    EXPECT_EQ(CellState_Ready, actualConstructedCell.getCellRef()._cellState);

    EXPECT_EQ(CellState_Ready, prevConstructedCell.getCellRef()._cellState);

    EXPECT_EQ(CellState_Ready, prevPrevConstructedCell.getCellRef()._cellState);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_1__gene_0)
{
    auto data = Description().addCreature({ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0)))}, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(2, hostCreature._numObjects);

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef(0);
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_1__gene_1)
{
    auto data = Description().addCreature({ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(1)))}, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true),
            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef(0);
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_FALSE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_2__concatenation_0_1__branch_0_1)
{
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature({ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0)))}, CreatureDescription()
            .id(0)
            .frontAngleId(InitialFrontAngleId), GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).numConcatenations(1).nodes({NodeDescription(), NodeDescription()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    EXPECT_EQ(InitialFrontAngleId, creature._frontAngleId);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef(0);
    EXPECT_EQ(CellState_Constructing, newCell.getCellRef()._cellState);
    EXPECT_TRUE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f + _parameters.constructorAdditionalOffspringDistance, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_2__branch_0_1)
{
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature({ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentConcatenation(0)))}, CreatureDescription()
            .id(0)
            .frontAngleId(InitialFrontAngleId), GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).numConcatenations(2).nodes({NodeDescription()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    EXPECT_EQ(InitialFrontAngleId + 1, hostCreature._frontAngleId);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef(0);
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f + _parameters.constructorAdditionalOffspringDistance, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

class ConstructorTests_BendingMuscles
    : public ConstructorTests
    , public testing::WithParamInterface<MuscleMode>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_BendingMuscles,
    ConstructorTests_BendingMuscles,
    ::testing::Values(MuscleMode_AutoBending, MuscleMode_ManualBending, MuscleMode_AngleBending));

TEST_P(ConstructorTests_BendingMuscles, creature_2__node_0_1__concatenation_1_2__branch_0_1__resetBendingMuscle)
{
    auto muscleModeType = GetParam();
    auto const InitialFrontAngleId = 4;

    auto muscleMode = [&muscleModeType] -> MuscleModeDescription {
        if (muscleModeType == MuscleMode_AutoBending)
            return AutoBendingDescription().initialAngle(90.0f);
        else if (muscleModeType == MuscleMode_ManualBending)
            return ManualBendingDescription().initialAngle(90.0f);
        else
            return AngleBendingDescription().initialAngle(90.0f);
    }();

    auto data = Description().addCreature({
                ObjectDescription().id(0).pos({100.0f, 100.0f}),
                ObjectDescription().id(1).pos({100.0f, 101.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentConcatenation(1).lastConstructedCellId(3))),
                ObjectDescription().id(2).pos({100.0f, 102.0f}),
                ObjectDescription().id(3).pos({100.0f + getOffspringDistance(), 101.0f}).type(CellDescription().cellType(MuscleDescription().mode(muscleMode))),
            }, CreatureDescription()
            .id(0)
            .frontAngleId(InitialFrontAngleId), GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).numConcatenations(2).nodes({NodeDescription().cellType(MuscleGenomeDescription())}),
        }));
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    EXPECT_EQ(InitialFrontAngleId + 1, hostCreature._frontAngleId);
    ASSERT_EQ(5, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(3);
    auto newCell = actualData.getOtherObjectRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(hostCell._pos + RealVector2D(1.0f, 0.0f), newCell._pos));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);

    auto muscle = std::get<MuscleDescription>(prevCell.getCellRef()._cellType);
    if (muscleModeType == MuscleMode_AutoBending) {
        EXPECT_FALSE(std::get<AutoBendingDescription>(muscle._mode)._initialAngle.has_value());
    } else if (muscleModeType == MuscleMode_ManualBending) {
        EXPECT_FALSE(std::get<ManualBendingDescription>(muscle._mode)._initialAngle.has_value());
    } else {
        EXPECT_FALSE(std::get<AngleBendingDescription>(muscle._mode)._initialAngle.has_value());
    }
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_0_2)
{
    auto const InitialFrontAngleId = 4;

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(2).nodes({NodeDescription()}),
    });
    auto data = Description().addCreature({
                ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(0))),
            }, CreatureDescription()
            .id(0)
            .frontAngleId(InitialFrontAngleId), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    EXPECT_EQ(InitialFrontAngleId + 1, hostCreature._frontAngleId);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef({0});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_1_2)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(2).nodes({NodeDescription()}),
    });
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(1))),
            ObjectDescription().id(1).pos({100.0f, 101.0f}),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef({0, 1});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_FALSE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(0.0f, 1.0f), newCell._pos));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(2, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_1_2__firstBranchMissing)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(2).nodes({NodeDescription()}),
    });
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(1))),
        }, CreatureDescription().id(0), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef({0});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(2, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__ignoreNumAdditionalConnectionsAtStart)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().referenceAngle(0).numAdditionalConnections(1),
        }),
    });
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(0))),
        }, CreatureDescription().id(0), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef({0});

    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_FALSE(actualData.hasConnection(newCell, hostCell));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_0_0)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0))),
            ObjectDescription().id(1).pos({101.0f, 100.0f}),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({NodeDescription()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostCell = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newCell = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(1.0f, 0.0f), newCell._pos));
    EXPECT_FALSE(actualData.hasConnection(0, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(1, newCell._id));
    EXPECT_TRUE(actualData.hasConnection(0, 1));
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_0_1)
{
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0))),
            ObjectDescription().id(1).pos({101.0f, 100.0f}),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef({0, 1});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(1.0f, 0.0f), newCell._pos));
    EXPECT_TRUE(actualData.hasConnection(0, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(1, newCell._id));
    EXPECT_TRUE(actualData.hasConnection(0, 1));
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_1_2)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(2).nodes({NodeDescription()}),
    });
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({101.0f, 100.0f}),
            ObjectDescription().id(1).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(1).constructionAngle(
                    10.0f)))  // constructionAngle should be ignored for the second branch
                .pos({100.0f, 100.0f}),
            ObjectDescription().id(2).pos({99.0f, 100.0f}),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(1);
    auto newCell = actualData.getOtherObjectRef({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(hostCell._pos + RealVector2D(0.0f, 1.0f), newCell._pos));
    EXPECT_TRUE(actualData.hasConnection(1, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(0, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(2, newCell._id));
    EXPECT_TRUE(actualData.hasConnection(1, 0));
    EXPECT_TRUE(actualData.hasConnection(1, 2));
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1)
{
    auto const ConstructionAngle = 0;
    //20.0f;

    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({101.0f, 100.0f}),
            ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0))),
            ObjectDescription().id(2).pos({100.0f, 101.0f}),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription().referenceAngle(ConstructionAngle)}),
        }));
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(1);
    auto newCell = actualData.getOtherObjectRef({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));
    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);
    EXPECT_TRUE(approxCompare(90.0f + 45.0f + ConstructionAngle, connection._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_1_2__concatenation_0_1__branch_0_0)
{
    auto const InitialFrontAngleId = 4;

    Description data;
    data.addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),
        }));
    data.addCreature({
                ObjectDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
            }, CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getCreatureRef(1);
    EXPECT_EQ(InitialFrontAngleId + 1, newCreature._frontAngleId);
    ASSERT_EQ(2, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(1);
    auto newCell = actualData.getOtherObjectRef({0, 1});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, newCell));
}

TEST_F(ConstructorTests, creature_1__node_1_2__concatenation_0_1__branch_0_1)
{
    auto const LastAngle = 45.0f;
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription(), NodeDescription().referenceAngle(LastAngle)}).separation(false).numBranches(1),
    });
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))),
            ObjectDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
        ASSERT_EQ(1, actualData._creatures.size());
        EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

        auto creature = actualData.getCreatureRef(0);
        ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

        auto hostCell = actualData.getObjectRef(0);
        auto prevCell = actualData.getObjectRef(1);
        auto newCell = actualData.getOtherObjectRef({0, 1});
        EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
        EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
        EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
        EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
        EXPECT_TRUE(actualData.hasConnection(hostCell, newCell));
        EXPECT_TRUE(approxCompare(180.0f + LastAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));
    }

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto prevCell = actualData.getObjectRef(1);
        EXPECT_EQ(CellState_Activating, prevCell.getCellRef()._cellState);
    }
}

TEST_F(ConstructorTests, creature_3__node_1_2__concatenation_0_1__branch_0_1)
{
    auto const MiddleAngle = 5.0f;
    auto const InitialFrontAngleId = 4;

    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription(), NodeDescription().referenceAngle(MiddleAngle)}).separation(false).numBranches(1),
    });
    auto data = Description().addCreature({
                ObjectDescription().id(0).pos({101.0f, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
                ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(3))),
                ObjectDescription().id(2).pos({100.0f, 101.0f}),
                ObjectDescription().id(3).pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * (1.0f + _parameters.constructorAdditionalOffspringDistance)).type(CellDescription().cellState(CellState_Constructing)),
            }, CreatureDescription()
            .id(0)
            .frontAngleId(InitialFrontAngleId), genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(creature._id).size());
    EXPECT_EQ(InitialFrontAngleId + 1, creature._frontAngleId);

    auto hostCell = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(3);
    auto newCell = actualData.getOtherObjectRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
    EXPECT_TRUE(actualData.hasConnection(hostCell, newCell));
    EXPECT_TRUE(approxCompare(180.0f + MiddleAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_3__node_1_2__concatenation_0_1__branch_0_1__mirrored)
{
    auto const MiddleAngle = 5.0f;
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription(), NodeDescription().referenceAngle(MiddleAngle)}).separation(false).numBranches(1),
    });
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 101.0f}),
            ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(3))),
            ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * (1.0f + _parameters.constructorAdditionalOffspringDistance)).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(creature._id).size());

    auto hostCell = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(3);
    auto newCell = actualData.getOtherObjectRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
    EXPECT_TRUE(actualData.hasConnection(hostCell, newCell));
    EXPECT_TRUE(approxCompare(180.0f + MiddleAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_3__node_1_2__concatenation_0_1__branch_0_1__onSpike)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription(), NodeDescription().numAdditionalConnections(0)}),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}),
            ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(3))),
            ObjectDescription().id(3).pos({11.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(creature._id).size());

    auto actualOtherObject = actualData.getObjectRef(1);
    auto actualHostCell = actualData.getObjectRef(2);
    auto actualPrevConstructedCell = actualData.getObjectRef(3);
    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3});

    ASSERT_EQ(1, actualOtherObject._connections.size());
    ASSERT_EQ(2, actualHostCell._connections.size());
    ASSERT_EQ(2, actualConstructedCell._connections.size());
    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualOtherObject, actualHostCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualHostCell, actualOtherObject)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_1__node_1_3__concatenation_0_1__branch_0_0)
{
    Description data;
    data.addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true).nodes(
                {NodeDescription().referenceAngle(0.0f), NodeDescription().referenceAngle(45.0f), NodeDescription().referenceAngle(0.0f)}),
        }));
    data.addCreature({
            ObjectDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(1), GenomeDescription().genes({
            GeneDescription().separation(true).nodes(
                {NodeDescription().referenceAngle(0.0f), NodeDescription().referenceAngle(45.0f), NodeDescription().referenceAngle(0.0f)}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(2, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(1);
    auto newCell = actualData.getOtherObjectRef({0, 1});

    EXPECT_EQ(CellState_Constructing, newCell.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(1, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(0, 1));
    EXPECT_TRUE(actualData.hasConnection(0, newCell._id));
    EXPECT_TRUE(approxCompare(180.0f + 45.0f, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(2, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_2_4__concatenation_0_1__branch_0_0__numAdditionalConnections_0)
{
    Description data;
    data.addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(2).lastConstructedCellId(2))),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription(),
                NodeDescription().referenceAngle(0),
                NodeDescription().referenceAngle(45.0f).numAdditionalConnections(0),
                NodeDescription(),
            }),
        }));
    data.addCreature({
            ObjectDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 99.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(2).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(1), GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription(),
                NodeDescription().referenceAngle(0),
                NodeDescription().referenceAngle(45.0f).numAdditionalConnections(0),
                NodeDescription(),
            }),
        }));
    data.addConnection(1, 2);
    data.addConnection(2, 0);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto prevPrevCell = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(2);
    auto newCell = actualData.getOtherObjectRef({0, 1, 2});

    EXPECT_EQ(CellState_Constructing, newCell.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(prevPrevCell, prevCell));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_EQ(1, prevPrevCell._connections.size());
    EXPECT_EQ(2, prevCell._connections.size());
    EXPECT_EQ(2, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());
    EXPECT_TRUE(approxCompare(180.0f + 45.0f, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(3, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

class ConstructorTests_AllAngleAlignments
    : public ConstructorTests
    , public testing::WithParamInterface<ConstructorAngleAlignment>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_AllAngleAlignments,
    ConstructorTests_AllAngleAlignments,
    ::testing::Values(
        ConstructorAngleAlignment_None,
        ConstructorAngleAlignment_180,
        ConstructorAngleAlignment_120,
        ConstructorAngleAlignment_90,
        ConstructorAngleAlignment_72,
        ConstructorAngleAlignment_60));

TEST_P(ConstructorTests_AllAngleAlignments, creature_1__node_2_4__concatenation_0_1__branch_0_0__numAdditionalConnections_1__angleAlignment)
{
    auto const NodeAngle = 5.0f;

    auto angleAlignment = GetParam();

    Description data;
    data.addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(2).lastConstructedCellId(2))),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription()
                .separation(true)
                .nodes({
                    NodeDescription(),
                    NodeDescription().referenceAngle(0),
                    NodeDescription().referenceAngle(NodeAngle).numAdditionalConnections(1),
                    NodeDescription(),
                })
                .angleAlignment(angleAlignment),
        }));
    data.addCreature({
            ObjectDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 99.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(2).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(1), GenomeDescription().genes({
            GeneDescription()
                .separation(true)
                .nodes({
                    NodeDescription(),
                    NodeDescription().referenceAngle(0),
                    NodeDescription().referenceAngle(NodeAngle).numAdditionalConnections(1),
                    NodeDescription(),
                })
                .angleAlignment(angleAlignment),
        }));
    data.addConnection(1, 2);
    data.addConnection(2, 0);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto prevPrevCell = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(2);
    auto newCell = actualData.getOtherObjectRef({0, 1, 2});

    EXPECT_EQ(CellState_Constructing, newCell.getCellRef()._cellState);
    if (angleAlignment != ConstructorAngleAlignment_180) {
        EXPECT_TRUE(actualData.hasConnection(prevPrevCell, prevCell));
        EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
        EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
        EXPECT_TRUE(actualData.hasConnection(newCell, prevPrevCell));
        EXPECT_EQ(2, prevPrevCell._connections.size());
        EXPECT_EQ(2, prevCell._connections.size());
        EXPECT_EQ(3, newCell._connections.size());
        EXPECT_EQ(1, hostCell._connections.size());

        auto refAngle1 = 0.0f;
        auto refAngle2 = 0.0f;
        auto refAngle3 = 0.0f;
        switch (angleAlignment) {
        case ConstructorAngleAlignment_None: {
            refAngle1 = 135.0f + NodeAngle;
            refAngle2 = 45.0f;
            refAngle3 = 180.0f - NodeAngle;
        } break;
        case ConstructorAngleAlignment_120: {
            refAngle1 = 180.0 - 120.0f + NodeAngle;
            refAngle2 = 120.0f;
            refAngle3 = 180.0f - NodeAngle;
        } break;
        case ConstructorAngleAlignment_90: {
            refAngle1 = 180.0 - 90.0f + NodeAngle;
            refAngle2 = 90.0f;
            refAngle3 = 180.0f - NodeAngle;
        } break;
        case ConstructorAngleAlignment_72: {
            refAngle1 = 180.0f - 72.0f + NodeAngle;
            refAngle2 = 72.0f;
            refAngle3 = 180.0f - NodeAngle;
        } break;
        case ConstructorAngleAlignment_60: {
            refAngle1 = 180.0f - 60.0f + NodeAngle;
            refAngle2 = 60.0f;
            refAngle3 = 180.0f - NodeAngle;
        } break;
        }
        EXPECT_TRUE(approxCompare(refAngle1, actualData.getConnection(newCell, hostCell)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(refAngle2, actualData.getConnection(newCell, prevPrevCell)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(refAngle3, actualData.getConnection(newCell, prevCell)._angleFromPrevious));
    } else {
        EXPECT_TRUE(actualData.hasConnection(prevPrevCell, prevCell));
        EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
        EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
        EXPECT_EQ(1, prevPrevCell._connections.size());
        EXPECT_EQ(2, prevCell._connections.size());
        EXPECT_EQ(2, newCell._connections.size());
        EXPECT_EQ(1, hostCell._connections.size());
        EXPECT_TRUE(approxCompare(180.0f + NodeAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));
    }

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(3, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_1_3__branch_0_1__concatenationAngle)
{
    auto const ConcatenationAngle = 20.0f;

    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription().referenceAngle(ConcatenationAngle)}).numConcatenations(3).separation(false).numBranches(1),
    });
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentConcatenation(1).currentNodeIndex(0).lastConstructedCellId(1))),
            ObjectDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(1);
    auto newCell = actualData.getOtherObjectRef({0, 1});

    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_EQ(1, prevCell._connections.size());
    EXPECT_EQ(2, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());
    EXPECT_TRUE(approxCompare(180.0f + ConcatenationAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(2, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_4__concatenation_1_2__branch_0_1__numAdditionalConnections_1)
{
    auto genome = GenomeDescription().genes({
        GeneDescription()
            .nodes({
                NodeDescription().referenceAngle(-90.0f).numAdditionalConnections(1),
                NodeDescription().referenceAngle(-90.0f),
                NodeDescription().referenceAngle(90.0f),
                NodeDescription().referenceAngle(90.0f),
            })
            .numConcatenations(2)
            .separation(false)
            .numBranches(1)
            .angleAlignment(ConstructorAngleAlignment_90),
    });
    auto addDistance = _parameters.constructorAdditionalOffspringDistance;
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentConcatenation(1).currentNodeIndex(0).lastConstructedCellId(4))),
            ObjectDescription().id(1).pos({100.0f + addDistance, 98.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(2).pos({100.0f + addDistance, 99.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos({101.0f + addDistance, 99.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(4).pos({101.0f + addDistance, 100.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(0, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(6, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(4);
    auto prevPrevPrevCell = actualData.getObjectRef(2);
    auto newCell = actualData.getOtherObjectRef({0, 1, 2, 3, 4});

    EXPECT_EQ(CellState_Constructing, newCell.getCellRef()._cellState);
    EXPECT_FALSE(newCell.getCellRef()._headCell);
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, prevPrevPrevCell));
    EXPECT_EQ(2, prevCell._connections.size());
    EXPECT_EQ(3, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());
    EXPECT_EQ(3, prevPrevPrevCell._connections.size());
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(newCell, prevCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(newCell, prevPrevPrevCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_bothSidesPresent)
{
    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).lastConstructedCellId(2))),
            ObjectDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos({10.0f + getOffspringDistance(), 9.5f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(4).pos({10.0f + getOffspringDistance(), 10.5f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).nodes(
                {NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(1)}),
        }));

    data.addConnection(1, 2);
    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f + 180.0f);
    data.addConnection(2, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(hostCreature._id).size());

    auto actualHostCell = actualData.getObjectRef(1);
    auto actualPrevConstructedCell = actualData.getObjectRef(2);
    auto actualUpperConstructedCell = actualData.getObjectRef(3);
    auto actualLowerConstructedCell = actualData.getObjectRef(4);
    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(3, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualUpperConstructedCell._connections.size());
    ASSERT_EQ(1, actualLowerConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, actualData.getConnection(actualConstructedCell, actualUpperConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(120.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevConstructedCell, actualUpperConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualPrevConstructedCell, actualLowerConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, actualData.getConnection(actualUpperConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualUpperConstructedCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualLowerConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_2__threeCellsWithSmallAngles__variant_1)
{
    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).lastConstructedCellId(2))),
            ObjectDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(4).pos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(false).nodes({NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(2)}),
        }));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.getConnectionRef(2, 3)._angleFromPrevious = 60.0f;
    data.getConnectionRef(2, 1)._angleFromPrevious = 300.0f;
    data.getConnectionRef(3, 4)._angleFromPrevious = 90.0f;
    data.getConnectionRef(3, 2)._angleFromPrevious = 270.0f;


    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto actualHostCell = actualData.getObjectRef(1);
    auto actualPrevConstructedCell = actualData.getObjectRef(2);
    auto actualPrevPrevConstructedCell = actualData.getObjectRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getObjectRef(4);
    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(4, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(3, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(240.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(300.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_2__threeCellsWithSmallAngles__variant_2)
{
    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({458.20f, 239.23f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).lastConstructedCellId(2))),
            ObjectDescription().id(2).pos({456.40f, 238.88f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos({455.96f, 239.75f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(4).pos({456.07f, 240.77f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(false).nodes({NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(2)}),
        }));
    auto const& object1 = data.getObjectRef(1);
    auto const& object2 = data.getObjectRef(2);
    auto const& cell3 = data.getObjectRef(3);

    data.addConnection(1, 2);
    auto cell3_refPos = object2._pos + Math::rotateClockwise(object1._pos - object2._pos, 120.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = cell3._pos + Math::rotateClockwise(object2._pos - cell3._pos, 120.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto actualHostCell = actualData.getObjectRef(1);
    auto actualPrevConstructedCell = actualData.getObjectRef(2);
    auto actualPrevPrevConstructedCell = actualData.getObjectRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getObjectRef(4);
    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(4, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(3, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(240.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, actualData.getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(240.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(300.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_1__threeCellsWithSmallAngles)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(1)}),
    });

    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2))),
            ObjectDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(4).pos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    auto cell3_refPos = data.getObjectRef(2)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = data.getObjectRef(3)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(creature._id).size());

    auto actualHostCell = actualData.getObjectRef(1);
    auto actualPrevConstructedCell = actualData.getObjectRef(2);
    auto origPrevPrevConstructedCell = data.getObjectRef(3);
    auto actualPrevPrevConstructedCell = actualData.getObjectRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getObjectRef(4);
    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, actualData.getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_EQ(origPrevPrevConstructedCell._connections, actualPrevPrevConstructedCell._connections);

    EXPECT_TRUE(approxCompare(150.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(210.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_1__90degAlignment)
{
    auto genome = GenomeDescription().genes({
        GeneDescription()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_90)
            .nodes({NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(1)}),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2))),
            ObjectDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(4).pos({10.0f + getOffspringDistance() - 1.0f, 9.0f - 0.2f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = data.getObjectRef(3)._pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(creature._id).size());

    auto actualHostCell = actualData.getObjectRef(1);
    auto actualPrevConstructedCell = actualData.getObjectRef(2);
    auto actualPrevPrevConstructedCell = actualData.getObjectRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getObjectRef(4);
    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(270.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_3__node_2_3__concatenation_0_1__branch_0_1__numAdditionalConnections_0__90degAlignment)
{
    auto genome = GenomeDescription().genes({
        GeneDescription()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_90)
            .nodes({NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(0)}),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(2).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2))),
            ObjectDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(creature._id).size());

    auto actualHostCell = actualData.getObjectRef(1);
    auto actualPrevConstructedCell = actualData.getObjectRef(2);
    auto actualPrevPrevConstructedCell = actualData.getObjectRef(3);
    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(2, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(1, actualPrevPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_1__90degAlignment__connectToCellWithAngleSpace)
{
    auto genome = GenomeDescription().genes({
        GeneDescription()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_90)
            .nodes({NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(1)}),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2))),
            ObjectDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos({10.0f + getOffspringDistance(), 10.0f - 0.5f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(4).pos({10.0f + getOffspringDistance(), 10.0f - 1.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = data.getObjectRef(3)._pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(creature._id).size());

    auto actualHostCell = actualData.getObjectRef(1);
    auto actualPrevConstructedCell = actualData.getObjectRef(2);
    auto actualPrevPrevConstructedCell = actualData.getObjectRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getObjectRef(4);
    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3, 4});

    ASSERT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(270.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(270.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_inf__branch_0_0)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).numConcatenations(GeneDescription::NumConcatenations_Infinite).nodes({NodeDescription()}),
    });
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(0))),
        }, CreatureDescription().id(0), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto newCell = actualData.getOtherObjectRef({0});

    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_1_inf__branch_0_0)
{
    Description data;
    data.addCreature({
            ObjectDescription().id(0).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(0).currentConcatenation(1).lastConstructedCellId(1))),
        }, CreatureDescription().id(0), GenomeDescription().genes({
            GeneDescription().separation(true).numConcatenations(GeneDescription::NumConcatenations_Infinite).nodes({NodeDescription()}),
        }));
    data.addCreature({
            ObjectDescription().id(1).pos({101.0f + _parameters.constructorAdditionalOffspringDistance, 100.0f}).type(CellDescription().cellState(CellState_Ready)),
        }, CreatureDescription().id(1), GenomeDescription().genes({
            GeneDescription().separation(true).numConcatenations(GeneDescription::NumConcatenations_Infinite).nodes({NodeDescription()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostCell = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(1);
    auto newCell = actualData.getOtherObjectRef({0, 1});

    EXPECT_EQ(CellState_Activating, newCell.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, prevCell));
    EXPECT_EQ(1, prevCell._connections.size());
    EXPECT_EQ(2, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());

    auto hostConstructor = std::get<ConstructorDescription>(hostCell.getCellRef()._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(2, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1__largeConstructionAngle)
{
    auto const ConstructionAngle = 180.0f;

    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription()}).separation(false).numBranches(1),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 99.0f}),
            ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().constructionAngle(ConstructionAngle).geneIndex(0).currentNodeIndex(0).autoTriggerInterval(100))),
            ObjectDescription().id(2).pos({100.1f, 101.0f}),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(hostCreature._id).size());

    auto const& hostCell = actualData.getObjectRef(1);
    auto const& newCell = actualData.getOtherObjectRef({0, 1, 2});

    auto angleSpan_cell2_cell0 = hostCell.getAngleSpan(2, 0);
    auto angleSpan_lastCell_and_cell0 = hostCell.getAngleSpan(newCell._id, 0);
    EXPECT_TRUE(approxCompare(Math::getNormalizedAngle(angleSpan_lastCell_and_cell0 + ConstructionAngle, 0.0f), angleSpan_cell2_cell0 / 2));
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1__frontAngle_leftSide)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription()}),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}),
            ObjectDescription().id(2).pos({9.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(0).autoTriggerInterval(1).geneIndex(0))),
            ObjectDescription().id(3).pos({9.0f, 11.0f}),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(creature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3});

    EXPECT_EQ(CellState_Ready, actualConstructedCell.getCellRef()._cellState);
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1__frontAngle_rightSide)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription()}),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({8.0f, 10.0f}),
            ObjectDescription().id(2).pos({9.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().currentNodeIndex(0).autoTriggerInterval(1).geneIndex(0))),
            ObjectDescription().id(3).pos({9.0f, 11.0f}),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(creature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3});

    EXPECT_EQ(CellState_Ready, actualConstructedCell.getCellRef()._cellState);
}

class ConstructorTests_AllShapes
    : public ConstructorTests
    , public testing::WithParamInterface<ConstructorShape>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_AllShapes,
    ConstructorTests_AllShapes,
    ::testing::Values(
        ConstructorShape_Segment,
        ConstructorShape_Triangle,
        ConstructorShape_Rectangle,
        ConstructorShape_Hexagon,
        ConstructorShape_Loop,
        ConstructorShape_Tube,
        ConstructorShape_Lolli,
        ConstructorShape_SmallLolli,
        ConstructorShape_Zigzag));

TEST_P(ConstructorTests_AllShapes, creature_3__generateShape)
{
    _parameters.friction.baseValue = 0.05f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto const ConstructionAngle = 8.0f;
    auto const LastAngle = -5.0f;
    auto const n = 20;

    auto shape = GetParam();

    auto gene = GeneDescription().separation(false).numBranches(1).shape(shape);
    gene._nodes.emplace_back(NodeDescription());
    for (int i = 0; i < n - 2; ++i) {
        gene._nodes.emplace_back(NodeDescription());
    }
    gene._nodes.emplace_back(NodeDescription().referenceAngle(LastAngle));
    auto genome = GenomeDescription().genes({gene});

    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({100.0f, 99.0f}),
            ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().usableEnergy(getConstructorEnergy() * n).cellType(ConstructorDescription().constructionAngle(ConstructionAngle).geneIndex(0).currentNodeIndex(0).autoTriggerInterval(200))),
            ObjectDescription().id(2).pos({100.1f, 101.0f}),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    // Construct offspring and record ids of constructed cells
    std::vector<uint64_t> createdCellIds;
    {
        for (int i = 0; i < n; ++i) {
            _simulationFacade->calcTimesteps(200);
            auto actualData = _simulationFacade->getSimulationData();

            ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
            ASSERT_EQ(1, actualData._creatures.size());
            EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

            auto hostCreature = actualData.getCreatureRef(0);
            ASSERT_EQ(3 + i + 1, actualData.getObjectsForCreature(hostCreature._id).size());

            auto hostCell = actualData.getObjectRef(1);

            std::set<uint64_t> knownCellIds(createdCellIds.begin(), createdCellIds.end());
            knownCellIds.insert(0);
            knownCellIds.insert(1);
            knownCellIds.insert(2);
            auto newCell = actualData.getOtherObjectRef(knownCellIds);
            createdCellIds.emplace_back(newCell._id);

            if (i < n - 1) {
                EXPECT_EQ(CellState_Constructing, newCell.getCellRef()._cellState);
            } else {
                EXPECT_EQ(CellState_Ready, newCell.getCellRef()._cellState);
            }
            EXPECT_TRUE(actualData.hasConnection(hostCell, newCell));
        }
    }

    // Check angles except for first and last node
    auto actualData = _simulationFacade->getSimulationData();
    auto shapeGenerator = ShapeGeneratorFactory::create(shape);
    for (int i = 0; i < n; ++i) {
        auto shapeResult = shapeGenerator->generateNextConstructionData();
        if (i > 0 && i < n - 1) {
            auto const& object = actualData.getObjectRef(createdCellIds.at(i));
            auto prevCellId = createdCellIds.at(i - 1);
            auto nextObjectId = createdCellIds.at(i + 1);
            auto angle = object.getAngleSpan(prevCellId, nextObjectId);
            angle = Math::getNormalizedAngle(angle - 180.0f, -180.0f);
            EXPECT_EQ(shapeResult.angle, angle);
            int numPrevConnections = 0;
            for (auto const& connection : object._connections) {
                if (connection._objectId < object._id) {
                    ++numPrevConnections;
                }
            }
            EXPECT_EQ(shapeResult.numAdditionalConnections, numPrevConnections - 1);
        }
    }

    // Check angles for first node
    {
        auto const& hostCell = actualData.getObjectRef(1);
        auto angleSpan_cell2_cell0 = hostCell.getAngleSpan(2, 0);
        auto angleSpan_lastCell_and_cell0 = hostCell._connections.at(0)._angleFromPrevious;
        EXPECT_TRUE(approxCompare(angleSpan_lastCell_and_cell0 + ConstructionAngle, angleSpan_cell2_cell0 / 2));
    }

    // Check angles for last node
    {
        auto const& object = actualData.getObjectRef(createdCellIds.back());
        auto prevCellId = createdCellIds.at(n - 2);
        auto nextObjectId = 1;  // = id of hostCell
        auto angle = object.getAngleSpan(prevCellId, nextObjectId);
        angle = Math::getNormalizedAngle(angle - 180.0f, -180.0f);
        EXPECT_EQ(LastAngle, angle);
    }
}

TEST_F(ConstructorTests, avoidDeadlockByLockingNearObjects)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({}),
        GeneDescription().separation(false).shape(ConstructorShape_Hexagon).nodes({NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription()}),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}),
            ObjectDescription().id(2).pos({10.0f, 9.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(1).currentNodeIndex(2).lastConstructedCellId(6))),
            ObjectDescription().id(3).pos({11.0f, 9.0f}),
            ObjectDescription().id(4).pos({11.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(1).currentNodeIndex(2).lastConstructedCellId(8))),
            ObjectDescription().id(5).pos({10.0f, 9.0f - getOffspringDistance() - 1.0f}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(0).geneIndex(1)),
            ObjectDescription().id(6).pos({10.0f, 9.0f - getOffspringDistance()}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(1).geneIndex(1)),
            ObjectDescription().id(7).pos({11.0f + getOffspringDistance() + 1.0f, 10.0f}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(0).geneIndex(1)),
            ObjectDescription().id(8).pos({11.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(1).geneIndex(1)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 1);

    data.addConnection(5, 6);
    data.addConnection(6, 2);

    data.addConnection(7, 8);
    data.addConnection(8, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(10, actualData.getObjectsForCreature(creature._id).size());
}

TEST_F(ConstructorTests, avoidConnectionsBetweenDifferentConstructions)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({}),
        GeneDescription().separation(false).shape(ConstructorShape_Hexagon).nodes({NodeDescription(), NodeDescription(), NodeDescription()}),
        GeneDescription().separation(false).shape(ConstructorShape_Hexagon).nodes({NodeDescription(), NodeDescription(), NodeDescription()}),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).nodeIndex(0).cellType(ConstructorDescription().geneIndex(1).currentNodeIndex(2).lastConstructedCellId(4))),
            ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().usableEnergy(getConstructorEnergy()).nodeIndex(1).cellType(ConstructorDescription().geneIndex(2).currentNodeIndex(2).lastConstructedCellId(6))),

            ObjectDescription().id(3).pos({10.0f, 10.0f - getOffspringDistance() - 1.0f}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(0).geneIndex(1).parentNodeIndex(0)),
            ObjectDescription().id(4).pos({10.0f, 10.0f - getOffspringDistance()}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(1).geneIndex(1).parentNodeIndex(0)),
            ObjectDescription().id(5).pos({11.0f, 10.0f - getOffspringDistance() - 1.0f}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(0).geneIndex(2).parentNodeIndex(1)),
            ObjectDescription().id(6).pos({11.0f, 10.0f - getOffspringDistance()}).type(CellDescription().cellState(CellState_Constructing).nodeIndex(1).geneIndex(2).parentNodeIndex(1)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);

    data.addConnection(3, 4);
    data.addConnection(4, 1);

    data.addConnection(5, 6);
    data.addConnection(6, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(8, actualData.getObjectsForCreature(creature._id).size());

    auto constructedCells = actualData.getOtherObjects({1, 2, 3, 4, 5, 6});
    ObjectDescription object1, object2;
    for (auto const& object : constructedCells) {
        if (object.getCellRef()._parentNodeIndex == 0) {
            object1 = object;
        } else {
            object2 = object;
        }
    }
    EXPECT_EQ(3, object1._connections.size());
    EXPECT_TRUE(actualData.hasConnection(object1._id, 1));
    EXPECT_TRUE(actualData.hasConnection(object1._id, 3));
    EXPECT_TRUE(actualData.hasConnection(object1._id, 4));

    EXPECT_EQ(3, object2._connections.size());
    EXPECT_TRUE(actualData.hasConnection(object2._id, 2));
    EXPECT_TRUE(actualData.hasConnection(object2._id, 5));
    EXPECT_TRUE(actualData.hasConnection(object2._id, 6));
}

enum class Separation
{
    No,
    Yes
};
class ConstructorTests_ProvideEnergy_Separation
    : public ConstructorTests
    , public testing::WithParamInterface<std::pair<ProvideEnergy, Separation>>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_ProvideEnergy,
    ConstructorTests_ProvideEnergy_Separation,
    ::testing::Values(
        std::make_pair(ProvideEnergy_CellOnly, Separation::No),
        std::make_pair(ProvideEnergy_CellAndGene, Separation::No),
        std::make_pair(ProvideEnergy_FreeGeneration, Separation::No),
        std::make_pair(ProvideEnergy_CellOnly, Separation::Yes),
        std::make_pair(ProvideEnergy_CellAndGene, Separation::Yes),
        std::make_pair(ProvideEnergy_FreeGeneration, Separation::Yes)));

TEST_P(ConstructorTests_ProvideEnergy_Separation, provideEnergy_sufficientEnergy)
{
    auto [provideEnergy, separation] = GetParam();

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
        GeneDescription().separation(separation == Separation::Yes).numBranches(2).numConcatenations(3).nodes({NodeDescription(), NodeDescription()}),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto constructorEnergy = [&] {
        if (provideEnergy == ProvideEnergy_FreeGeneration) {
            return normalCellEnergy;
        }
        if (provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No) {
            return normalCellEnergy * (2 * 3 * 2 + 2) + 1.0f;
        } else {
            return normalCellEnergy * 2 + 1.0f;
        }
    }();
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(constructorEnergy).cellType(
                    ConstructorDescription().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDescription().id(1).pos({10.0f + getOffspringDistance(), 10.0f}),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({0, 1});

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        EXPECT_TRUE(approxCompare(normalCellEnergy, actualData.getObjectRef(0).getCellRef()._usableEnergy));
        auto newConstructor = std::get<ConstructorDescription>(actualConstructedCell.getCellRef()._cellType);
        if (separation == Separation::Yes) {
            EXPECT_EQ(ProvideEnergy_CellOnly, newConstructor._provideEnergy);
        } else {
            EXPECT_EQ(ProvideEnergy_FreeGeneration, newConstructor._provideEnergy);
        }
    } else {
        if (provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No) {
            EXPECT_TRUE(approxCompare(normalCellEnergy * (2 * 3 * 2 + 1), actualConstructedCell.getCellRef()._usableEnergy));
        } else {
            EXPECT_TRUE(approxCompare(normalCellEnergy, actualConstructedCell.getCellRef()._usableEnergy));
        }
    }
}

TEST_P(ConstructorTests_ProvideEnergy_Separation, provideEnergy_insufficientEnergy)
{
    auto [provideEnergy, separation] = GetParam();

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        GTEST_SKIP() << "Skipping test because FreeGeneration always has enough energy.";
    }
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
        GeneDescription().separation(separation == Separation::Yes).numBranches(2).numConcatenations(3).nodes({NodeDescription(), NodeDescription()}),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto constructorEnergy =
        provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No ? normalCellEnergy * (2 * 3 * 2 + 2) - 1.0f : normalCellEnergy * 2 - 1.0f;
    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(constructorEnergy).cellType(
                    ConstructorDescription().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDescription().id(1).pos({10.0f + getOffspringDistance(), 10.0f}),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());
}

TEST_P(ConstructorTests_ProvideEnergy_Separation, provideEnergy_infiniteConcatenations)
{
    auto [provideEnergy, separation] = GetParam();

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
            NodeDescription(),
        }),
        GeneDescription()
            .separation(separation == Separation::Yes)
            .numBranches(2)
            .numConcatenations(std::numeric_limits<int>::max())
            .nodes({NodeDescription(), NodeDescription()}),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto constructorEnergy = [&] {
        if (provideEnergy == ProvideEnergy_FreeGeneration) {
            return normalCellEnergy;
        }
        if (provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No) {
            return normalCellEnergy * (2 + 2) + 1.0f;
        } else {
            return normalCellEnergy * 2 + 1.0f;
        }
    }();


    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(constructorEnergy).cellType(
                    ConstructorDescription().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDescription().id(1).pos({10.0f + getOffspringDistance(), 10.0f}),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({0, 1});

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        EXPECT_TRUE(approxCompare(normalCellEnergy, actualData.getObjectRef(0).getCellRef()._usableEnergy));
    } else {
        if (provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No) {
            EXPECT_TRUE(approxCompare(normalCellEnergy * (2 + 1), actualConstructedCell.getCellRef()._usableEnergy));
        } else {
            EXPECT_TRUE(approxCompare(normalCellEnergy, actualConstructedCell.getCellRef()._usableEnergy));
        }
    }
}

class ConstructorTests_ProvideEnergy
    : public ConstructorTests
    , public testing::WithParamInterface<ProvideEnergy>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_ProvideEnergy,
    ConstructorTests_ProvideEnergy,
    ::testing::Values(
        ProvideEnergy_CellOnly,
        ProvideEnergy_CellAndGene,
        ProvideEnergy_FreeGeneration));

TEST_P(ConstructorTests_ProvideEnergy, provideEnergy_depotWithInitialStoredEnergy_sufficientEnergy)
{
    auto provideEnergy = GetParam();

    auto const InitialStoredUsableEnergy = 50.0f;

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription().cellType(DepotGenomeDescription().initialStoredUsableEnergy(InitialStoredUsableEnergy)),
            NodeDescription(),
        }),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    // Calculate required energy: normalCellEnergy + InitialStoredUsableEnergy for depot cell
    auto constructorEnergy = [&] {
        if (provideEnergy == ProvideEnergy_FreeGeneration) {
            return normalCellEnergy;
        }

        if (provideEnergy == ProvideEnergy_CellAndGene) {
            return normalCellEnergy * (2 + 2) + 1.0f + InitialStoredUsableEnergy;   // Contains energy for all nodes in gene
        } else {
            return normalCellEnergy * 2 + InitialStoredUsableEnergy + 1.0f; // Contains energy for depot node in gene
        }
    }();

    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(constructorEnergy).cellType(
                    ConstructorDescription().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDescription().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

    // For separation, offspring is in a separate creature
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({0, 1});

    // Verify the constructed cell is a depot with stored energy
    EXPECT_EQ(CellType_Depot, actualConstructedCell.getCellRef().getCellType());
    auto const& depot = std::get<DepotDescription>(actualConstructedCell.getCellRef()._cellType);
    EXPECT_TRUE(approxCompare(InitialStoredUsableEnergy, depot._storedUsableEnergy));
    if (provideEnergy != ProvideEnergy_FreeGeneration) {
        EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    }
}

TEST_P(ConstructorTests_ProvideEnergy, provideEnergy_depotWithInitialStoredEnergy_insufficientEnergy)
{
    auto provideEnergy = GetParam();

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        GTEST_SKIP() << "Skipping test because FreeGeneration always has enough energy.";
    }

    auto const InitialStoredUsableEnergy = 50.0f;

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription().cellType(DepotGenomeDescription().initialStoredUsableEnergy(InitialStoredUsableEnergy)),
            NodeDescription(),
        }),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    // Not enough energy: just enough for cell but not for depot initial energy
    auto constructorEnergy = normalCellEnergy * 2 + InitialStoredUsableEnergy - 1.0f;

    auto data = Description().addCreature({
            ObjectDescription().id(0).pos({10.0f, 10.0f}).type(CellDescription().usableEnergy(constructorEnergy).cellType(
                    ConstructorDescription().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDescription().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());  // Host + previously constructed, no new cell

    if (provideEnergy != ProvideEnergy_FreeGeneration) {
        EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    }
}

TEST_F(ConstructorTests, regressionTestMassiveReplicationsWithSeeds)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1))}),
        GeneDescription().separation(false).nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
        }),
    });
    std::vector<ObjectDescription> largeCreatureCells;
    for (int i = 0; i < 50; ++i) {
        largeCreatureCells.emplace_back(
            ObjectDescription().id(i).pos({toFloat(i), 0.0f}).type(CellDescription().cellType(ConstructorDescription().geneIndex(0).autoTriggerInterval(30))));
    }
    Description largeCreatureData;
    largeCreatureData.addCreature(largeCreatureCells, CreatureDescription(), genome);
    for (int i = 1; i < 50; ++i) {
        largeCreatureData.addConnection(i, i - 1);
    }
    auto largeData = Description();
    for (int i = 0; i < 10; ++i) {
        auto clone = largeCreatureData;
        DescriptionEditService::get().setCenter(clone, {100.0f, toFloat(i) * 20});
        largeData.add(std::move(clone));
    }

    _parameters.externalEnergyControlToggle.value = true;
    _parameters.externalEnergy.value = 1e7f;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(largeData);
    _simulationFacade->calcTimesteps(10000);
}

TEST_F(ConstructorTests, angleCorrectionByInnerSumOfPolygon)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).angleAlignment(ConstructorAngleAlignment_90).nodes({
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
            NodeDescription(),
            NodeDescription().numAdditionalConnections(1),
        }),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(0).pos(RealVector2D{10.0f, 8.5f} + Math::unitVectorOfAngle(180.0f - 45.0f) * getOffspringDistance()).type(CellDescription().cellType(ConstructorDescription().currentNodeIndex(4).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(4).provideEnergy(
                    ProvideEnergy_FreeGeneration))),

            ObjectDescription().id(1).pos({10.0f, 11.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(2).pos({10.0f, 9.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos({9.0f, 9.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(4).pos({10.0f, 8.5f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 0);
    data.getConnectionRef(3, 4)._angleFromPrevious = 270.0f;
    data.getConnectionRef(3, 2)._angleFromPrevious = 90.0f;
    data.getConnectionRef(4, 0)._angleFromPrevious = 270.0f;
    data.getConnectionRef(4, 3)._angleFromPrevious = 90.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(6, actualData.getObjectsForCreature(creature._id).size());

    auto object = [&](uint64_t id) { return actualData.getObjectRef(id); };
    auto constructedCell = actualData.getOtherObjectRef({0, 1, 2, 3, 4});

    EXPECT_EQ(1, object(1)._connections.size());
    EXPECT_EQ(3, object(2)._connections.size());
    EXPECT_EQ(2, object(3)._connections.size());
    EXPECT_EQ(2, object(4)._connections.size());
    EXPECT_EQ(3, constructedCell._connections.size());
    EXPECT_EQ(1, object(0)._connections.size());

    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnectionRef(2, constructedCell._id)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnectionRef(2, 1)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnectionRef(2, 3)._angleFromPrevious));
}

TEST_F(ConstructorTests, angleCorrectionByInnerSumOfPolygon_mirrored)
{
    auto genome = GenomeDescription().genes({
        GeneDescription()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_90)
            .nodes({
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription().numAdditionalConnections(1),
            }),
    });

    auto data = Description().addCreature({
            ObjectDescription().id(0).pos(RealVector2D{10.0f, 8.5f} + Math::unitVectorOfAngle(180.0f + 45.0f) * getOffspringDistance()).type(CellDescription().cellType(ConstructorDescription().currentNodeIndex(4).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(4).provideEnergy(
                    ProvideEnergy_FreeGeneration))),

            ObjectDescription().id(1).pos({10.0f, 11.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(2).pos({10.0f, 9.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(3).pos({11.0f, 9.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(4).pos({10.0f, 8.5f}).type(CellDescription().cellState(CellState_Constructing)),
        }, CreatureDescription().id(0), genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 0);
    data.getConnectionRef(3, 4)._angleFromPrevious = 90.0f;
    data.getConnectionRef(3, 2)._angleFromPrevious = 270.0f;
    data.getConnectionRef(4, 0)._angleFromPrevious = 90.0f;
    data.getConnectionRef(4, 3)._angleFromPrevious = 270.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(6, actualData.getObjectsForCreature(creature._id).size());

    auto object = [&](uint64_t id) { return actualData.getObjectRef(id); };
    auto constructedCell = actualData.getOtherObjectRef({0, 1, 2, 3, 4});

    EXPECT_EQ(1, object(1)._connections.size());
    EXPECT_EQ(3, object(2)._connections.size());
    EXPECT_EQ(2, object(3)._connections.size());
    EXPECT_EQ(2, object(4)._connections.size());
    EXPECT_EQ(3, constructedCell._connections.size());
    EXPECT_EQ(1, object(0)._connections.size());

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnectionRef(2, constructedCell._id)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnectionRef(2, 1)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnectionRef(2, 3)._angleFromPrevious));
}
