#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/ShapeGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescTestDataFactory.h>

#include "IntegrationTestFramework.h"

#include "PersisterInterface/SerializerService.h"

class ConstructorTests : public IntegrationTestFramework
{
public:
    ConstructorTests()
        : IntegrationTestFramework()
    {
        _descTestDataFactory = &DescTestDataFactory::get();
    }

    ~ConstructorTests() = default;

protected:
    float getConstructorEnergy() const
    {
        return _parameters.normalCellEnergy.value[0] * 3.5f;  // Provide enough energy
    }
    float getOffspringDistance() const { return 1.0f; }

    DescTestDataFactory* _descTestDataFactory;
};

TEST_F(ConstructorTests, alreadyFinished)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentBranch(1))),
            ObjectDesc().id(1).pos({100.0f, 101.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(false).numBranches(1).nodes({NodeDesc()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
    EXPECT_EQ(0, hostConstructor._currentOffspring);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, emptyGenome)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentBranch(0))),
        },
        CreatureDesc().id(0),
        GenomeDesc());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, emptyGene)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentBranch(0))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(true)}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, nodeIndexOutOfRange)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentBranch(0).currentNodeIndex(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(true)}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, geneIndexOutOfRange)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(1).currentBranch(0).currentNodeIndex(0))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(true)}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, insufficientEnergy)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().constructor(ConstructorDesc().geneIndex(0).currentBranch(0).currentNodeIndex(0))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(true).nodes({NodeDesc()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, manuallyTriggered_withSignal_failed)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().constructor(
                    ConstructorDesc().autoTriggerInterval(std::nullopt).geneIndex(0).currentBranch(0).currentNodeIndex(0))),  // Not enough energy
            ObjectDesc().id(1).pos({101.0f, 100.0f}).type(CellDesc().signal({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(true).nodes({NodeDesc()})}));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(2, creature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[Channels::ConstructorSuccess]));
}

TEST_F(ConstructorTests, manuallyTriggered_withSignal_success)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().autoTriggerInterval(std::nullopt).geneIndex(0).currentBranch(0).currentNodeIndex(0))),
            ObjectDesc().id(1).pos({101.0f, 100.0f}).type(CellDesc().signal({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(false).nodes({NodeDesc()})}));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(3, creature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
    EXPECT_TRUE(approxCompare(1.0f, hostObject.getCellRef()._signal._channels[Channels::ConstructorSuccess]));
}

TEST_F(ConstructorTests, manuallyTriggered_withoutSignal)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().autoTriggerInterval(std::nullopt).geneIndex(0).currentBranch(0).currentNodeIndex(0))),
            ObjectDesc().id(1).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(false).nodes({NodeDesc()})}));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(2, creature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, lastConstructedCellNotFound)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentBranch(0).lastConstructedCellId(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(false).numBranches(1).nodes({NodeDesc()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(2, hostCreature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

// Restriction to SimulationParameters::minObjectDistance has been removed
TEST_F(ConstructorTests, DISABLED_insufficientSpace)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().geneIndex(0).currentConcatenation(0).currentNodeIndex(1).lastConstructedCellId(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc()}),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({100.5f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(1, hostCreature._numObjects);

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    ASSERT_EQ(1, newCreature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, crossingLinks)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(10)
                .pos({10.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().currentNodeIndex(3).lastConstructedCellId(3))),
            ObjectDesc().id(3).pos({10.0f, 10.0f + 1.8f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(2)),
            ObjectDesc().id(2).pos({9.0f, 9.0f + 1.8f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(1)),
            ObjectDesc().id(1).pos({11.0f, 9.0f + 1.8f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(3, 10);
    data.addConnection(2, 3);
    data.addConnection(1, 2);
    data.addConnection(3, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    // No cell constructed
    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(creature._id).size());
}

using NodeParameter = DescTestDataFactory::NodeParameter;
class ConstructorTests_AllNodeTypes
    : public ConstructorTests
    , public testing::WithParamInterface<NodeParameter>
{};

INSTANTIATE_TEST_SUITE_P(ConstructorTests_AllNodeTypes, ConstructorTests_AllNodeTypes, ::testing::ValuesIn(DescTestDataFactory::get().getAllNodeParameters()));

TEST_P(ConstructorTests_AllNodeTypes, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0)
{
    auto nodeParameter = GetParam();
    auto constexpr FrontAngleId = 5;

    auto randomNode = _descTestDataFactory->createNonDefaultNodeDesc(nodeParameter);

    auto data = Desc().addCreature(
        {ObjectDesc().pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc()).frontAngleId(FrontAngleId))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({randomNode}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_EQ(FrontAngleId, newObject.getCellRef()._frontAngleId);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));
    EXPECT_TRUE(_descTestDataFactory->compare(newObject, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostObject._id, newObject._id));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_P(ConstructorTests_AllNodeTypes, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0__preview)
{
    auto nodeParameter = GetParam();
    auto randomNode = _descTestDataFactory->createNonDefaultNodeDesc(nodeParameter);

    auto data = Desc().addCreature(
        {ObjectDesc().pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc()))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({randomNode}),
        }));

    _simulationFacade->setPreviewData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctionsForPreview();

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

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(Math::length(hostObject._pos - newObject._pos) > 50.0f);  // Preview specific: Move seed far away from construction
    EXPECT_TRUE(_descTestDataFactory->compare(newObject, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostObject._id, newObject._id));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0__preview_detail)
{
    auto randomNode = _descTestDataFactory->createNonDefaultNodeDesc(NodeParameter{CellType_Base});

    auto data = Desc().addCreature(
        {ObjectDesc().pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc()))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({randomNode}),
        }));

    _simulationFacade->setPreviewData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctionsForPreview(true);

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

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(Math::length(hostObject._pos - newObject._pos) > 50.0f);  // Preview specific: Move seed far away from construction
    EXPECT_TRUE(_descTestDataFactory->compare(newObject, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostObject._id, newObject._id));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_1)
{
    auto data = Desc().addCreature(
        {ObjectDesc().pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(1)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true),
            GeneDesc().separation(true).nodes({NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));
    EXPECT_FALSE(actualData.hasConnection(hostObject._id, newObject._id));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
    EXPECT_EQ(1, hostConstructor._currentOffspring);
}

TEST_F(ConstructorTests, creature_1__node_2_3__concatenation_0_1__branch_0_0__frontAngle_upperSide)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc(), NodeDesc().constructor(ConstructorGenomeDesc())}),
    });
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({10.0f, 10.0f})
                                .type(CellDesc()
                                          .usableEnergy(getConstructorEnergy())
                                          .constructor(ConstructorDesc().currentNodeIndex(2).geneIndex(0).lastConstructedCellId(2))),
                        },
                        CreatureDesc().id(0),
                        genome)
                    .addCreature(
                        {
                            ObjectDesc().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
                            ObjectDesc().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).type(CellDesc().cellState(CellState_Constructing)),
                        },
                        CreatureDesc().id(1),
                        genome);
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();
    _simulationFacade->setCurrentTimestep(3);
    for (int i = 0; i < 3; ++i) {
        _simulationFacade->testOnly_calcTimestepWithCellFunctions();
    }

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
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc(), NodeDesc().constructor(ConstructorGenomeDesc())}),
    });
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({10.0f, 10.0f})
                                .type(CellDesc()
                                          .usableEnergy(getConstructorEnergy())
                                          .constructor(ConstructorDesc().currentNodeIndex(2).geneIndex(0).lastConstructedCellId(2))),
                        },
                        CreatureDesc().id(0),
                        genome)
                    .addCreature(
                        {
                            ObjectDesc().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
                            ObjectDesc().id(3).pos({10.0f + getOffspringDistance(), 11.0f}).type(CellDesc().cellState(CellState_Constructing)),
                        },
                        CreatureDesc().id(1),
                        genome);
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();
    _simulationFacade->setCurrentTimestep(3);
    for (int i = 0; i < 3; ++i) {
        _simulationFacade->testOnly_calcTimestepWithCellFunctions();
    }

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
    auto data = Desc().addCreature(
        {ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(false).numBranches(1).nodes({NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(2, hostCreature._numObjects);

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef(0);
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
    EXPECT_EQ(0, hostConstructor._currentOffspring);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_1__gene_1)
{
    auto data = Desc().addCreature(
        {ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(1)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true),
            GeneDesc().separation(false).numBranches(1).nodes({NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef(0);
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_FALSE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_2__concatenation_0_1__branch_0_1)
{
    auto data = Desc().addCreature(
        {ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(false).numBranches(1).numConcatenations(1).nodes({NodeDesc(), NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef(0);
    EXPECT_EQ(CellState_Constructing, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_2__branch_0_1)
{
    auto data = Desc().addCreature(
        {ObjectDesc()
             .id(0)
             .pos({100.0f, 100.0f})
             .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentConcatenation(0)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(false).numBranches(1).numConcatenations(2).nodes({NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef(0);
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
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

    auto muscleMode = [&muscleModeType] -> MuscleModeDesc {
        if (muscleModeType == MuscleMode_AutoBending)
            return AutoBendingDesc().initialAngle(90.0f);
        else if (muscleModeType == MuscleMode_ManualBending)
            return ManualBendingDesc().initialAngle(90.0f);
        else
            return AngleBendingDesc().initialAngle(90.0f);
    }();

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}),
            ObjectDesc()
                .id(1)
                .pos({100.0f, 101.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().geneIndex(0).currentConcatenation(1).lastConstructedCellId(3))),
            ObjectDesc().id(2).pos({100.0f, 102.0f}),
            ObjectDesc().id(3).pos({100.0f + getOffspringDistance(), 101.0f}).type(CellDesc().cellType(MuscleDesc().mode(muscleMode))),
        },
        CreatureDesc().id(0).frontAngleId(InitialFrontAngleId),
        GenomeDesc().genes({
            GeneDesc().separation(false).numBranches(1).numConcatenations(2).nodes({NodeDesc().cellType(MuscleGenomeDesc())}),
        }));
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    EXPECT_EQ(InitialFrontAngleId + 1, hostCreature._frontAngleId);
    ASSERT_EQ(5, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(3);
    auto newObject = actualData.getOtherObjectRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(hostObject._pos + RealVector2D(0.5f, 0.0f), newObject._pos));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);

    auto muscle = std::get<MuscleDesc>(prevCell.getCellRef()._cellType);
    if (muscleModeType == MuscleMode_AutoBending) {
        EXPECT_FALSE(std::get<AutoBendingDesc>(muscle._mode)._initialAngle.has_value());
    } else if (muscleModeType == MuscleMode_ManualBending) {
        EXPECT_FALSE(std::get<ManualBendingDesc>(muscle._mode)._initialAngle.has_value());
    } else {
        EXPECT_FALSE(std::get<AngleBendingDesc>(muscle._mode)._initialAngle.has_value());
    }
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_0_2)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numBranches(2).nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentBranch(0))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef({0});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_1_2)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numBranches(2).nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentBranch(1))),
            ObjectDesc().id(1).pos({100.0f, 101.0f}),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef({0, 1});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(hostObject._pos - RealVector2D(0.0f, 0.5f), newObject._pos));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(2, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_1_2__firstBranchMissing)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numBranches(2).nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentBranch(1))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef({0});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(2, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__ignoreNumAdditionalConnectionsAtStart)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({
            NodeDesc().referenceAngle(0).numAdditionalConnections(1),
        }),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(0))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef({0});

    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_FALSE(actualData.hasConnection(newObject, hostObject));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_0_0)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0))),
            ObjectDesc().id(1).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({NodeDesc()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(hostObject._pos - RealVector2D(0.5f, 0.0f), newObject._pos));
    EXPECT_FALSE(actualData.hasConnection(0, newObject._id));
    EXPECT_FALSE(actualData.hasConnection(1, newObject._id));
    EXPECT_TRUE(actualData.hasConnection(0, 1));
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_0_1)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0))),
            ObjectDesc().id(1).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(false).numBranches(1).nodes({NodeDesc()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef({0, 1});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(hostObject._pos - RealVector2D(0.5f, 0.0f), newObject._pos));
    EXPECT_TRUE(actualData.hasConnection(0, newObject._id));
    EXPECT_FALSE(actualData.hasConnection(1, newObject._id));
    EXPECT_TRUE(actualData.hasConnection(0, 1));
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_1_2)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numBranches(2).nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({101.0f, 100.0f}),
            ObjectDesc()
                .id(1)
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().geneIndex(0).currentBranch(1).constructionAngle(
                              10.0f)))  // constructionAngle should be ignored for the second branch
                .pos({100.0f, 100.0f}),
            ObjectDesc().id(2).pos({99.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(1);
    auto newObject = actualData.getOtherObjectRef({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(hostObject._pos + RealVector2D(0.0f, 0.5f), newObject._pos));
    EXPECT_TRUE(actualData.hasConnection(1, newObject._id));
    EXPECT_FALSE(actualData.hasConnection(0, newObject._id));
    EXPECT_FALSE(actualData.hasConnection(2, newObject._id));
    EXPECT_TRUE(actualData.hasConnection(1, 0));
    EXPECT_TRUE(actualData.hasConnection(1, 2));
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1)
{
    auto const ConstructionAngle = 0;
    //20.0f;

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({101.0f, 100.0f}),
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0))),
            ObjectDesc().id(2).pos({100.0f, 101.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(false).numBranches(1).nodes({NodeDesc().referenceAngle(ConstructionAngle)}),
        }));
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(1);
    auto newObject = actualData.getOtherObjectRef({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));
    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);
    EXPECT_TRUE(approxCompare(90.0f + 45.0f + ConstructionAngle, connection._angleFromPrevious));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_1_2__concatenation_0_1__branch_0_0)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc()}),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({99.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(2, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(1);
    auto newObject = actualData.getOtherObjectRef({0, 1});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newObject));
    EXPECT_FALSE(actualData.hasConnection(hostObject, prevCell));
    EXPECT_FALSE(actualData.hasConnection(hostObject, newObject));
}

TEST_F(ConstructorTests, creature_1__node_1_2__concatenation_0_1__branch_0_1)
{
    auto const LastAngle = 45.0f;
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc().referenceAngle(LastAngle)}).separation(false).numBranches(1),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({99.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->testOnly_calcTimestepWithCellFunctions();
    {
        auto actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
        ASSERT_EQ(1, actualData._creatures.size());
        EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

        auto creature = actualData.getCreatureRef(0);
        ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

        auto hostObject = actualData.getObjectRef(0);
        auto prevCell = actualData.getObjectRef(1);
        auto newObject = actualData.getOtherObjectRef({0, 1});
        EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
        EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));
        EXPECT_TRUE(actualData.hasConnection(prevCell, newObject));
        EXPECT_FALSE(actualData.hasConnection(hostObject, prevCell));
        EXPECT_TRUE(actualData.hasConnection(hostObject, newObject));
        EXPECT_TRUE(approxCompare(180.0f + LastAngle, actualData.getConnection(newObject, hostObject)._angleFromPrevious));
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

    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc().referenceAngle(MiddleAngle)}).separation(false).numBranches(1),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({101.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(3))),
            ObjectDesc().id(2).pos({100.0f, 101.0f}),
            ObjectDesc().id(3).pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * 1.0f).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(creature._id).size());

    auto hostObject = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(3);
    auto newObject = actualData.getOtherObjectRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newObject));
    EXPECT_FALSE(actualData.hasConnection(hostObject, prevCell));
    EXPECT_TRUE(actualData.hasConnection(hostObject, newObject));
    EXPECT_TRUE(approxCompare(180.0f + MiddleAngle, actualData.getConnection(newObject, hostObject)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_3__node_1_2__concatenation_0_1__branch_0_1__mirrored)
{
    auto const MiddleAngle = 5.0f;
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc().referenceAngle(MiddleAngle)}).separation(false).numBranches(1),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 101.0f}),
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(3))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * 1.0f).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(creature._id).size());

    auto hostObject = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(3);
    auto newObject = actualData.getOtherObjectRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newObject));
    EXPECT_FALSE(actualData.hasConnection(hostObject, prevCell));
    EXPECT_TRUE(actualData.hasConnection(hostObject, newObject));
    EXPECT_TRUE(approxCompare(180.0f + MiddleAngle, actualData.getConnection(newObject, hostObject)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_3__node_1_2__concatenation_0_1__branch_0_1__onSpike)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({NodeDesc(), NodeDesc().numAdditionalConnections(0)}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}),
            ObjectDesc()
                .id(2)
                .pos({11.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().currentNodeIndex(1).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(3))),
            ObjectDesc().id(3).pos({11.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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
    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({NodeDesc().referenceAngle(0.0f), NodeDesc().referenceAngle(45.0f), NodeDesc().referenceAngle(0.0f)}),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({99.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({NodeDesc().referenceAngle(0.0f), NodeDesc().referenceAngle(45.0f), NodeDesc().referenceAngle(0.0f)}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(2, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(1);
    auto newObject = actualData.getOtherObjectRef({0, 1});

    EXPECT_EQ(CellState_Constructing, newObject.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(1, newObject._id));
    EXPECT_FALSE(actualData.hasConnection(0, 1));
    EXPECT_TRUE(actualData.hasConnection(0, newObject._id));
    EXPECT_TRUE(approxCompare(180.0f + 45.0f, actualData.getConnection(newObject, hostObject)._angleFromPrevious));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(2, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_2_4__concatenation_0_1__branch_0_0__numAdditionalConnections_0)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(2).lastConstructedCellId(2))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({
                NodeDesc(),
                NodeDesc().referenceAngle(0),
                NodeDesc().referenceAngle(45.0f).numAdditionalConnections(0),
                NodeDesc(),
            }),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({99.0f, 99.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(2).pos({99.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({
                NodeDesc(),
                NodeDesc().referenceAngle(0),
                NodeDesc().referenceAngle(45.0f).numAdditionalConnections(0),
                NodeDesc(),
            }),
        }));
    data.addConnection(1, 2);
    data.addConnection(2, 0);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto prevPrevCell = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(2);
    auto newObject = actualData.getOtherObjectRef({0, 1, 2});

    EXPECT_EQ(CellState_Constructing, newObject.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(prevPrevCell, prevCell));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newObject));
    EXPECT_TRUE(actualData.hasConnection(newObject, hostObject));
    EXPECT_EQ(1, prevPrevCell._connections.size());
    EXPECT_EQ(2, prevCell._connections.size());
    EXPECT_EQ(2, newObject._connections.size());
    EXPECT_EQ(1, hostObject._connections.size());
    EXPECT_TRUE(approxCompare(180.0f + 45.0f, actualData.getConnection(newObject, hostObject)._angleFromPrevious));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
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

    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(2).lastConstructedCellId(2))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc()
                .separation(true)
                .nodes({
                    NodeDesc(),
                    NodeDesc().referenceAngle(0),
                    NodeDesc().referenceAngle(NodeAngle).numAdditionalConnections(1),
                    NodeDesc(),
                })
                .angleAlignment(angleAlignment),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({98.0f, 99.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(2).pos({98.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc()
                .separation(true)
                .nodes({
                    NodeDesc(),
                    NodeDesc().referenceAngle(0),
                    NodeDesc().referenceAngle(NodeAngle).numAdditionalConnections(1),
                    NodeDesc(),
                })
                .angleAlignment(angleAlignment),
        }));
    data.addConnection(2, 0);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto prevPrevCell = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(2);
    auto newObject = actualData.getOtherObjectRef({0, 1, 2});

    EXPECT_EQ(CellState_Constructing, newObject.getCellRef()._cellState);
    if (angleAlignment != ConstructorAngleAlignment_180) {
        EXPECT_TRUE(actualData.hasConnection(prevPrevCell, prevCell));
        EXPECT_TRUE(actualData.hasConnection(prevCell, newObject));
        EXPECT_TRUE(actualData.hasConnection(newObject, hostObject));
        EXPECT_TRUE(actualData.hasConnection(newObject, prevPrevCell));
        EXPECT_EQ(2, prevPrevCell._connections.size());
        EXPECT_EQ(2, prevCell._connections.size());
        EXPECT_EQ(3, newObject._connections.size());
        EXPECT_EQ(1, hostObject._connections.size());

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
        EXPECT_TRUE(approxCompare(refAngle1, actualData.getConnection(newObject, hostObject)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(refAngle2, actualData.getConnection(newObject, prevPrevCell)._angleFromPrevious));
        EXPECT_TRUE(approxCompare(refAngle3, actualData.getConnection(newObject, prevCell)._angleFromPrevious));
    } else {
        EXPECT_TRUE(actualData.hasConnection(prevPrevCell, prevCell));
        EXPECT_TRUE(actualData.hasConnection(prevCell, newObject));
        EXPECT_TRUE(actualData.hasConnection(newObject, hostObject));
        EXPECT_EQ(1, prevPrevCell._connections.size());
        EXPECT_EQ(2, prevCell._connections.size());
        EXPECT_EQ(2, newObject._connections.size());
        EXPECT_EQ(1, hostObject._connections.size());
        EXPECT_TRUE(approxCompare(180.0f + NodeAngle, actualData.getConnection(newObject, hostObject)._angleFromPrevious));
    }

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(3, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_1_3__branch_0_1__concatenationAngle)
{
    auto const ConcatenationAngle = 20.0f;

    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().referenceAngle(ConcatenationAngle)}).numConcatenations(3).separation(false).numBranches(1),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().geneIndex(0).currentConcatenation(1).currentNodeIndex(0).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({99.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(1);
    auto newObject = actualData.getOtherObjectRef({0, 1});

    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(prevCell, newObject));
    EXPECT_TRUE(actualData.hasConnection(newObject, hostObject));
    EXPECT_EQ(1, prevCell._connections.size());
    EXPECT_EQ(2, newObject._connections.size());
    EXPECT_EQ(1, hostObject._connections.size());
    EXPECT_TRUE(approxCompare(180.0f + ConcatenationAngle, actualData.getConnection(newObject, hostObject)._angleFromPrevious));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(2, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_4__concatenation_1_2__branch_0_1__numAdditionalConnections_1)
{
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .nodes({
                NodeDesc().referenceAngle(-90.0f).numAdditionalConnections(1),
                NodeDesc().referenceAngle(-90.0f),
                NodeDesc().referenceAngle(90.0f),
                NodeDesc().referenceAngle(90.0f),
            })
            .numConcatenations(2)
            .separation(false)
            .numBranches(1)
            .angleAlignment(ConstructorAngleAlignment_90),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().geneIndex(0).currentConcatenation(1).currentNodeIndex(0).lastConstructedCellId(4))),
            ObjectDesc().id(1).pos({100.0f, 98.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(2).pos({100.0f, 99.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos({101.0f, 99.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(4).pos({101.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(0, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(6, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(4);
    auto prevPrevPrevCell = actualData.getObjectRef(2);
    auto newObject = actualData.getOtherObjectRef({0, 1, 2, 3, 4});

    EXPECT_EQ(CellState_Constructing, newObject.getCellRef()._cellState);
    EXPECT_FALSE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(actualData.hasConnection(prevCell, newObject));
    EXPECT_TRUE(actualData.hasConnection(newObject, hostObject));
    EXPECT_TRUE(actualData.hasConnection(newObject, prevPrevPrevCell));
    EXPECT_EQ(2, prevCell._connections.size());
    EXPECT_EQ(3, newObject._connections.size());
    EXPECT_EQ(1, hostObject._connections.size());
    EXPECT_EQ(3, prevPrevPrevCell._connections.size());
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(newObject, prevCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(newObject, prevPrevPrevCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnection(newObject, hostObject)._angleFromPrevious));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_bothSidesPresent)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().currentNodeIndex(3).autoTriggerInterval(1).lastConstructedCellId(2))),
            ObjectDesc().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos({10.0f + getOffspringDistance(), 9.5f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(4).pos({10.0f + getOffspringDistance(), 10.5f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(false).numBranches(1).nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc().numAdditionalConnections(1)}),
        }));

    data.addConnection(1, 2);
    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f + 180.0f);
    data.addConnection(2, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().currentNodeIndex(3).autoTriggerInterval(1).lastConstructedCellId(2))),
            ObjectDesc().id(2).pos({10.0f + getOffspringDistance() + 0.8f, 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc()
                .id(3)
                .pos(RealVector2D(10.0f + getOffspringDistance() + 0.8f + 0.2f, 10.0f) + offset * 0.1f)
                .type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc()
                .id(4)
                .pos(RealVector2D(10.0f + getOffspringDistance() + 0.8f, 10.0f) + offset * 0.2f)
                .type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(false).nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc().numAdditionalConnections(2)}),
        }));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.getConnectionRef(2, 3)._angleFromPrevious = 60.0f;
    data.getConnectionRef(2, 1)._angleFromPrevious = 300.0f;
    data.getConnectionRef(3, 4)._angleFromPrevious = 120.0f;
    data.getConnectionRef(3, 2)._angleFromPrevious = 240.0f;


    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({458.20f, 239.23f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().currentNodeIndex(3).autoTriggerInterval(1).lastConstructedCellId(2))),
            ObjectDesc().id(2).pos({456.40f, 238.88f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos({455.96f, 239.75f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(4).pos({456.07f, 240.77f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(false).nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc().numAdditionalConnections(2)}),
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
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc().numAdditionalConnections(1)}),
    });

    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2))),
            ObjectDesc().id(2).pos({10.0f + getOffspringDistance() + 0.8f, 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc()
                .id(3)
                .pos(RealVector2D(10.0f + getOffspringDistance() + 0.8f + 0.2f, 10.0f) + offset * 0.1f)
                .type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc()
                .id(4)
                .pos(RealVector2D(10.0f + getOffspringDistance() + 0.8f, 10.0f) + offset * 0.2f)
                .type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    data.getConnectionRef(2, 3)._angleFromPrevious = 60.0f;
    data.getConnectionRef(2, 1)._angleFromPrevious = 300.0f;
    data.getConnectionRef(3, 4)._angleFromPrevious = 120.0f;
    data.getConnectionRef(3, 2)._angleFromPrevious = 240.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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
    ASSERT_EQ(3, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(1, actualPrevPrevPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, actualData.getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(240.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));

    EXPECT_EQ(actualPrevPrevPrevConstructedCell._connections, actualPrevPrevPrevConstructedCell._connections);
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_1__90degAlignment)
{
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_90)
            .nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc().numAdditionalConnections(1)}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2))),
            ObjectDesc().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(4).pos({10.0f + getOffspringDistance() - 1.0f, 9.0f - 0.2f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = data.getObjectRef(3)._pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).angleAlignment(ConstructorAngleAlignment_90).nodes({NodeDesc(), NodeDesc(), NodeDesc().numAdditionalConnections(0)}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().currentNodeIndex(2).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2))),
            ObjectDesc().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_90)
            .nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc().numAdditionalConnections(1)}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2))),
            ObjectDesc().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos({10.0f + getOffspringDistance(), 10.0f - 0.5f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(4).pos({10.0f + getOffspringDistance(), 10.0f - 1.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = data.getObjectRef(3)._pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).numConcatenations(GeneDesc::NumConcatenations_Infinite).nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(0))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef({0});

    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(newObject, hostObject));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_1_inf__branch_0_0)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().geneIndex(0).currentNodeIndex(0).currentConcatenation(1).lastConstructedCellId(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).numConcatenations(GeneDesc::NumConcatenations_Infinite).nodes({NodeDesc()}),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({101.0f, 100.0f}).type(CellDesc().cellState(CellState_Ready)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc().separation(true).numConcatenations(GeneDesc::NumConcatenations_Infinite).nodes({NodeDesc()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(newCreature._id).size());

    auto hostObject = actualData.getObjectRef(0);
    auto prevCell = actualData.getObjectRef(1);
    auto newObject = actualData.getOtherObjectRef({0, 1});

    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(actualData.hasConnection(newObject, hostObject));
    EXPECT_TRUE(actualData.hasConnection(newObject, prevCell));
    EXPECT_EQ(1, prevCell._connections.size());
    EXPECT_EQ(2, newObject._connections.size());
    EXPECT_EQ(1, hostObject._connections.size());

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(2, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1__largeConstructionAngle)
{
    auto const ConstructionAngle = 180.0f;

    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}).separation(false).numBranches(1),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 99.0f}),
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().constructionAngle(ConstructionAngle).geneIndex(0).currentNodeIndex(0).autoTriggerInterval(100))),
            ObjectDesc().id(2).pos({100.1f, 101.0f}),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(hostCreature._id).size());

    auto const& hostObject = actualData.getObjectRef(1);
    auto const& newObject = actualData.getOtherObjectRef({0, 1, 2});

    auto angleSpan_cell2_cell0 = hostObject.getAngleSpan(2, 0);
    auto angleSpan_lastObject_and_cell0 = hostObject.getAngleSpan(newObject._id, 0);
    EXPECT_TRUE(approxCompare(Math::getNormalizedAngle(angleSpan_lastObject_and_cell0 + ConstructionAngle, 0.0f), angleSpan_cell2_cell0 / 2));
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1__frontAngle_leftSide)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}),
            ObjectDesc()
                .id(2)
                .pos({9.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().currentNodeIndex(0).autoTriggerInterval(1).geneIndex(0))),
            ObjectDesc().id(3).pos({9.0f, 11.0f}),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 2; ++i) {
        _simulationFacade->testOnly_calcTimestepWithCellFunctions();
    }

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
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({8.0f, 10.0f}),
            ObjectDesc()
                .id(2)
                .pos({9.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().currentNodeIndex(0).autoTriggerInterval(1).geneIndex(0))),
            ObjectDesc().id(3).pos({9.0f, 11.0f}),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 2; ++i) {
        _simulationFacade->testOnly_calcTimestepWithCellFunctions();
    }

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, actualData.getObjectsForCreature(creature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3});

    EXPECT_EQ(CellState_Ready, actualConstructedCell.getCellRef()._cellState);
}

enum class ConstructionType
{
    Normal,
    Seed
};

class ConstructorTests_AllShapes
    : public ConstructorTests
    , public testing::WithParamInterface<std::tuple<ConstructorShape, ConstructionType, float>>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_AllShapes,
    ConstructorTests_AllShapes,
    ::testing::Combine(
        ::testing::Values(
            ConstructorShape_Segment,
            ConstructorShape_Triangle,
            ConstructorShape_Rectangle,
            ConstructorShape_Hexagon,
            ConstructorShape_Loop,
            ConstructorShape_Tube,
            ConstructorShape_Lolli,
            ConstructorShape_SmallLolli,
            ConstructorShape_Zigzag),
        ::testing::Values(ConstructionType::Normal, ConstructionType::Seed),
        ::testing::Values(0.5f, 1.0f, 2.0f)));

TEST_P(ConstructorTests_AllShapes, creature_3__generateShape)
{
    _parameters.friction.baseValue = 0.05f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto const ConstructionAngle = 8.0f;
    auto const LastAngle = -5.0f;
    auto const n = 20;
    auto const AutoTriggerInterval = 60;

    auto [shape, type, connectionDistance] = GetParam();

    auto gene = GeneDesc().separation(false).numBranches(1).shape(shape).connectionDistance(connectionDistance);
    gene._nodes.emplace_back(NodeDesc());
    for (int i = 0; i < n - 2; ++i) {
        gene._nodes.emplace_back(NodeDesc());
    }
    gene._nodes.emplace_back(NodeDesc().referenceAngle(LastAngle));
    auto genome = GenomeDesc().genes({gene});

    Desc data;
    if (type == ConstructionType::Normal) {
        data = Desc().addCreature(
            {
                ObjectDesc().id(0).pos({100.0f, 100.0f - 5.0f}).type(CellDesc()),
                ObjectDesc().id(1).pos({100.0f, 100.0f - 4.5f}).type(CellDesc()),
                ObjectDesc().id(2).pos({100.0f, 100.0f - 4.0f}).type(CellDesc()),
                ObjectDesc().id(3).pos({100.0f, 100.0f - 3.5f}).type(CellDesc()),
                ObjectDesc().id(4).pos({100.0f, 100.0f - 3.0f}).type(CellDesc()),
                ObjectDesc().id(5).pos({100.0f, 100.0f - 2.5f}).type(CellDesc()),
                ObjectDesc().id(6).pos({100.0f, 100.0f - 2.0f}).type(CellDesc()),
                ObjectDesc().id(7).pos({100.0f, 100.0f - 1.5f}).type(CellDesc()),
                ObjectDesc().id(8).pos({100.0f, 100.0f - 1.0f}).type(CellDesc()),
                ObjectDesc().id(9).pos({100.0f, 100.0f - 0.5f}).type(CellDesc()),
                ObjectDesc()
                    .id(10)
                    .pos({100.0f, 100.0f})
                    .type(CellDesc()
                              .headCell(true)
                              .usableEnergy(getConstructorEnergy() * n)
                              .constructor(ConstructorDesc()
                                               .constructionAngle(ConstructionAngle)
                                               .geneIndex(0)
                                               .currentNodeIndex(0)
                                               .autoTriggerInterval(AutoTriggerInterval))),
                ObjectDesc().id(11).pos({100.1f, 100.0f + 0.5f}).type(CellDesc()),
                ObjectDesc().id(12).pos({100.1f, 100.0f + 1.0f}).type(CellDesc()),
                ObjectDesc().id(13).pos({100.1f, 100.0f + 1.5f}).type(CellDesc()),
                ObjectDesc().id(14).pos({100.1f, 100.0f + 2.0f}).type(CellDesc()),
                ObjectDesc().id(15).pos({100.1f, 100.0f + 2.5f}).type(CellDesc()),
                ObjectDesc().id(16).pos({100.1f, 100.0f + 3.0f}).type(CellDesc()),
                ObjectDesc().id(17).pos({100.1f, 100.0f + 3.5f}).type(CellDesc()),
                ObjectDesc().id(18).pos({100.1f, 100.0f + 4.0f}).type(CellDesc()),
                ObjectDesc().id(19).pos({100.1f, 100.0f + 4.5f}).type(CellDesc()),
                ObjectDesc().id(20).pos({100.1f, 100.0f + 5.0f}).type(CellDesc()),
            },
            CreatureDesc().id(0),
            genome);
        for (int i = 0; i < 20; ++i) {
            data.addConnection(i, i + 1);
        }
    } else {
        data = Desc().addCreature(
            {
                ObjectDesc()
                    .id(10)
                    .pos({100.0f, 100.0f})
                    .type(CellDesc()
                              .headCell(true)
                              .usableEnergy(getConstructorEnergy() * n)
                              .constructor(ConstructorDesc()
                                               .constructionAngle(ConstructionAngle)
                                               .geneIndex(0)
                                               .currentNodeIndex(0)
                                               .autoTriggerInterval(AutoTriggerInterval))),
            },
            CreatureDesc().id(0),
            genome);
    }
    _simulationFacade->setSimulationData(data);

    // Construct offspring and record ids of constructed cells
    std::vector<uint64_t> createdCellIds;
    {
        for (int i = 0; i < n; ++i) {
            int anticipatedNumObjectsForCreature = type == ConstructionType::Normal ? 22 + i : 2 + i;
            Desc actualData;
            CreatureDesc hostCreature;
            int retryCount = 0;
            do {
                _simulationFacade->calcTimesteps(AutoTriggerInterval);
                actualData = _simulationFacade->getSimulationData();

                ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
                ASSERT_EQ(1, actualData._creatures.size());
                EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

                hostCreature = actualData.getCreatureRef(0);
                if (++retryCount == 100) {
                    FAIL();
                }
            } while (actualData.getObjectsForCreature(hostCreature._id).size() != anticipatedNumObjectsForCreature);

            auto hostObject = actualData.getObjectRef(10);

            std::set<uint64_t> knownCellIds(createdCellIds.begin(), createdCellIds.end());
            if (type == ConstructionType::Normal) {
                for (int j = 0; j <= 20; ++j) {
                    knownCellIds.insert(j);
                }
            } else {
                knownCellIds.insert(10);
            }
            auto newObject = actualData.getOtherObjectRef(knownCellIds);
            createdCellIds.emplace_back(newObject._id);

            if (i < n - 1) {
                EXPECT_EQ(CellState_Constructing, newObject.getCellRef()._cellState);
            } else {
                EXPECT_EQ(CellState_Ready, newObject.getCellRef()._cellState);
            }
            EXPECT_TRUE(actualData.hasConnection(hostObject, newObject));
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
            EXPECT_TRUE(approxCompare(shapeResult.angle, angle));
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
    if (type == ConstructionType::Normal) {
        auto const& hostObject = actualData.getObjectRef(10);
        auto angleSpan_cell11_cell9 = hostObject.getAngleSpan(11, 9);
        auto angleSpan_hostObject_and_cell9 = hostObject._connections.at(0)._angleFromPrevious;
        EXPECT_TRUE(approxCompare(angleSpan_hostObject_and_cell9 + ConstructionAngle, angleSpan_cell11_cell9 / 2));
    }

    // Check angles for last node
    {
        auto const& object = actualData.getObjectRef(createdCellIds.back());
        auto prevCellId = createdCellIds.at(n - 2);
        auto nextObjectId = 10;  // = id of hostObject
        auto angle = object.getAngleSpan(prevCellId, nextObjectId);
        angle = Math::getNormalizedAngle(angle - 180.0f, -180.0f);
        EXPECT_EQ(LastAngle, angle);
    }
}

TEST_F(ConstructorTests, avoidDeadlockByLockingNearObjects)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({}),
        GeneDesc().separation(false).shape(ConstructorShape_Hexagon).nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}),
            ObjectDesc()
                .id(2)
                .pos({10.0f, 9.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(1).currentNodeIndex(2).lastConstructedCellId(6))),
            ObjectDesc().id(3).pos({11.0f, 9.0f}),
            ObjectDesc()
                .id(4)
                .pos({11.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(1).currentNodeIndex(2).lastConstructedCellId(8))),
            ObjectDesc().id(5).pos({10.0f, 9.0f - getOffspringDistance() - 1.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0).geneIndex(1)),
            ObjectDesc().id(6).pos({10.0f, 9.0f - getOffspringDistance()}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(1).geneIndex(1)),
            ObjectDesc().id(7).pos({11.0f + getOffspringDistance() + 1.0f, 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0).geneIndex(1)),
            ObjectDesc().id(8).pos({11.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(1).geneIndex(1)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 1);

    data.addConnection(5, 6);
    data.addConnection(6, 2);

    data.addConnection(7, 8);
    data.addConnection(8, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(10, actualData.getObjectsForCreature(creature._id).size());
}

TEST_F(ConstructorTests, avoidConnectionsBetweenDifferentConstructions)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({}),
        GeneDesc().separation(false).shape(ConstructorShape_Hexagon).nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        GeneDesc().separation(false).shape(ConstructorShape_Hexagon).nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .nodeIndex(0)
                          .constructor(ConstructorDesc().geneIndex(1).currentNodeIndex(2).lastConstructedCellId(4))),
            ObjectDesc()
                .id(2)
                .pos({11.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .nodeIndex(1)
                          .constructor(ConstructorDesc().geneIndex(2).currentNodeIndex(2).lastConstructedCellId(6))),

            ObjectDesc()
                .id(3)
                .pos({10.1f, 10.0f - getOffspringDistance() - 1.0f})
                .type(CellDesc().cellState(CellState_Constructing).nodeIndex(0).geneIndex(1).parentNodeIndex(0)),
            ObjectDesc()
                .id(4)
                .pos({10.0f, 10.0f - getOffspringDistance()})
                .type(CellDesc().cellState(CellState_Constructing).nodeIndex(1).geneIndex(1).parentNodeIndex(0)),
            ObjectDesc()
                .id(5)
                .pos({11.1f, 10.0f - getOffspringDistance() - 1.0f})
                .type(CellDesc().cellState(CellState_Constructing).nodeIndex(0).geneIndex(2).parentNodeIndex(1)),
            ObjectDesc()
                .id(6)
                .pos({11.0f, 10.0f - getOffspringDistance()})
                .type(CellDesc().cellState(CellState_Constructing).nodeIndex(1).geneIndex(2).parentNodeIndex(1)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(1, 2);

    data.addConnection(4, 1);
    data.addConnection(3, 4);

    data.addConnection(6, 2);
    data.addConnection(5, 6);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(8, actualData.getObjectsForCreature(creature._id).size());

    auto constructedCells = actualData.getOtherObjects({1, 2, 3, 4, 5, 6});
    ObjectDesc object1, object2;
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

    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
            NodeDesc(),
        }),
        GeneDesc().separation(separation == Separation::Yes).numBranches(2).numConcatenations(3).nodes({NodeDesc(), NodeDesc()}),
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
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(constructorEnergy)
                          .constructor(
                              ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({0, 1});
    EXPECT_TRUE(approxCompare(normalCellEnergy, actualConstructedCell.getCellRef()._usableEnergy));

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        EXPECT_TRUE(approxCompare(normalCellEnergy, actualData.getObjectRef(0).getCellRef()._usableEnergy));
        EXPECT_TRUE(approxCompare(0.0f, actualConstructedCell.getCellRef()._rawEnergy));
        EXPECT_TRUE(approxCompare(0.0f, actualConstructedCell.getCellRef()._reservedEnergy));
        auto newConstructor = actualConstructedCell.getCellRef()._constructor.value();
        if (separation == Separation::Yes) {
            EXPECT_EQ(ProvideEnergy_CellOnly, newConstructor._provideEnergy);
        } else {
            EXPECT_EQ(ProvideEnergy_FreeGeneration, newConstructor._provideEnergy);
        }
    } else {
        if (provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No) {
            EXPECT_TRUE(approxCompare(0, actualConstructedCell.getCellRef()._rawEnergy));
            EXPECT_TRUE(approxCompare(normalCellEnergy * (2 * 3 * 2), actualConstructedCell.getCellRef()._reservedEnergy));
        } else {
            EXPECT_TRUE(approxCompare(0.0f, actualConstructedCell.getCellRef()._rawEnergy));
            EXPECT_TRUE(approxCompare(0.0f, actualConstructedCell.getCellRef()._reservedEnergy));
        }
    }
}

TEST_P(ConstructorTests_ProvideEnergy_Separation, provideEnergy_insufficientEnergy)
{
    auto [provideEnergy, separation] = GetParam();

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        GTEST_SKIP() << "Skipping test because FreeGeneration always has enough energy.";
    }
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
            NodeDesc(),
        }),
        GeneDesc().separation(separation == Separation::Yes).numBranches(2).numConcatenations(3).nodes({NodeDesc(), NodeDesc()}),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto constructorEnergy =
        provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No ? normalCellEnergy * (2 * 3 * 2 + 2) - 1.0f : normalCellEnergy * 2 - 1.0f;
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(constructorEnergy)
                          .constructor(
                              ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());
}

TEST_P(ConstructorTests_ProvideEnergy_Separation, provideEnergy_infiniteConcatenations)
{
    auto [provideEnergy, separation] = GetParam();

    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
            NodeDesc(),
        }),
        GeneDesc().separation(separation == Separation::Yes).numBranches(2).numConcatenations(std::numeric_limits<int>::max()).nodes({NodeDesc(), NodeDesc()}),
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


    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(constructorEnergy)
                          .constructor(
                              ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({0, 1});
    EXPECT_TRUE(approxCompare(normalCellEnergy, actualConstructedCell.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(0, actualConstructedCell.getCellRef()._rawEnergy));

    EXPECT_TRUE(approxCompare(0, actualData.getObjectRef(0).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(0, actualData.getObjectRef(0).getCellRef()._reservedEnergy));

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        EXPECT_TRUE(approxCompare(0, actualConstructedCell.getCellRef()._reservedEnergy));
        EXPECT_TRUE(approxCompare(normalCellEnergy, actualData.getObjectRef(0).getCellRef()._usableEnergy));
    } else {
        if (provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No) {
            EXPECT_TRUE(approxCompare(normalCellEnergy * 2, actualConstructedCell.getCellRef()._reservedEnergy));
        } else {
            EXPECT_TRUE(approxCompare(0, actualConstructedCell.getCellRef()._reservedEnergy));
        }
    }
}

TEST_F(ConstructorTests, constructWithReservedEnergy)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Desc().addCreature(
        {ObjectDesc()
             .pos({100.0f, 100.0f})
             .type(CellDesc()
                       .usableEnergy(normalCellEnergy * 1.5)
                       .reservedEnergy(normalCellEnergy / 2 + NEAR_ZERO)
                       .constructor(ConstructorDesc().autoTriggerInterval(3)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().separation(true).nodes({NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numObjects);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numObjects);

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_TRUE(approxCompare(normalCellEnergy, hostObject.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(0, hostObject.getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(0, hostObject.getCellRef()._reservedEnergy));

    EXPECT_TRUE(approxCompare(normalCellEnergy, newObject.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(0, newObject.getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(0, newObject.getCellRef()._reservedEnergy));
}

class ConstructorTests_ProvideEnergy
    : public ConstructorTests
    , public testing::WithParamInterface<ProvideEnergy>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_ProvideEnergy,
    ConstructorTests_ProvideEnergy,
    ::testing::Values(ProvideEnergy_CellOnly, ProvideEnergy_CellAndGene, ProvideEnergy_FreeGeneration));

TEST_P(ConstructorTests_ProvideEnergy, provideEnergy_depotWithInitialStoredEnergy_sufficientEnergy)
{
    auto provideEnergy = GetParam();

    auto const InitialStoredUsableEnergy = 50.0f;

    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc().cellType(DepotGenomeDesc().initialStoredUsableEnergy(InitialStoredUsableEnergy)),
            NodeDesc(),
        }),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    // Calculate required energy: normalCellEnergy + InitialStoredUsableEnergy for depot cell
    auto constructorEnergy = [&] {
        if (provideEnergy == ProvideEnergy_FreeGeneration) {
            return normalCellEnergy;
        }

        if (provideEnergy == ProvideEnergy_CellAndGene) {
            return normalCellEnergy * (2 + 2) + 1.0f + InitialStoredUsableEnergy;  // Contains energy for all nodes in gene
        } else {
            return normalCellEnergy * 2 + InitialStoredUsableEnergy + 1.0f;  // Contains energy for depot node in gene
        }
    }();

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(constructorEnergy)
                          .constructor(
                              ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

    // For separation, offspring is in a separate creature
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({0, 1});

    // Verify the constructed cell is a depot with stored energy
    EXPECT_EQ(CellType_Depot, actualConstructedCell.getCellRef().getCellType());
    auto const& depot = std::get<DepotDesc>(actualConstructedCell.getCellRef()._cellType);
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

    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc().cellType(DepotGenomeDesc().initialStoredUsableEnergy(InitialStoredUsableEnergy)),
            NodeDesc(),
        }),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    // Not enough energy: just enough for cell but not for depot initial energy
    auto constructorEnergy = normalCellEnergy * 2 + InitialStoredUsableEnergy - 1.0f;

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(constructorEnergy)
                          .constructor(
                              ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).currentNodeIndex(1).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().separation(false).nodes({
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
            NodeDesc(),
        }),
    });
    std::vector<ObjectDesc> largeCreatureCells;
    for (int i = 0; i < 50; ++i) {
        largeCreatureCells.emplace_back(
            ObjectDesc().id(i).pos({toFloat(i), 0.0f}).type(CellDesc().headCell(true).constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(30))));
    }
    Desc largeCreatureData;
    largeCreatureData.addCreature(largeCreatureCells, CreatureDesc(), genome);
    for (int i = 1; i < 50; ++i) {
        largeCreatureData.addConnection(i, i - 1);
    }
    auto largeData = Desc();
    for (int i = 0; i < 10; ++i) {
        auto clone = largeCreatureData;
        DescEditService::get().setCenter(clone, {100.0f, toFloat(i) * 20});
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
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_90)
            .nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc().numAdditionalConnections(1),
            }),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos(RealVector2D{10.0f, 8.5f} + Math::unitVectorOfAngle(180.0f - 45.0f) * getOffspringDistance())
                .type(CellDesc().constructor(ConstructorDesc().currentNodeIndex(4).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(4).provideEnergy(
                    ProvideEnergy_FreeGeneration))),
            ObjectDesc().id(1).pos({10.0f, 11.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(2).pos({10.0f, 9.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos({9.0f, 9.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(4).pos({10.0f, 8.5f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(4, 0);
    data.addConnection(3, 4);
    data.addConnection(2, 3);
    data.addConnection(1, 2);
    data.getConnectionRef(3, 4)._angleFromPrevious = 270.0f;
    data.getConnectionRef(3, 2)._angleFromPrevious = 90.0f;
    data.getConnectionRef(4, 0)._angleFromPrevious = 270.0f;
    data.getConnectionRef(4, 3)._angleFromPrevious = 90.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

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
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_90)
            .nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc().numAdditionalConnections(1),
            }),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos(RealVector2D{10.0f, 8.5f} + Math::unitVectorOfAngle(180.0f + 45.0f) * getOffspringDistance())
                .type(CellDesc().constructor(ConstructorDesc().currentNodeIndex(4).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(4).provideEnergy(
                    ProvideEnergy_FreeGeneration))),

            ObjectDesc().id(1).pos({10.0f, 11.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(2).pos({10.0f, 9.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos({11.0f, 9.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(4).pos({10.0f, 8.5f}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(4, 0);
    data.addConnection(3, 4);
    data.addConnection(2, 3);
    data.addConnection(1, 2);
    data.getConnectionRef(3, 4)._angleFromPrevious = 90.0f;
    data.getConnectionRef(3, 2)._angleFromPrevious = 270.0f;
    data.getConnectionRef(4, 0)._angleFromPrevious = 90.0f;
    data.getConnectionRef(4, 3)._angleFromPrevious = 270.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(6, actualData.getObjectsForCreature(creature._id).size());

    auto object = [&](uint64_t id) { return actualData.getObjectRef(id); };
    auto constructedCell = actualData.getOtherObjectRef({0, 1, 2, 3, 4});

    ASSERT_EQ(1, object(1)._connections.size());
    ASSERT_EQ(3, object(2)._connections.size());
    ASSERT_EQ(2, object(3)._connections.size());
    ASSERT_EQ(2, object(4)._connections.size());
    ASSERT_EQ(3, constructedCell._connections.size());
    ASSERT_EQ(1, object(0)._connections.size());

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnectionRef(2, constructedCell._id)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnectionRef(2, 1)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(90.0f, actualData.getConnectionRef(2, 3)._angleFromPrevious));
}

TEST_F(ConstructorTests, angleCorrectionByInnerSumOfPolygon_preventZeroAngles)
{
    auto genome = GenomeDesc().genes({
        GeneDesc()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_60)
            .nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc().numAdditionalConnections(2),
            }),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos(RealVector2D{697.63, 356.23})
                .type(CellDesc().constructor(ConstructorDesc().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(1).provideEnergy(
                    ProvideEnergy_FreeGeneration))),
            ObjectDesc().id(1).pos({698.22, 355.43}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(2).pos({697.21, 355.64}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc().id(3).pos({696.96, 356.55}).type(CellDesc().cellState(CellState_Constructing)),
        },
        CreatureDesc().id(0),
        genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.getConnectionRef(1, 0)._angleFromPrevious = 300.0f;
    data.getConnectionRef(1, 2)._angleFromPrevious = 60.0f;
    data.getConnectionRef(2, 1)._angleFromPrevious = 240.0f;
    data.getConnectionRef(2, 3)._angleFromPrevious = 120.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(creature._id).size());

    auto object = [&](uint64_t id) { return actualData.getObjectRef(id); };
    auto constructedCell = actualData.getOtherObjectRef({0, 1, 2, 3});

    ASSERT_EQ(2, object(1)._connections.size());
    ASSERT_EQ(3, object(2)._connections.size());
    ASSERT_EQ(2, object(3)._connections.size());
    ASSERT_EQ(4, constructedCell._connections.size());
    ASSERT_EQ(1, object(0)._connections.size());

    EXPECT_TRUE(approxCompare(300.0f, actualData.getConnectionRef(1, constructedCell._id)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnectionRef(1, 2)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(240.0f, actualData.getConnectionRef(2, 1)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnectionRef(2, 3)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnectionRef(2, constructedCell._id)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(300.0f, actualData.getConnectionRef(3, 2)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(60.0f, actualData.getConnectionRef(3, constructedCell._id)._angleFromPrevious));
}

TEST_F(ConstructorTests, externalEnergyInflowOnlyForFirstOffspring_firstOffspring)
{
    _parameters.externalEnergyControlToggle.value = true;
    _parameters.externalEnergy.value = Infinity<float>::value;
    _parameters.externalEnergyInflowOnlyForFirstOffspring.value = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto normalEnergy = _parameters.normalCellEnergy.value[0];
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(normalEnergy).constructor(ConstructorDesc().geneIndex(0).currentOffspring(0))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(true).nodes({NodeDesc()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    // Construction should succeed because currentOffspring == 0 allows energy inflow
    ASSERT_EQ(2, actualData._creatures.size());
}

TEST_F(ConstructorTests, externalEnergyInflowOnlyForFirstOffspring_secondOffspring)
{
    _parameters.externalEnergyControlToggle.value = true;
    _parameters.externalEnergy.value = Infinity<float>::value;
    _parameters.externalEnergyInflowOnlyForFirstOffspring.value = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto normalEnergy = _parameters.normalCellEnergy.value[0];
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(normalEnergy).constructor(ConstructorDesc().geneIndex(0).currentOffspring(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().separation(true).nodes({NodeDesc()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    // Construction should fail because currentOffspring > 0 blocks energy inflow and cell has insufficient energy
    ASSERT_EQ(1, actualData._creatures.size());
}
