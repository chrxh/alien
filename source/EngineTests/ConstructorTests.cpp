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

    static float getOffspringDistance() { return 1.0f; }
    static float getReservedEnergy(CellDesc const& cell) { return cell._constructor.has_value() ? cell._constructor->_reservedEnergy : 0.0f; }

    DescTestDataFactory* _descTestDataFactory;
};

TEST_F(ConstructorTests, alreadyFinished)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(1).separation(false).numBranches(1))),
            ObjectDesc().id(1).pos({100.0f, 101.0f}).type(CellDesc()),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc()}),
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
    EXPECT_EQ(0, hostConstructor._currentOffspring);
    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, emptyGenome)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0))),
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
    ASSERT_EQ(1, creature._numCells);

    // Verify no active signal
    auto hostObject = actualData.getObjectRef(0);
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, emptyGene)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc()}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numCells);

    auto hostObject = actualData.getObjectRef(0);

    // Verify no active signal
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, insufficientEnergy)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().constructor(ConstructorDesc().geneIndex(0).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes({NodeDesc()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(1, creature._numCells);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();

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
                .type(CellDesc().constructor(ConstructorDesc().autoTriggerInterval(std::nullopt).geneIndex(0).separation(true))),  // Not enough energy
            ObjectDesc().id(1).pos({101.0f, 100.0f}).type(CellDesc().signal({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes({NodeDesc()})}));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(2, creature._numCells);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[Channels::ConstructorSuccess]));
}

TEST_F(ConstructorTests, manuallyTriggered_withSignal_success)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().autoTriggerInterval(std::nullopt).geneIndex(0).separation(false))),
            ObjectDesc().id(1).pos({101.0f, 100.0f}).type(CellDesc().signal({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes({NodeDesc()})}));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(3, creature._numCells);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_TRUE(approxCompare(1.0f, hostObject.getCellRef()._signal._channels[Channels::ConstructorSuccess]));
}

TEST_F(ConstructorTests, manuallyTriggered_withoutSignal)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().autoTriggerInterval(std::nullopt).geneIndex(0).separation(false))),
            ObjectDesc().id(1).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes({NodeDesc()})}));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());
    ASSERT_EQ(2, creature._numCells);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();

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
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(1).separation(false).numBranches(1))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes({NodeDesc()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(2, hostCreature._numCells);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();

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
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(1).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc(), NodeDesc()}),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({100.5f, 100.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc(), NodeDesc()}),
        }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(1, hostCreature._numCells);

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    ASSERT_EQ(1, newCreature._numCells);

    auto hostObject = actualData.getObjectRef(0);
    auto hostConstructor = hostObject.getCellRef()._constructor.value();

    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_F(ConstructorTests, crossingLinks_allowed)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(10)
                .pos({10.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().lastConstructedCellId(3).separation(true))),
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

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, actualData.getObjectsForCreature(creature._id).size());
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
    auto randomNode = _descTestDataFactory->createNonDefaultNodeDesc(nodeParameter);

    auto data = Desc().addCreature(
        {ObjectDesc().pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().separation(true)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({randomNode}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numCells);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numCells);

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_EQ(newCreature._headUpdateId, newObject.getCellRef()._headUpdateId);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));
    EXPECT_FALSE(actualData.hasConnection(hostObject._id, newObject._id));

    EXPECT_TRUE(_descTestDataFactory->compare(newObject, randomNode));
    auto newConstructor = newObject.getCellRef()._constructor.value();
    EXPECT_EQ(ProvideEnergy_CellOnly, newConstructor._provideEnergy);

    // Verify no active signal
    EXPECT_TRUE(approxCompare(0.0f, hostObject.getCellRef()._signal._channels[0]));
}

TEST_P(ConstructorTests_AllNodeTypes, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0__preview)
{
    auto nodeParameter = GetParam();
    auto randomNode = _descTestDataFactory->createNonDefaultNodeDesc(nodeParameter);

    auto data = Desc().addCreature(
        {ObjectDesc().pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().separation(true)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({randomNode}),
        }));

    _simulationFacade->setPreviewData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctionsForPreview();

    auto actualData = _simulationFacade->getPreviewData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numCells);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numCells);

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(Math::length(hostObject._pos - newObject._pos) > 50.0f);  // Preview specific: Move seed far away from construction
    EXPECT_FALSE(actualData.hasConnection(hostObject._id, newObject._id));

    EXPECT_TRUE(_descTestDataFactory->compare(newObject, randomNode));
    auto newConstructor = newObject.getCellRef()._constructor.value();
    EXPECT_EQ(ProvideEnergy_CellOnly, newConstructor._provideEnergy);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0__preview_detail)
{
    auto randomNode = _descTestDataFactory->createNonDefaultNodeDesc(NodeParameter{CellType_Base});

    auto data = Desc().addCreature(
        {ObjectDesc().pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().separation(true)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({randomNode}),
        }));

    _simulationFacade->setPreviewData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctionsForPreview(true);

    auto actualData = _simulationFacade->getPreviewData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numCells);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numCells);

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(Math::length(hostObject._pos - newObject._pos) > 50.0f);  // Preview specific: Move seed far away from construction
    EXPECT_TRUE(_descTestDataFactory->compare(newObject, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostObject._id, newObject._id));
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_1)
{
    auto data = Desc().addCreature(
        {ObjectDesc().pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(1).separation(true)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc(),
            GeneDesc().nodes({NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numCells);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numCells);

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));
    EXPECT_FALSE(actualData.hasConnection(hostObject._id, newObject._id));

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(1, hostConstructor._currentOffspring);
}

TEST_F(ConstructorTests, creature_1__node_2_3__concatenation_0_1__branch_0_0__frontAngle_upperSide)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().shape(ConstructorShape_Triangle).nodes({NodeDesc(), NodeDesc(), NodeDesc().constructor(ConstructorGenomeDesc().separation(true))}),
    });
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({10.0f, 10.0f})
                                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(2).separation(true))),
                        },
                        CreatureDesc().id(0),
                        genome)
                    .addCreature(
                        {
                            ObjectDesc().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(1)),
                            ObjectDesc().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
                        },
                        CreatureDesc().id(1),
                        genome);
    // Order of connections is unusual and not as it would be in a normal construction process
    // Thi is intentional to test that the ConstructionProcessor does not produce an invalid state
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(3, actualData.getObjectsForCreature(newCreature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3});
    auto hostCell = actualData.getObjectRef(1);
    auto prevConstructedCell = actualData.getObjectRef(2);
    auto prevPrevConstructedCell = actualData.getObjectRef(3);

    EXPECT_TRUE(hostCell._connections.empty());

    EXPECT_TRUE(actualData.hasConnection(2, 3));
    EXPECT_TRUE(actualData.hasConnection(2, actualConstructedCell._id));

    EXPECT_TRUE(actualData.hasConnection(3, 2));
    EXPECT_TRUE(actualData.hasConnection(2, actualConstructedCell._id));

    EXPECT_TRUE(actualData.hasConnection(actualConstructedCell._id, 2));
    EXPECT_TRUE(actualData.hasConnection(actualConstructedCell._id, 3));

    EXPECT_EQ(CellState_Activating, actualConstructedCell.getCellRef()._cellState);
    EXPECT_EQ(CellState_Constructing, prevConstructedCell.getCellRef()._cellState);
    EXPECT_EQ(CellState_Constructing, prevPrevConstructedCell.getCellRef()._cellState);
}

TEST_F(ConstructorTests, creature_1__node_2_3__concatenation_0_1__branch_0_0__frontAngle_lowerSide)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().shape(ConstructorShape_Triangle).nodes({NodeDesc(), NodeDesc(), NodeDesc().constructor(ConstructorGenomeDesc().separation(true))}),
    });
    auto data = Desc()
                    .addCreature(
                        {
                            ObjectDesc()
                                .id(1)
                                .pos({10.0f, 10.0f})
                                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(2).separation(true))),
                        },
                        CreatureDesc().id(0),
                        genome)
                    .addCreature(
                        {
                            ObjectDesc().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(1)),
                            ObjectDesc().id(3).pos({10.0f + getOffspringDistance(), 11.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
                        },
                        CreatureDesc().id(1),
                        genome);

    // Order of connections is unusual and not as it would be in a normal construction process
    // Thi is intentional to test that the ConstructionProcessor does not produce an invalid state
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(3, actualData.getObjectsForCreature(newCreature._id).size());

    auto actualConstructedCell = actualData.getOtherObjectRef({1, 2, 3});
    auto hostCell = actualData.getObjectRef(1);
    auto prevConstructedCell = actualData.getObjectRef(2);
    auto prevPrevConstructedCell = actualData.getObjectRef(3);

    EXPECT_TRUE(hostCell._connections.empty());

    EXPECT_TRUE(actualData.hasConnection(2, 3));
    EXPECT_TRUE(actualData.hasConnection(2, actualConstructedCell._id));

    EXPECT_TRUE(actualData.hasConnection(3, 2));
    EXPECT_TRUE(actualData.hasConnection(2, actualConstructedCell._id));

    EXPECT_TRUE(actualData.hasConnection(actualConstructedCell._id, 2));
    EXPECT_TRUE(actualData.hasConnection(actualConstructedCell._id, 3));

    EXPECT_EQ(CellState_Activating, actualConstructedCell.getCellRef()._cellState);
    EXPECT_EQ(CellState_Constructing, prevConstructedCell.getCellRef()._cellState);
    EXPECT_EQ(CellState_Constructing, prevPrevConstructedCell.getCellRef()._cellState);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_1__gene_0)
{
    auto data = Desc().addCreature(
        {ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(false).numBranches(1)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, actualData.getObjectsForCreature(hostCreature._id).size());
    ASSERT_EQ(2, hostCreature._numCells);

    auto hostObject = actualData.getObjectRef(0);
    auto newObject = actualData.getOtherObjectRef(0);
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(newObject.getCellRef()._headCell);
    EXPECT_TRUE(approxCompare(0.5f, Math::length(hostObject._pos - newObject._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = hostObject.getCellRef()._constructor.value();
    EXPECT_EQ(0, hostConstructor._currentOffspring);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_1__gene_1)
{
    auto data = Desc().addCreature(
        {ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(1).separation(false).numBranches(1)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc(),
            GeneDesc().nodes({NodeDesc()}),
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
}

TEST_F(ConstructorTests, creature_1__node_0_2__concatenation_0_1__branch_0_1)
{
    auto data = Desc().addCreature(
        {ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(false).numBranches(1).numConcatenations(1)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc(), NodeDesc()}),
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
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_2__branch_0_1)
{
    auto data = Desc().addCreature(
        {ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(false).numBranches(1).numConcatenations(2)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc()}),
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
}

class ConstructorTests_BendingMuscles
    : public ConstructorTests
    , public testing::WithParamInterface<MuscleMode>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_BendingMuscles,
    ConstructorTests_BendingMuscles,
    ::testing::Values(MuscleMode_AutoBending, MuscleMode_ManualBending, MuscleMode_AngleBending));

TEST_P(ConstructorTests_BendingMuscles, creature_3__node_0_1__concatenation_1_2__branch_0_1__resetBendingMuscle)
{
    auto muscleModeType = GetParam();

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
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc()),
            ObjectDesc()
                .id(1)
                .pos({100.0f, 101.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(3).separation(false).numBranches(1).numConcatenations(2))),
            ObjectDesc().id(2).pos({100.0f, 102.0f}).type(CellDesc()),
            ObjectDesc()
                .id(3)
                .pos({100.0f + getOffspringDistance(), 101.0f})
                .type(CellDesc().cellType(MuscleDesc().mode(muscleMode)).cellState(CellState_Constructing).nodeIndex(0)),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc().cellType(MuscleGenomeDesc())}),
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
    ASSERT_EQ(5, actualData.getObjectsForCreature(hostCreature._id).size());

    auto hostObject = actualData.getObjectRef(1);
    auto prevCell = actualData.getObjectRef(3);
    auto newObject = actualData.getOtherObjectRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newObject.getCellRef()._cellState);
    EXPECT_TRUE(approxCompare(hostObject._pos + RealVector2D(0.5f, 0.0f), newObject._pos));

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);

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
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(false).numBranches(2))),
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
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_1_2)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(false).numBranches(2))),
            ObjectDesc().id(1).pos({100.0f, 101.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_1_2__branch_0_1)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(1).separation(false).numBranches(1).numConcatenations(2))),
            ObjectDesc().id(1).pos({100.0f, 101.0f}).type(CellDesc().nodeIndex(0)),
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

    ASSERT_TRUE(actualData.hasConnection(hostObject._id, newObject._id));

    auto connection = actualData.getConnection(hostObject, newObject);
    EXPECT_EQ(1.0f, connection._distance);
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_0_0)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(true))),
            ObjectDesc().id(1).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc()}),
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
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(false).numBranches(1))),
            ObjectDesc().id(1).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc()}),
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
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({101.0f, 100.0f}),
            ObjectDesc()
                .id(1)
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().geneIndex(0).constructionAngle(10.0f).separation(false).numBranches(2)))  // constructionAngle should be ignored for the second branch
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
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(false).numBranches(1))),
            ObjectDesc().id(2).pos({100.0f, 101.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc().referenceAngle(ConstructionAngle)}),
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
}

TEST_F(ConstructorTests, creature_1__node_1_2__concatenation_0_1__branch_0_0)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(1).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc(), NodeDesc()}),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({99.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc(), NodeDesc()}),
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
        GeneDesc().nodes({NodeDesc(), NodeDesc().referenceAngle(LastAngle)}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(1).separation(false).numBranches(1))),
            ObjectDesc().id(1).pos({99.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
        GeneDesc().nodes({NodeDesc(), NodeDesc().referenceAngle(MiddleAngle)}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({101.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(3).separation(false).numBranches(1))),
            ObjectDesc().id(2).pos({100.0f, 101.0f}),
            ObjectDesc()
                .id(3)
                .pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * 1.0f)
                .type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
        GeneDesc().nodes({NodeDesc(), NodeDesc().referenceAngle(MiddleAngle)}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 101.0f}),
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(3).separation(false).numBranches(1))),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing)),
            ObjectDesc()
                .id(3)
                .pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * 1.0f)
                .type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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

TEST_F(ConstructorTests, creature_1__node_1_3__concatenation_0_1__branch_0_0)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(1).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc().referenceAngle(0.0f), NodeDesc().referenceAngle(45.0f), NodeDesc().referenceAngle(0.0f)}),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({99.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc().referenceAngle(0.0f), NodeDesc().referenceAngle(45.0f), NodeDesc().referenceAngle(0.0f)}),
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
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(newObject, hostObject)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_1_3__branch_0_1__concatenationAngle)
{
    auto const ConcatenationAngle = 20.0f;

    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().referenceAngle(ConcatenationAngle)}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(1).separation(false).numBranches(1).numConcatenations(3))),
            ObjectDesc().id(1).pos({99.0f, 100.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_inf__branch_0_0)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
    });
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).separation(true).numConcatenations(ConstructorDesc::NumConcatenations_Infinite))),
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
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_1_inf__branch_0_0)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({100.0f, 100.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(0).lastConstructedCellId(1).separation(true).numConcatenations(ConstructorDesc::NumConcatenations_Infinite))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc()}),
        }));
    data.addCreature(
        {
            ObjectDesc().id(1).pos({101.0f, 100.0f}).type(CellDesc().cellState(CellState_Ready).nodeIndex(0)),
        },
        CreatureDesc().id(1),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc()}),
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
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1__largeConstructionAngle)
{
    auto const ConstructionAngle = 180.0f;

    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 99.0f}),
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .usableEnergy(getConstructorEnergy())
                          .constructor(ConstructorDesc().constructionAngle(ConstructionAngle).geneIndex(0).autoTriggerInterval(100).separation(false).numBranches(1))),
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
        GeneDesc().nodes({NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}),
            ObjectDesc()
                .id(2)
                .pos({9.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().autoTriggerInterval(1).geneIndex(0).separation(false))),
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
        GeneDesc().nodes({NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({8.0f, 10.0f}),
            ObjectDesc()
                .id(2)
                .pos({9.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().autoTriggerInterval(1).geneIndex(0).separation(false))),
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
            ConstructorShape_Tube,
            ConstructorShape_LargeLolli,
            ConstructorShape_SmallLolli,
            ConstructorShape_Zigzag),
        ::testing::Values(ConstructionType::Normal, ConstructionType::Seed),
        ::testing::Values(0.5f, 1.0f, 1.5f)));

TEST_P(ConstructorTests_AllShapes, generateShape_genericCheck)
{
    auto const ConstructionAngle = 8.0f;
    auto const LastAngle = -5.0f;
    auto const n = 20;
    auto const AutoTriggerInterval = 20;

    auto [shape, type, connectionDistance] = GetParam();

    auto gene = GeneDesc().shape(shape).connectionDistance(connectionDistance);
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
                ObjectDesc().id(0).pos({100.0f, 100.0f - 5.0f}),
                ObjectDesc().id(1).pos({100.0f, 100.0f - 4.5f}),
                ObjectDesc().id(2).pos({100.0f, 100.0f - 4.0f}),
                ObjectDesc().id(3).pos({100.0f, 100.0f - 3.5f}),
                ObjectDesc().id(4).pos({100.0f, 100.0f - 3.0f}),
                ObjectDesc().id(5).pos({100.0f, 100.0f - 2.5f}),
                ObjectDesc().id(6).pos({100.0f, 100.0f - 2.0f}),
                ObjectDesc().id(7).pos({100.0f, 100.0f - 1.5f}),
                ObjectDesc().id(8).pos({100.0f, 100.0f - 1.0f}),
                ObjectDesc().id(9).pos({100.0f, 100.0f - 0.5f}),
                ObjectDesc()
                    .id(10)
                    .pos({100.0f, 100.0f})
                    .type(CellDesc()
                              .headCell(true)
                              .usableEnergy(getConstructorEnergy() * n)
                              .constructor(ConstructorDesc().constructionAngle(ConstructionAngle).geneIndex(0).autoTriggerInterval(AutoTriggerInterval).separation(false))),
                ObjectDesc().id(11).pos({100.1f, 100.0f + 0.5f}),
                ObjectDesc().id(12).pos({100.1f, 100.0f + 1.0f}),
                ObjectDesc().id(13).pos({100.1f, 100.0f + 1.5f}),
                ObjectDesc().id(14).pos({100.1f, 100.0f + 2.0f}),
                ObjectDesc().id(15).pos({100.1f, 100.0f + 2.5f}),
                ObjectDesc().id(16).pos({100.1f, 100.0f + 3.0f}),
                ObjectDesc().id(17).pos({100.1f, 100.0f + 3.5f}),
                ObjectDesc().id(18).pos({100.1f, 100.0f + 4.0f}),
                ObjectDesc().id(19).pos({100.1f, 100.0f + 4.5f}),
                ObjectDesc().id(20).pos({100.1f, 100.0f + 5.0f}),
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
                              .constructor(ConstructorDesc().constructionAngle(ConstructionAngle).geneIndex(0).autoTriggerInterval(AutoTriggerInterval).separation(false))),
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
    ShapeGenerator shapeGenerator;
    for (int i = 0; i < n; ++i) {
        auto shapeResult = shapeGenerator.generateNextConstructionData(shape);
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

TEST_F(ConstructorTests, generateTriangle_67cells_withSeparation)
{
    auto const n = 67;
    auto const AutoTriggerInterval = 10;

    auto gene = GeneDesc().shape(ConstructorShape_Triangle);
    for (int i = 0; i < n; ++i) {
        gene._nodes.emplace_back(NodeDesc());
    }
    auto genome = GenomeDesc().genes({gene});

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(10)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .headCell(true)
                          .usableEnergy(getConstructorEnergy() * n)
                          .constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(AutoTriggerInterval).separation(true).numBranches(1))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);

    // Give the constructor time to build the entire triangle
    Desc actualData;
    int retryCount = 0;
    do {
        _simulationFacade->calcTimesteps(AutoTriggerInterval);
        actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

        if (++retryCount == n * 10) {
            FAIL() << "Failed to construct triangle";
        }
    } while (actualData._objects.size() != n + 1);

    // Check 1: Verify the offspring creature has exactly 67 cells
    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto offspringCreature = actualData.getOtherCreatureRef(0);
    auto offspringCells = actualData.getObjectsForCreature(offspringCreature._id);
    ASSERT_EQ(n, offspringCells.size());

    // Check 2: Verify the triangular connection topology
    // With edges 0-10, total nodes = sum of max(k+1, 2) for k=0..10 = 2+2+3+4+5+6+7+8+9+10+11 = 67.
    // Expected topology:
    // - 36 interior nodes with 6 connections each
    // - 27 boundary edge nodes with 4 connections each
    // - 2 corner nodes with 2 connections each
    // - 1 transition node with 3 connections (second-to-last node)
    // - 1 tail node with 1 connection (last constructed node)
    int conn1 = 0, conn2 = 0, conn3 = 0, conn4 = 0, conn6 = 0;
    for (auto const& cell : offspringCells) {
        auto numConnections = toInt(cell._connections.size());
        if (numConnections == 1) {
            ++conn1;
        } else if (numConnections == 2) {
            ++conn2;
        } else if (numConnections == 3) {
            ++conn3;
        } else if (numConnections == 4) {
            ++conn4;
        } else if (numConnections == 6) {
            ++conn6;
        } else {
            FAIL() << "Unexpected connection count " << numConnections << " for cell " << cell._id;
        }
    }
    EXPECT_EQ(1, conn1);
    EXPECT_EQ(2, conn2);
    EXPECT_EQ(1, conn3);
    EXPECT_EQ(27, conn4);
    EXPECT_EQ(36, conn6);

    // Check 3: Verify angles in the constructed triangle
    // For interior cells (6 connections), each _angleFromPrevious should be approximately 60 degrees.
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 6) {
            for (auto const& conn : cell._connections) {
                EXPECT_TRUE(approxCompare(60.0f, conn._angleFromPrevious, 0.5f))
                    << "Interior cell " << cell._id << " has connection angle " << conn._angleFromPrevious << " (expected ~60)";
            }
        }
    }
}

TEST_F(ConstructorTests, generateRectangle_49cells_withSeparation)
{
    auto const n = 49;
    auto const AutoTriggerInterval = 10;

    auto gene = GeneDesc().shape(ConstructorShape_Rectangle);
    for (int i = 0; i < n; ++i) {
        gene._nodes.emplace_back(NodeDesc());
    }
    auto genome = GenomeDesc().genes({gene});

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(10)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .headCell(true)
                          .usableEnergy(getConstructorEnergy() * n)
                          .constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(AutoTriggerInterval).separation(true).numBranches(1))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);

    // Give the constructor time to build the entire rectangle
    Desc actualData;
    int retryCount = 0;
    do {
        _simulationFacade->calcTimesteps(AutoTriggerInterval);
        actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

        if (++retryCount == n * 10) {
            FAIL() << "Failed to construct rectangle";
        }
    } while (actualData._objects.size() != n + 1);

    // Check 1: Verify the offspring creature has exactly 49 cells
    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto offspringCreature = actualData.getOtherCreatureRef(0);
    auto offspringCells = actualData.getObjectsForCreature(offspringCreature._id);
    ASSERT_EQ(n, offspringCells.size());

    // Check 2: Verify the rectangular connection topology
    // A 7x7 rectangle has:
    // - 25 interior nodes with 4 connections each
    // - 20 edge nodes (non-corner) with 3 connections each
    // - 4 corner nodes with 2 connections each
    int numCorners = 0;
    int numEdgeNodes = 0;
    int numInnerNodes = 0;
    for (auto const& cell : offspringCells) {
        auto numConnections = toInt(cell._connections.size());
        if (numConnections == 2) {
            ++numCorners;
        } else if (numConnections == 3) {
            ++numEdgeNodes;
        } else if (numConnections == 4) {
            ++numInnerNodes;
        } else {
            FAIL() << "Unexpected connection count " << numConnections << " for cell " << cell._id;
        }
    }
    EXPECT_EQ(4, numCorners);
    EXPECT_EQ(20, numEdgeNodes);
    EXPECT_EQ(25, numInnerNodes);

    // Check 3: Verify angles in the constructed rectangle
    // For interior cells (4 connections), each _angleFromPrevious should be approximately 90 degrees.
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 4) {
            for (auto const& conn : cell._connections) {
                EXPECT_TRUE(approxCompare(90.0f, conn._angleFromPrevious, 0.5f))
                    << "Interior cell " << cell._id << " has connection angle " << conn._angleFromPrevious << " (expected ~90)";
            }
        }
    }

    // Check 4: Verify that the outer ring forms a single cycle of 24 nodes with 4 corners
    // Collect outer ring node IDs
    std::set<uint64_t> outerNodeIds;
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 2 || cell._connections.size() == 3) {
            outerNodeIds.insert(cell._id);
        }
    }

    // Build adjacency among outer ring nodes
    std::map<uint64_t, std::vector<uint64_t>> outerAdjacency;
    for (auto const& cell : offspringCells) {
        if (outerNodeIds.count(cell._id)) {
            for (auto const& conn : cell._connections) {
                if (outerNodeIds.count(conn._objectId)) {
                    outerAdjacency[cell._id].push_back(conn._objectId);
                }
            }
        }
    }

    // Each corner (2 connections total) should have exactly 2 outer ring neighbors
    // Each edge node (3 connections total) should have exactly 2 outer ring neighbors
    for (auto const& [nodeId, neighbors] : outerAdjacency) {
        EXPECT_EQ(2, neighbors.size()) << "Outer ring node " << nodeId << " should have exactly 2 outer ring neighbors";
    }

    // Traverse the outer ring to verify it forms a single cycle of 24 nodes with 4 corners.
    // Start from a corner node for clean edge counting.
    uint64_t startCorner = 0;
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 2) {
            startCorner = cell._id;
            break;
        }
    }

    {
        auto currentId = startCorner;
        uint64_t prevId = 0;
        int traversedNodes = 0;
        int cornersSeen = 0;
        int nodesSinceLastCorner = 0;
        std::vector<int> edgeLengths;

        do {
            ++traversedNodes;
            auto const& cell = actualData.getObjectRef(currentId);
            bool isCorner = cell._connections.size() == 2;
            if (isCorner) {
                ++cornersSeen;
                if (cornersSeen > 1) {
                    edgeLengths.push_back(nodesSinceLastCorner + 2);  // +2 for both corner endpoints
                }
                nodesSinceLastCorner = 0;
            } else {
                ++nodesSinceLastCorner;
            }

            auto const& neighbors = outerAdjacency.at(currentId);
            auto nextId = (neighbors.at(0) != prevId || traversedNodes == 1) ? neighbors.at(0) : neighbors.at(1);
            prevId = currentId;
            currentId = nextId;
        } while (currentId != startCorner && traversedNodes < 30);

        // Close the last edge (from last corner back to start corner)
        edgeLengths.push_back(nodesSinceLastCorner + 2);

        EXPECT_EQ(24, traversedNodes) << "Outer ring should form a cycle of 24 nodes";
        EXPECT_EQ(4, cornersSeen) << "Outer ring should have 4 corners";
        ASSERT_EQ(4, edgeLengths.size()) << "Should have 4 edges";
        for (int i = 0; i < toInt(edgeLengths.size()); ++i) {
            EXPECT_EQ(7, edgeLengths.at(i)) << "Edge " << i << " should have 7 nodes";
        }
    }
}

TEST_F(ConstructorTests, generateHexagon_61cells_withSeparation)
{
    auto const n = 61;
    auto const AutoTriggerInterval = 10;

    auto gene = GeneDesc().shape(ConstructorShape_Hexagon);
    for (int i = 0; i < n; ++i) {
        gene._nodes.emplace_back(NodeDesc());
    }
    auto genome = GenomeDesc().genes({gene});

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(10)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .headCell(true)
                          .usableEnergy(getConstructorEnergy() * n)
                          .constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(AutoTriggerInterval).separation(true).numBranches(1))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);

    // Give the constructor time to build the entire hexagon
    Desc actualData;
    int retryCount = 0;
    do {
        _simulationFacade->calcTimesteps(AutoTriggerInterval);
        actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

        if (++retryCount == n * 10) {
            FAIL() << "Failed to construct hexagon";
        }
    } while (actualData._objects.size() != n + 1);

    // Check 1: Verify the offspring creature has exactly 61 cells
    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto offspringCreature = actualData.getOtherCreatureRef(0);
    auto offspringCells = actualData.getObjectsForCreature(offspringCreature._id);
    ASSERT_EQ(n, offspringCells.size());

    // Check 2: Verify the hexagonal connection topology
    // A hexagon with 5 rings (ring 0 = center, ring 4 = outermost) has:
    // - 37 inner nodes (rings 0-3) with 6 connections each
    // - 18 outer edge nodes (non-corner, ring 4) with 4 connections each
    // - 6 outer corner nodes (ring 4) with 3 connections each
    int numCorners = 0;
    int numEdgeNodes = 0;
    int numInnerNodes = 0;
    for (auto const& cell : offspringCells) {
        auto numConnections = toInt(cell._connections.size());
        if (numConnections == 3) {
            ++numCorners;
        } else if (numConnections == 4) {
            ++numEdgeNodes;
        } else if (numConnections == 6) {
            ++numInnerNodes;
        } else {
            FAIL() << "Unexpected connection count " << numConnections << " for cell " << cell._id;
        }
    }
    EXPECT_EQ(6, numCorners);
    EXPECT_EQ(18, numEdgeNodes);
    EXPECT_EQ(37, numInnerNodes);

    // Check 3: Verify that each edge has 5 nodes (3 edge nodes + 2 corner endpoints)
    // An edge node (4 connections) must be connected to exactly 2 other edge/corner nodes on the outer ring.
    // A corner node (3 connections) must be connected to exactly 2 edge nodes.
    // Collect outer ring node IDs
    std::set<uint64_t> outerNodeIds;
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 3 || cell._connections.size() == 4) {
            outerNodeIds.insert(cell._id);
        }
    }

    // Build adjacency among outer ring nodes
    std::map<uint64_t, std::vector<uint64_t>> outerAdjacency;
    for (auto const& cell : offspringCells) {
        if (outerNodeIds.count(cell._id)) {
            for (auto const& conn : cell._connections) {
                if (outerNodeIds.count(conn._objectId)) {
                    outerAdjacency[cell._id].push_back(conn._objectId);
                }
            }
        }
    }

    // Each corner (3 connections total) should have exactly 2 outer ring neighbors
    // Each edge node (4 connections total) should have exactly 2 outer ring neighbors
    for (auto const& [nodeId, neighbors] : outerAdjacency) {
        EXPECT_EQ(2, neighbors.size()) << "Outer ring node " << nodeId << " should have exactly 2 outer ring neighbors";
    }

    // Traverse the outer ring to verify it forms a single cycle of 24 nodes with 6 corners.
    // Start from a corner node for clean edge counting.
    uint64_t startCorner = 0;
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 3) {
            startCorner = cell._id;
            break;
        }
    }

    {
        auto currentId = startCorner;
        uint64_t prevId = 0;
        int traversedNodes = 0;
        int cornersSeen = 0;
        int nodesSinceLastCorner = 0;
        std::vector<int> edgeLengths;

        do {
            ++traversedNodes;
            auto const& cell = actualData.getObjectRef(currentId);
            bool isCorner = cell._connections.size() == 3;
            if (isCorner) {
                ++cornersSeen;
                if (cornersSeen > 1) {
                    edgeLengths.push_back(nodesSinceLastCorner + 2);  // +2 for both corner endpoints
                }
                nodesSinceLastCorner = 0;
            } else {
                ++nodesSinceLastCorner;
            }

            auto const& neighbors = outerAdjacency.at(currentId);
            auto nextId = (neighbors.at(0) != prevId || traversedNodes == 1) ? neighbors.at(0) : neighbors.at(1);
            prevId = currentId;
            currentId = nextId;
        } while (currentId != startCorner && traversedNodes < 30);

        // Close the last edge (from last corner back to start corner)
        edgeLengths.push_back(nodesSinceLastCorner + 2);

        EXPECT_EQ(24, traversedNodes) << "Outer ring should form a cycle of 24 nodes";
        EXPECT_EQ(6, cornersSeen) << "Outer ring should have 6 corners";
        ASSERT_EQ(6, edgeLengths.size()) << "Should have 6 edges";
        for (int i = 0; i < edgeLengths.size(); ++i) {
            EXPECT_EQ(5, edgeLengths.at(i)) << "Edge " << i << " should have 5 nodes";
        }
    }
}

TEST_F(ConstructorTests, generateTube_19cells_withSeparation)
{
    auto const n = 19;
    auto const AutoTriggerInterval = 10;

    auto gene = GeneDesc().shape(ConstructorShape_Tube);
    for (int i = 0; i < n; ++i) {
        gene._nodes.emplace_back(NodeDesc());
    }
    auto genome = GenomeDesc().genes({gene});

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(10)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .headCell(true)
                          .usableEnergy(getConstructorEnergy() * n)
                          .constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(AutoTriggerInterval).separation(true).numBranches(1))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);

    // Give the constructor time to build the entire tube
    Desc actualData;
    int retryCount = 0;
    do {
        _simulationFacade->calcTimesteps(AutoTriggerInterval);
        actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

        if (++retryCount == n * 10) {
            FAIL() << "Failed to construct tube";
        }
    } while (actualData._objects.size() != n + 1);

    // Check 1: Verify the offspring creature has exactly 19 cells
    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto offspringCreature = actualData.getOtherCreatureRef(0);
    auto offspringCells = actualData.getObjectsForCreature(offspringCreature._id);
    ASSERT_EQ(n, offspringCells.size());

    // Check 2: Verify the tube connection topology
    // A tube with 7 columns (19 nodes) has:
    //   01  06  07  12  13  18
    // 00  02  05  08  11  14  17
    //   03  04  09  10  15  16
    //
    // - 5 interior middle-row nodes with 6 connections each
    // - 8 edge top/bottom nodes (non-end) with 4 connections each
    // - 6 end-cap nodes with 3 connections each
    int numEndCaps = 0;
    int numEdgeNodes = 0;
    int numInnerNodes = 0;
    for (auto const& cell : offspringCells) {
        auto numConnections = toInt(cell._connections.size());
        if (numConnections == 3) {
            ++numEndCaps;
        } else if (numConnections == 4) {
            ++numEdgeNodes;
        } else if (numConnections == 6) {
            ++numInnerNodes;
        } else {
            FAIL() << "Unexpected connection count " << numConnections << " for cell " << cell._id;
        }
    }
    EXPECT_EQ(6, numEndCaps);
    EXPECT_EQ(8, numEdgeNodes);
    EXPECT_EQ(5, numInnerNodes);

    // Check 3: Verify angles in the constructed tube
    // For interior cells (6 connections), each _angleFromPrevious should be approximately 60 degrees.
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 6) {
            for (auto const& conn : cell._connections) {
                EXPECT_TRUE(approxCompare(60.0f, conn._angleFromPrevious, 0.5f))
                    << "Interior cell " << cell._id << " has connection angle " << conn._angleFromPrevious << " (expected ~60)";
            }
        }
    }

    // Check 4: Verify that the outer ring forms a single cycle of 14 nodes with 6 corners
    // Collect outer ring node IDs
    std::set<uint64_t> outerNodeIds;
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 3 || cell._connections.size() == 4) {
            outerNodeIds.insert(cell._id);
        }
    }

    // Build adjacency among outer ring nodes
    std::map<uint64_t, std::vector<uint64_t>> outerAdjacency;
    for (auto const& cell : offspringCells) {
        if (outerNodeIds.count(cell._id)) {
            for (auto const& conn : cell._connections) {
                if (outerNodeIds.count(conn._objectId)) {
                    outerAdjacency[cell._id].push_back(conn._objectId);
                }
            }
        }
    }

    // Each outer ring node should have exactly 2 outer ring neighbors
    for (auto const& [nodeId, neighbors] : outerAdjacency) {
        EXPECT_EQ(2, neighbors.size()) << "Outer ring node " << nodeId << " should have exactly 2 outer ring neighbors";
    }

    // Traverse the outer ring to verify it forms a single cycle of 14 nodes with 6 corners.
    // Start from a corner node (degree 3) for clean edge counting.
    uint64_t startCorner = 0;
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 3) {
            startCorner = cell._id;
            break;
        }
    }

    {
        auto currentId = startCorner;
        uint64_t prevId = 0;
        int traversedNodes = 0;
        int cornersSeen = 0;
        int nodesSinceLastCorner = 0;
        std::vector<int> edgeLengths;

        do {
            ++traversedNodes;
            auto const& cell = actualData.getObjectRef(currentId);
            bool isCorner = cell._connections.size() == 3;
            if (isCorner) {
                ++cornersSeen;
                if (cornersSeen > 1) {
                    edgeLengths.push_back(nodesSinceLastCorner + 2);  // +2 for both corner endpoints
                }
                nodesSinceLastCorner = 0;
            } else {
                ++nodesSinceLastCorner;
            }

            auto const& neighbors = outerAdjacency.at(currentId);
            auto nextId = (neighbors.at(0) != prevId || traversedNodes == 1) ? neighbors.at(0) : neighbors.at(1);
            prevId = currentId;
            currentId = nextId;
        } while (currentId != startCorner && traversedNodes < 20);

        // Close the last edge (from last corner back to start corner)
        edgeLengths.push_back(nodesSinceLastCorner + 2);

        EXPECT_EQ(14, traversedNodes) << "Outer ring should form a cycle of 14 nodes";
        EXPECT_EQ(6, cornersSeen) << "Outer ring should have 6 corners";
        ASSERT_EQ(6, edgeLengths.size()) << "Should have 6 edges";

        // The tube has 2 long edges (top and bottom runs, each 6 nodes) and 4 short edges (end-cap transitions, each 2 nodes)
        std::sort(edgeLengths.begin(), edgeLengths.end());
        std::vector<int> expectedEdgeLengths = {2, 2, 2, 2, 6, 6};
        EXPECT_EQ(expectedEdgeLengths, edgeLengths);
    }
}

TEST_F(ConstructorTests, generateZigzag_24cells_withSeparation)
{
    auto const n = 24;
    auto const AutoTriggerInterval = 10;

    auto gene = GeneDesc().shape(ConstructorShape_Zigzag);
    for (int i = 0; i < n; ++i) {
        gene._nodes.emplace_back(NodeDesc());
    }
    auto genome = GenomeDesc().genes({gene});

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(10)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .headCell(true)
                          .usableEnergy(getConstructorEnergy() * n)
                          .constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(AutoTriggerInterval).separation(true).numBranches(1))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);

    // Give the constructor time to build the entire zigzag
    Desc actualData;
    int retryCount = 0;
    do {
        _simulationFacade->calcTimesteps(AutoTriggerInterval);
        actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

        if (++retryCount == n * 10) {
            FAIL() << "Failed to construct zigzag";
        }
    } while (actualData._objects.size() != n + 1);

    // Check 1: Verify the offspring creature has exactly 24 cells
    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto offspringCreature = actualData.getOtherCreatureRef(0);
    auto offspringCells = actualData.getObjectsForCreature(offspringCreature._id);
    ASSERT_EQ(n, offspringCells.size());

    // Check 2: Verify the zigzag connection topology
    // A zigzag with 24 nodes (3 full cycles of 8) has:
    //     02              10              18
    //   01  03          09  11          17  19
    // 00      04      08      12      16      20
    //           05  07          13  15          21  23
    //             06              14              22
    //
    // - 1 end node with 1 connection (node 0)
    // - 12 chain nodes with 2 connections
    // - 11 triangle nodes with 3 connections
    int numEndNodes = 0;
    int numChainNodes = 0;
    int numTriangleNodes = 0;
    for (auto const& cell : offspringCells) {
        auto numConnections = toInt(cell._connections.size());
        if (numConnections == 1) {
            ++numEndNodes;
        } else if (numConnections == 2) {
            ++numChainNodes;
        } else if (numConnections == 3) {
            ++numTriangleNodes;
        } else {
            FAIL() << "Unexpected connection count " << numConnections << " for cell " << cell._id;
        }
    }
    EXPECT_EQ(1, numEndNodes);
    EXPECT_EQ(12, numChainNodes);
    EXPECT_EQ(11, numTriangleNodes);

    // Check 3: Verify precise angles of all connections of all cells
    for (auto const& cell : offspringCells) {
        auto numConnections = toInt(cell._connections.size());
        if (numConnections == 1) {
            EXPECT_TRUE(approxCompare(360.0f, cell._connections.at(0)._angleFromPrevious, 0.5f))
                << "End cell " << cell._id << " has angle " << cell._connections.at(0)._angleFromPrevious << " (expected 360)";
        } else if (numConnections == 2) {
            std::vector<float> angles = {cell._connections.at(0)._angleFromPrevious, cell._connections.at(1)._angleFromPrevious};
            std::sort(angles.begin(), angles.end());
            auto isStraight = approxCompare(180.0f, angles.at(0), 0.5f) && approxCompare(180.0f, angles.at(1), 0.5f);
            auto isTriangleTip = approxCompare(60.0f, angles.at(0), 0.5f) && approxCompare(300.0f, angles.at(1), 0.5f);
            EXPECT_TRUE(isStraight || isTriangleTip) << "Chain cell " << cell._id << " has angles {" << angles.at(0) << ", " << angles.at(1)
                                                     << "} (expected {180, 180} or {60, 300})";
        } else if (numConnections == 3) {
            for (auto const& conn : cell._connections) {
                EXPECT_TRUE(approxCompare(120.0f, conn._angleFromPrevious, 0.5f))
                    << "Triangle cell " << cell._id << " has angle " << conn._angleFromPrevious << " (expected 120)";
            }
        }
    }

    // Check 4: Verify triangles
    // The zigzag forms 6 triangles: {1,2,3}, {5,6,7}, {9,10,11}, {13,14,15}, {17,18,19}, {21,22,23}
    // Each triangle consists of 3 mutually connected nodes.
    // Count triangles by finding groups of 3 nodes where each pair is connected.
    std::map<uint64_t, std::set<uint64_t>> adjacency;
    for (auto const& cell : offspringCells) {
        adjacency[cell._id] = {};
    }
    for (auto const& cell : offspringCells) {
        for (auto const& conn : cell._connections) {
            adjacency.at(cell._id).insert(conn._objectId);
        }
    }

    int numTriangles = 0;
    std::vector<uint64_t> cellIds;
    for (auto const& cell : offspringCells) {
        cellIds.push_back(cell._id);
    }
    for (int i = 0; i < toInt(cellIds.size()); ++i) {
        for (int j = i + 1; j < toInt(cellIds.size()); ++j) {
            for (int k = j + 1; k < toInt(cellIds.size()); ++k) {
                auto a = cellIds.at(i), b = cellIds.at(j), c = cellIds.at(k);
                if (adjacency.at(a).count(b) && adjacency.at(a).count(c) && adjacency.at(b).count(c)) {
                    ++numTriangles;
                }
            }
        }
    }
    EXPECT_EQ(6, numTriangles);
}

TEST_F(ConstructorTests, generateSmallLolli_19cells_withSeparation)
{
    auto const n = 19;
    auto const AutoTriggerInterval = 10;

    auto gene = GeneDesc().shape(ConstructorShape_SmallLolli);
    for (int i = 0; i < n; ++i) {
        gene._nodes.emplace_back(NodeDesc());
    }
    auto genome = GenomeDesc().genes({gene});

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(10)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .headCell(true)
                          .usableEnergy(getConstructorEnergy() * n)
                          .constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(AutoTriggerInterval).separation(true).numBranches(1))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);

    // Give the constructor time to build the entire small lolli
    Desc actualData;
    int retryCount = 0;
    do {
        _simulationFacade->calcTimesteps(AutoTriggerInterval);
        actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

        if (++retryCount == n * 10) {
            FAIL() << "Failed to construct small lolli";
        }
    } while (actualData._objects.size() != n + 1);

    // Check 1: Verify the offspring creature has exactly 19 cells
    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto offspringCreature = actualData.getOtherCreatureRef(0);
    auto offspringCells = actualData.getObjectsForCreature(offspringCreature._id);
    ASSERT_EQ(n, offspringCells.size());

    // Check 2: Verify angles in the hub node
    // The hub node (6 connections) should have approximately 60 degree angles between consecutive connections.
    for (auto const& cell : offspringCells) {
        if (cell._connections.size() == 6) {
            for (auto const& conn : cell._connections) {
                EXPECT_TRUE(approxCompare(60.0f, conn._angleFromPrevious, 0.5f))
                    << "Hub cell " << cell._id << " has connection angle " << conn._angleFromPrevious << " (expected ~60)";
            }
        }
    }

    // Check 3: Verify per-node connection counts match the expected shape
    // Build offspring in construction order and verify each node's expected connection count.
    // Expected connections per node:
    // node 00: 3 (chain to 01, required by 03, required by 04)
    // node 01: 3 (chain to 00, chain to 02, required by 03)
    // node 02: 3 (chain to 01, chain to 03, required by 06)
    // node 03: 6 (chain to 02, chain to 04, requires 01, requires 00, required by 05, required by 06)
    // node 04: 3 (chain to 03, chain to 05, requires 00)
    // node 05: 3 (chain to 04, chain to 06, requires 03)
    // node 06: 4 (chain to 05, chain to 07, requires 03, requires 02)
    // node 07-17: 2 (chain connections)
    // node 18: 1 (chain to 17 only, tail end)
    int expectedConnections[] = {3, 3, 3, 6, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1};

    // Sort offspring cells by id to get construction order
    auto sortedCells = offspringCells;
    std::sort(sortedCells.begin(), sortedCells.end(), [](auto const& a, auto const& b) { return a._id < b._id; });
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(expectedConnections[i], toInt(sortedCells.at(i)._connections.size()))
            << "Node " << i << " (cell id " << sortedCells.at(i)._id << ") has " << sortedCells.at(i)._connections.size() << " connections, expected "
            << expectedConnections[i];
    }
}

TEST_F(ConstructorTests, generateLargeLolli_25cells_withSeparation)
{
    auto const n = 25;
    auto const AutoTriggerInterval = 10;

    auto gene = GeneDesc().shape(ConstructorShape_LargeLolli);
    for (int i = 0; i < n; ++i) {
        gene._nodes.emplace_back(NodeDesc());
    }
    auto genome = GenomeDesc().genes({gene});

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(10)
                .pos({100.0f, 100.0f})
                .type(CellDesc()
                          .headCell(true)
                          .usableEnergy(getConstructorEnergy() * n)
                          .constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(AutoTriggerInterval).separation(true).numBranches(1))),
        },
        CreatureDesc().id(0),
        genome);

    _simulationFacade->setSimulationData(data);

    // Give the constructor time to build the entire large lolli
    Desc actualData;
    int retryCount = 0;
    do {
        _simulationFacade->calcTimesteps(AutoTriggerInterval);
        actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());

        if (++retryCount == n * 10) {
            FAIL() << "Failed to construct large lolli";
        }
    } while (actualData._objects.size() != n + 1);

    // Check 1: Verify the offspring creature has exactly 25 cells
    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());

    auto offspringCreature = actualData.getOtherCreatureRef(0);
    auto offspringCells = actualData.getObjectsForCreature(offspringCreature._id);
    ASSERT_EQ(n, offspringCells.size());

    // Check 2: Verify angles for all cells
    // Interior hex cells (6 connections): all angles should be ~60 degrees.
    // Boundary head cells (3-5 connections): angles should be multiples of ~60 degrees.
    // Tail cells (2 connections): angles should be ~180 degrees (straight line).
    for (auto const& cell : offspringCells) {
        auto numConnections = toInt(cell._connections.size());
        if (numConnections == 6) {
            for (auto const& conn : cell._connections) {
                EXPECT_TRUE(approxCompare(60.0f, conn._angleFromPrevious, 0.5f))
                    << "Interior cell " << cell._id << " has connection angle " << conn._angleFromPrevious << " (expected ~60)";
            }
        } else if (numConnections >= 3 && numConnections <= 5) {
            for (auto const& conn : cell._connections) {
                auto angle = conn._angleFromPrevious;
                bool isMultipleOf60 = approxCompare(60.0f, angle, 0.5f) || approxCompare(120.0f, angle, 0.5f) || approxCompare(180.0f, angle, 0.5f)
                    || approxCompare(240.0f, angle, 0.5f);
                EXPECT_TRUE(isMultipleOf60) << "Boundary cell " << cell._id << " (degree " << numConnections << ") has connection angle " << angle
                                            << " (expected multiple of ~60)";
            }
        } else if (numConnections == 2) {
            for (auto const& conn : cell._connections) {
                EXPECT_TRUE(approxCompare(180.0f, conn._angleFromPrevious, 0.5f))
                    << "Tail/edge cell " << cell._id << " has connection angle " << conn._angleFromPrevious << " (expected ~180)";
            }
        }
    }

    // Check 3: Verify per-node connection counts match the expected shape
    // The large lolli has a diamond head (19 nodes) + tail (6 nodes):
    //     00  01  08
    //   03  02  07  09
    // 04  05  06  11  10
    //   15  14  13  12
    //     16  17  18
    //               19
    //                 20
    //                   21
    //                     22
    //                       23
    //                         24
    // Interior hex nodes (6 conn): 02, 05, 06, 07, 11, 13, 14
    // Boundary head nodes: 00(3), 01(4), 03(4), 04(3), 08(3), 09(4), 10(3), 12(4), 15(4), 16(3), 17(4), 18(4)
    // Tail nodes: 19-23(2 each), 24(1 = tail end)
    int expectedConnections[] = {3, 4, 6, 4, 3, 6, 6, 6, 3, 4, 3, 6, 4, 6, 6, 4, 3, 4, 4, 2, 2, 2, 2, 2, 1};

    auto sortedCells = offspringCells;
    std::sort(sortedCells.begin(), sortedCells.end(), [](auto const& a, auto const& b) { return a._id < b._id; });
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(expectedConnections[i], toInt(sortedCells.at(i)._connections.size()))
            << "Node " << i << " (cell id " << sortedCells.at(i)._id << ") has " << sortedCells.at(i)._connections.size() << " connections, expected "
            << expectedConnections[i];
    }
}

TEST_F(ConstructorTests, avoidDeadlockByLockingNearObjects)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({}),
        GeneDesc().shape(ConstructorShape_Hexagon).nodes({NodeDesc(), NodeDesc(), NodeDesc(), NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}),
            ObjectDesc()
                .id(2)
                .pos({10.0f, 9.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(1).lastConstructedCellId(6).separation(false))),
            ObjectDesc().id(3).pos({11.0f, 9.0f}),
            ObjectDesc()
                .id(4)
                .pos({11.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).constructor(ConstructorDesc().geneIndex(1).lastConstructedCellId(8).separation(false))),
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
        GeneDesc().shape(ConstructorShape_Triangle).nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        GeneDesc().shape(ConstructorShape_Triangle).nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({10.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).nodeIndex(0).constructor(ConstructorDesc().geneIndex(1).lastConstructedCellId(4).separation(false))),
            ObjectDesc()
                .id(2)
                .pos({11.0f, 10.0f})
                .type(CellDesc().usableEnergy(getConstructorEnergy()).nodeIndex(1).constructor(ConstructorDesc().geneIndex(2).lastConstructedCellId(6).separation(false))),

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
        std::make_pair(ProvideEnergy_FreeGeneration, Separation::No),
        std::make_pair(ProvideEnergy_CellOnly, Separation::Yes),
        std::make_pair(ProvideEnergy_FreeGeneration, Separation::Yes)));

TEST_P(ConstructorTests_ProvideEnergy_Separation, provideEnergy_sufficientEnergy)
{
    auto [provideEnergy, separation] = GetParam();

    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc(),
        }),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto constructorEnergy = [&] {
        if (provideEnergy == ProvideEnergy_FreeGeneration) {
            return normalCellEnergy;
        }
        return normalCellEnergy * 2 + 1.0f;
    }();
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(constructorEnergy)
                          .constructor(ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
        EXPECT_TRUE(approxCompare(0.0f, getReservedEnergy(actualConstructedCell.getCellRef())));
        auto newConstructor = actualConstructedCell.getCellRef()._constructor.value();
        if (separation == Separation::Yes) {
            EXPECT_EQ(ProvideEnergy_CellOnly, newConstructor._provideEnergy);
        } else {
            EXPECT_EQ(ProvideEnergy_FreeGeneration, newConstructor._provideEnergy);
        }
    } else {
        EXPECT_TRUE(approxCompare(0.0f, actualConstructedCell.getCellRef()._rawEnergy));
        EXPECT_TRUE(approxCompare(0.0f, getReservedEnergy(actualConstructedCell.getCellRef())));
    }
}

TEST_P(ConstructorTests_ProvideEnergy_Separation, provideEnergy_insufficientEnergy)
{
    auto [provideEnergy, separation] = GetParam();

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        GTEST_SKIP() << "Skipping test because FreeGeneration always has enough energy.";
    }
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc(),
        }),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto constructorEnergy = normalCellEnergy * 2 - 1.0f;
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(constructorEnergy)
                          .constructor(ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false)),
            NodeDesc(),
        }),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto constructorEnergy = [&] {
        if (provideEnergy == ProvideEnergy_FreeGeneration) {
            return normalCellEnergy;
        }
        return normalCellEnergy * 2 + 1.0f;
    }();

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(constructorEnergy)
                          .constructor(ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).autoTriggerInterval(1).lastConstructedCellId(1))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
    EXPECT_TRUE(approxCompare(0, getReservedEnergy(actualData.getObjectRef(0).getCellRef())));

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        EXPECT_TRUE(approxCompare(0, getReservedEnergy(actualConstructedCell.getCellRef())));
        EXPECT_TRUE(approxCompare(normalCellEnergy, actualData.getObjectRef(0).getCellRef()._usableEnergy));
    } else {
        EXPECT_TRUE(approxCompare(0, getReservedEnergy(actualConstructedCell.getCellRef())));
    }
}

TEST_F(ConstructorTests, constructConstructorNodeWithReservedEnergy)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto reservedEnergy = normalCellEnergy * 2 * 2 * 3;
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).reservedEnergy(reservedEnergy).separation(false).numBranches(2).numConcatenations(3)),
            NodeDesc(),
        }),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(reservedEnergy + normalCellEnergy * 2 + 1.0f)
                          .constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(1).lastConstructedCellId(1).separation(false))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
    EXPECT_TRUE(approxCompare(0.0f, actualConstructedCell.getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(reservedEnergy, getReservedEnergy(actualConstructedCell.getCellRef())));
}

TEST_F(ConstructorTests, constructWithReservedEnergy)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Desc().addCreature(
        {ObjectDesc()
             .pos({100.0f, 100.0f})
             .type(CellDesc()
                       .usableEnergy(normalCellEnergy * 1.5)
                       .constructor(ConstructorDesc().autoTriggerInterval(3).reservedEnergy(normalCellEnergy / 2 + NEAR_ZERO).separation(true)))},
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(hostCreature._id).size());
    EXPECT_EQ(1, hostCreature._numCells);

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, actualData.getObjectsForCreature(newCreature._id).size());
    EXPECT_EQ(1, newCreature._numCells);

    auto hostObject = actualData.getObjectsForCreature(hostCreature._id).front();
    auto newObject = actualData.getObjectsForCreature(newCreature._id).front();
    EXPECT_TRUE(approxCompare(normalCellEnergy, hostObject.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(0, hostObject.getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(0, getReservedEnergy(hostObject.getCellRef())));

    EXPECT_TRUE(approxCompare(normalCellEnergy, newObject.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(0, newObject.getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(0, getReservedEnergy(newObject.getCellRef())));
}

class ConstructorTests_ProvideEnergy
    : public ConstructorTests
    , public testing::WithParamInterface<ProvideEnergy>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_ProvideEnergy,
    ConstructorTests_ProvideEnergy,
    ::testing::Values(ProvideEnergy_CellOnly, ProvideEnergy_FreeGeneration));

TEST_P(ConstructorTests_ProvideEnergy, provideEnergy_depotWithInitialStoredEnergy_sufficientEnergy)
{
    auto provideEnergy = GetParam();

    auto const InitialStoredUsableEnergy = 50.0f;

    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
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

        return normalCellEnergy * 2 + InitialStoredUsableEnergy + 1.0f;  // Contains energy for depot node in gene
    }();

    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(0)
                .pos({10.0f, 10.0f})
                .type(CellDesc()
                          .usableEnergy(constructorEnergy)
                          .constructor(ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).autoTriggerInterval(1).lastConstructedCellId(1).separation(true))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
        GeneDesc().nodes({
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
                          .constructor(ConstructorDesc().provideEnergy(provideEnergy).geneIndex(0).autoTriggerInterval(1).lastConstructedCellId(1).separation(true))),
            ObjectDesc().id(1).pos({10.0f + getOffspringDistance(), 10.0f}).type(CellDesc().cellState(CellState_Constructing).nodeIndex(0)),
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
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false))}),
        GeneDesc().nodes({
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
            ObjectDesc().id(i).pos({toFloat(i), 0.0f}).type(CellDesc().headCell(true).constructor(ConstructorDesc().geneIndex(0).autoTriggerInterval(30).separation(true))));
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

TEST_F(ConstructorTests, externalEnergyInflowOnlyForFirstOffspring_firstOffspring)
{
    _parameters.externalEnergyControlToggle.value = true;
    _parameters.externalEnergy.value = Infinity<float>::value;
    _parameters.externalEnergyInflowOnlyForFirstOffspring.value = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto normalEnergy = _parameters.normalCellEnergy.value[0];
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(normalEnergy).constructor(ConstructorDesc().geneIndex(0).currentOffspring(0).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes({NodeDesc()})}));

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
            ObjectDesc().id(0).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(normalEnergy).constructor(ConstructorDesc().geneIndex(0).currentOffspring(1).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({GeneDesc().nodes({NodeDesc()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_calcTimestepWithCellFunctions();

    auto actualData = _simulationFacade->getSimulationData();

    // Construction should fail because currentOffspring > 0 blocks energy inflow and cell has insufficient energy
    ASSERT_EQ(1, actualData._creatures.size());
}
