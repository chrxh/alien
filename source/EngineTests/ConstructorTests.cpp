#include <gtest/gtest.h>

#include "Base/Math.h"
#include "EngineInterface/NumberGenerator.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "EngineTestData/DescriptionTestDataFactory.h"

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
    
    float getConstructorEnergy() const { return _parameters.normalCellEnergy.value[0] * 2.5f; }

    bool compare(CellDescription const& cell, NodeDescription const& node)
    {
        if (cell._color != node._color) {
            return false;
        }
        if (!cell._neuralNetwork.has_value()) {
            return false;
        }
        for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
            if (cell._neuralNetwork->_weights[i] != node._neuralNetwork._weights[i]) {
                return false;
            }
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            if (cell._neuralNetwork->_biases[i] != node._neuralNetwork._biases[i]) {
                return false;
            }
            if (cell._neuralNetwork->_activationFunctions[i] != node._neuralNetwork._activationFunctions[i]) {
                return false;
            }
        }
        if (cell._signalRoutingRestriction._active != node._signalRoutingRestriction._active) {
            return false;
        }
        if (cell._signalRoutingRestriction._baseAngle != node._signalRoutingRestriction._baseAngle) {
            return false;
        }
        if (cell._signalRoutingRestriction._openingAngle != node._signalRoutingRestriction._openingAngle) {
            return false;
        }

        auto nodeType = node.getCellType();
        switch (cell.getCellType()) {
        case CellType_Base: {
            if (nodeType != CellTypeGenome_Base) {
                return false;
            }
        } break;
        case CellType_Depot: {
            if (nodeType != CellTypeGenome_Depot) {
                return false;
            }
            auto const& depot = std::get<DepotDescription>(cell._cellTypeData);
            auto const& nodeDepot = std::get<DepotGenomeDescription>(node._cellTypeData);
            if (depot._mode != nodeDepot._mode) {
                return false;
            }
        } break;
        case CellType_Constructor: {
            if (nodeType != CellTypeGenome_Constructor) {
                return false;
            }
            auto const& constructor = std::get<ConstructorDescription>(cell._cellTypeData);
            auto const& nodeConstructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
            if (constructor._autoTriggerInterval != nodeConstructor._autoTriggerInterval) {
                return false;
            }
            if (constructor._geneIndex != nodeConstructor._geneIndex) {
                return false;
            }
            if (constructor._constructionActivationTime != nodeConstructor._constructionActivationTime) {
                return false;
            }
        }
        break;
        case CellType_Sensor: {
            if (nodeType != CellTypeGenome_Sensor) {
                return false;
            }
            auto const& sensor = std::get<SensorDescription>(cell._cellTypeData);
            auto const& nodeSensor = std::get<SensorGenomeDescription>(node._cellTypeData);
            if (sensor._autoTriggerInterval != nodeSensor._autoTriggerInterval) {
                return false;
            }
            if (sensor._minDensity != nodeSensor._minDensity) {
                return false;
            }
            if (sensor._minRange != nodeSensor._minRange) {
                return false;
            }
            if (sensor._maxRange != nodeSensor._maxRange) {
                return false;
            }
            if (sensor._restrictToColor != nodeSensor._restrictToColor) {
                return false;
            }
            if (sensor._restrictToCreatures != nodeSensor._restrictToCreatures) {
                return false;
            }
        } break;
        case CellType_Generator: {
            if (nodeType != CellTypeGenome_Generator) {
                return false;
            }
            auto const& generator = std::get<GeneratorDescription>(cell._cellTypeData);
            auto const& nodeGenerator = std::get<GeneratorGenomeDescription>(node._cellTypeData);
            if (generator._autoTriggerInterval != nodeGenerator._autoTriggerInterval) {
                return false;
            }
            if (generator._pulseType != nodeGenerator._pulseType) {
                return false;
            }
            if (generator._alternationInterval != nodeGenerator._alternationInterval) {
                return false;
            }
        } break;
        case CellType_Attacker: {
            if (nodeType != CellTypeGenome_Attacker) {
                return false;
            }
        } break;
        case CellType_Injector: {
            if (nodeType != CellTypeGenome_Injector) {
                return false;
            }
            auto const& injector = std::get<InjectorDescription>(cell._cellTypeData);
            auto const& nodeInjector = std::get<InjectorGenomeDescription>(node._cellTypeData);
            if (injector._mode != nodeInjector._mode) {
                return false;
            }
        } break;
        case CellType_Muscle: {
            if (nodeType != CellTypeGenome_Muscle) {
                return false;
            }
            auto const& muscle = std::get<MuscleDescription>(cell._cellTypeData);
            auto const& nodeMuscle = std::get<MuscleGenomeDescription>(node._cellTypeData);
            if (muscle.getMode() != nodeMuscle.getMode()) {
                return false;
            }
            switch (muscle.getMode()) {
            case MuscleMode_AutoBending: {
                auto const& autoBending = std::get<AutoBendingDescription>(muscle._mode);
                auto const& nodeAutoBending = std::get<AutoBendingGenomeDescription>(nodeMuscle._mode);
                if (autoBending._maxAngleDeviation != nodeAutoBending._maxAngleDeviation) {
                    return false;
                }
                if (autoBending._frontBackVelRatio != nodeAutoBending._frontBackVelRatio) {
                    return false;
                }
            } break;
            case MuscleMode_ManualBending: {
                auto const& manualBending = std::get<ManualBendingDescription>(muscle._mode);
                auto const& nodeManualBending = std::get<ManualBendingGenomeDescription>(nodeMuscle._mode);
                if (manualBending._maxAngleDeviation != nodeManualBending._maxAngleDeviation) {
                    return false;
                }
                if (manualBending._frontBackVelRatio != nodeManualBending._frontBackVelRatio) {
                    return false;
                }
            } break;
            case MuscleMode_AngleBending: {
                auto const& angleBending = std::get<AngleBendingDescription>(muscle._mode);
                auto const& nodeAngleBending = std::get<AngleBendingGenomeDescription>(nodeMuscle._mode);
                if (angleBending._maxAngleDeviation != nodeAngleBending._maxAngleDeviation) {
                    return false;
                }
                if (angleBending._frontBackVelRatio != nodeAngleBending._frontBackVelRatio) {
                    return false;
                }
            } break;
            case MuscleMode_AutoCrawling: {
                auto const& autoCrawling = std::get<AutoCrawlingDescription>(muscle._mode);
                auto const& nodeAutoCrawling = std::get<AutoCrawlingGenomeDescription>(nodeMuscle._mode);
                if (autoCrawling._maxDistanceDeviation != nodeAutoCrawling._maxDistanceDeviation) {
                    return false;
                }
                if (autoCrawling._frontBackVelRatio != nodeAutoCrawling._frontBackVelRatio) {
                    return false;
                }
            } break;
            case MuscleMode_ManualCrawling: {
                auto const& manualCrawling = std::get<ManualCrawlingDescription>(muscle._mode);
                auto const& nodeManualCrawling = std::get<ManualCrawlingGenomeDescription>(nodeMuscle._mode);
                if (manualCrawling._maxDistanceDeviation != nodeManualCrawling._maxDistanceDeviation) {
                    return false;
                }
                if (manualCrawling._frontBackVelRatio != nodeManualCrawling._frontBackVelRatio) {
                    return false;
                }
            } break;
            case MuscleMode_DirectMovement: {
            } break;
            default:
                return false;
            }
        } break;
        case CellType_Defender: {
            if (nodeType != CellTypeGenome_Defender) {
                return false;
            }
            auto const& defender = std::get<DefenderDescription>(cell._cellTypeData);
            auto const& nodeDefender = std::get<DefenderGenomeDescription>(node._cellTypeData);
            if (defender._mode != nodeDefender._mode) {
                return false;
            }
        } break;
        case CellType_Reconnector: {
            if (nodeType != CellTypeGenome_Reconnector) {
                return false;
            }
            auto const& reconnector = std::get<ReconnectorDescription>(cell._cellTypeData);
            auto const& nodeReconnector = std::get<ReconnectorGenomeDescription>(node._cellTypeData);
            if (reconnector._restrictToColor != nodeReconnector._restrictToColor) {
                return false;
            }
            if (reconnector._restrictToCreatures != nodeReconnector._restrictToCreatures) {
                return false;
            }
        } break;
        case CellType_Detonator: {
            if (nodeType != CellTypeGenome_Detonator) {
                return false;
            }
            auto const& detonator = std::get<DetonatorDescription>(cell._cellTypeData);
            auto const& nodeDetonator = std::get<DetonatorGenomeDescription>(node._cellTypeData);
            if (detonator._countdown != nodeDetonator._countdown) {
                return false;
            }
        } break;
        default:
            return false;
        }
        return true;
    }

    DescriptionTestDataFactory* _descriptionTestDataFactory;
};

TEST_F(ConstructorTests, alreadyFinished)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().numBranches(1).nodes({NodeDescription()}),
                GeneDescription().numBranches(1).nodes({NodeDescription()}),
            }))
            .cells({
                CellDescription().id(0).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(1).currentBranch(1)).pos({100.0f, 100.0f}),
                CellDescription().id(1).pos({100.0f, 101.0f}),
            }),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(2, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, emptyGenome)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription())
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(0))
                    .pos({100.0f, 100.0f}),
            }),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, emptyGene)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({GeneDescription()}))
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(0))
                    .pos({100.0f, 100.0f}),
            }),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, nodeIndexOutOfRange)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({GeneDescription()}))
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(0).currentNodeIndex(1))
                    .pos({100.0f, 100.0f}),
            }),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, geneIndexOutOfRange)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({GeneDescription()}))
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellTypeData(ConstructorDescription().geneIndex(1).currentBranch(0).currentNodeIndex(0))
                    .pos({100.0f, 100.0f}),
            }),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, insufficientEnergy)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({GeneDescription()}))
            .cells({
                CellDescription()
                    .id(0)
                    .cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(0).currentNodeIndex(0))
                    .pos({100.0f, 100.0f}),
            }),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, lastConstructedCellNotFound)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({GeneDescription().numBranches(1).nodes({NodeDescription()})}))
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(0).lastConstructedCellId(1))
                    .pos({100.0f, 100.0f}),
            }),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

using NodeParameter = DescriptionTestDataFactory::NodeParameter;
class ConstructorTests_AllNodeTypes
    : public ConstructorTests
    , public testing::WithParamInterface<NodeParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    ConstructorTests_AllNodeTypes,
    ConstructorTests_AllNodeTypes,
    ::testing::Values(
        NodeParameter{CellTypeGenome_Base},
        NodeParameter{CellTypeGenome_Depot},
        NodeParameter{CellTypeGenome_Constructor},
        NodeParameter{CellTypeGenome_Sensor},
        NodeParameter{CellTypeGenome_Generator},
        NodeParameter{CellTypeGenome_Attacker},
        NodeParameter{CellTypeGenome_Injector},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_AutoBending},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_ManualBending},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_AngleBending},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_AutoCrawling},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_ManualCrawling},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_DirectMovement},
        NodeParameter{CellTypeGenome_Defender},
        NodeParameter{CellTypeGenome_Reconnector},
        NodeParameter{CellTypeGenome_Detonator}));

TEST_P(ConstructorTests_AllNodeTypes, creature_1__gene_0__node_0_1__concatenation_0_1__branch_0_0)
{
    auto const FrontAngle = 10.0f;

    auto nodeParameter = GetParam();
    auto randomNode = _descriptionTestDataFactory->createRandomNodeDescription(nodeParameter);

    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription()
                        .genes({
                            GeneDescription().nodes({randomNode}),
                        })
                        .frontAngle(FrontAngle))
            .cells({CellDescription()
                        .energy(getConstructorEnergy())
                        .cellTypeData(ConstructorDescription().geneIndex(0))
                        .pos({100.0f, 100.0f})
                        .angleToFront(FrontAngle)}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(FrontAngle, newCell._angleToFront));
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(compare(newCell, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_1__node_0_1__concatenation_0_1__branch_0_0)
{
    auto const FrontAngle = 10.0f;

    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription()
                        .genes({
                            GeneDescription().nodes({NodeDescription()}),
                            GeneDescription().nodes({NodeDescription()}),
                        })
                        .frontAngle(FrontAngle))
            .cells({CellDescription()
                        .id(0)
                        .energy(getConstructorEnergy())
                        .cellTypeData(ConstructorDescription().geneIndex(1))
                        .pos({100.0f, 100.0f})
                        .angleToFront(FrontAngle)}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(2, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCell(0);
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(FrontAngle, newCell._angleToFront));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_0_1__concatenation_0_1__branch_0_1)
{
    auto const FrontAngle = 10.0f;

    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription()
                        .genes({
                            GeneDescription().numBranches(1).nodes({NodeDescription()}),
                        })
                        .frontAngle(FrontAngle))
            .cells({CellDescription()
                        .energy(getConstructorEnergy())
                        .cellTypeData(ConstructorDescription().geneIndex(0))
                        .pos({100.0f, 100.0f})
                        .angleToFront(FrontAngle)}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(FrontAngle, newCell._angleToFront));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_0_2__concatenation_0_1__branch_0_1)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().numBranches(1).numConcatenations(1).nodes({NodeDescription(), NodeDescription()}),
            }))
            .cells({CellDescription()
                        .energy(getConstructorEnergy())
                        .cellTypeData(ConstructorDescription().geneIndex(0))
                        .pos({100.0f, 100.0f})}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(0, newCell._angleToFront));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f + _parameters.constructorAdditionalOffspringDistance, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_0_1__concatenation_0_2__branch_0_1)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().numBranches(1).numConcatenations(2).nodes({NodeDescription()}),
            }))
            .cells({CellDescription()
                        .energy(getConstructorEnergy())
                        .cellTypeData(ConstructorDescription().geneIndex(0).currentConcatenation(0))
                        .pos({100.0f, 100.0f})}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(0, newCell._angleToFront));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f + _parameters.constructorAdditionalOffspringDistance, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_0_1__concatenation_0_1__branch_1_2)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numBranches(2).nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription()
            .id(0)
            .genome(genome)
            .cells({CellDescription()
                        .id(0)
                        .energy(getConstructorEnergy())
                        .cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(1))
                        .pos({100.0f, 100.0f})}),

        // Offspring
        CreatureDescription().id(1).genome(genome).cells({CellDescription().id(1).pos({100.0f, 101.0f})}),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = actualData.getOtherCell({0, 1});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(0.0f, 1.0f), newCell._pos));
    EXPECT_TRUE(approxCompare(0, newCell._angleToFront));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(2, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_1__node_0_1__concatenation_0_1__branch_0_1)
{
    auto const FrontAngle = 10.0f;

    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription()
                        .genes({
                            GeneDescription().nodes({NodeDescription()}),
                            GeneDescription().numBranches(1).nodes({NodeDescription()}),
                        })
                        .frontAngle(FrontAngle))
            .cells({CellDescription()
                        .id(0)
                        .energy(getConstructorEnergy())
                        .cellTypeData(ConstructorDescription().geneIndex(1))
                        .pos({100.0f, 100.0f})
                        .angleToFront(FrontAngle)}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(2, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCell(0);
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(-180.0f + FrontAngle, newCell._angleToFront));
    EXPECT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_0_1__concatenation_0_1__branch_0_0__ignoreNumAdditionalConnectionsAtStart)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({
            NodeDescription().referenceAngle(0).numAdditionalConnections(1),
        }),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription().id(0).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0).currentNodeIndex(0)).pos({100.0f, 100.0f}),
        }),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCell({0});

    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_FALSE(actualData.hasConnection(newCell, hostCell));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_2__gene_0__node_0_1__concatenation_0_1__branch_0_0)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().nodes({NodeDescription()}),
            }))
            .cells({
                CellDescription().id(0).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f}),
                CellDescription().id(1).pos({101.0f, 100.0f}),
            }),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(1.0f, 0.0f), newCell._pos));
    EXPECT_FALSE(actualData.hasConnection(0, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(1, newCell._id));
    EXPECT_TRUE(actualData.hasConnection(0, 1));
}

TEST_F(ConstructorTests, creature_2__gene_0__node_0_1__concatenation_0_1__branch_0_1)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().numBranches(1).nodes({NodeDescription()}),
            }))
            .cells({
                CellDescription().id(0).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f}),
                CellDescription().id(1).pos({101.0f, 100.0f}),
            }),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(1.0f, 0.0f), newCell._pos));
    EXPECT_TRUE(actualData.hasConnection(0, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(1, newCell._id));
    EXPECT_TRUE(actualData.hasConnection(0, 1));
}

TEST_F(ConstructorTests, creature_2__gene_0__node_0_1__concatenation_0_1__branch_1_2)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numBranches(2).nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription().id(0).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(1)).pos({100.0f, 100.0f}),
            CellDescription().id(1).pos({101.0f, 100.0f}),
        }),

        // Offspring
        CreatureDescription().id(1).genome(genome).cells({CellDescription().id(2).pos({99.0f, 100.0f})}),
    });
    data.addConnection(0, 1);
    data.addConnection(0, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCell({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(0.0f, -1.0f), newCell._pos));
    EXPECT_TRUE(actualData.hasConnection(0, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(1, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(2, newCell._id));
    EXPECT_TRUE(actualData.hasConnection(0, 1));
    EXPECT_TRUE(actualData.hasConnection(0, 2));
}

TEST_F(ConstructorTests, creature_2__gene_1__node_0_1__concatenation_0_1__branch_1_2)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().numBranches(2).nodes({NodeDescription()}),
                GeneDescription().numBranches(2).nodes({NodeDescription()}),
            }))
            .cells(
                {CellDescription()
                     .id(0)
                     .energy(getConstructorEnergy())
                     .cellTypeData(ConstructorDescription().geneIndex(1).currentBranch(1))
                     .pos({100.0f, 100.0f}),
                 CellDescription().id(1).pos({101.0f, 100.0f}),
                 CellDescription().id(2).pos({99.0f, 100.0f})}),
    });
    data.addConnection(0, 1);
    data.addConnection(0, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(4, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCell({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(0.0f, -1.0f), newCell._pos));
    EXPECT_TRUE(actualData.hasConnection(0, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(1, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(2, newCell._id));
    EXPECT_TRUE(actualData.hasConnection(0, 1));
    EXPECT_TRUE(actualData.hasConnection(0, 2));
}

TEST_F(ConstructorTests, creature_3__gene_1__node_0_1__concatenation_0_1__branch_0_1)
{
    auto const FrontAngle = 10.0f;
    auto const ConstructionAngle = 20.0f;

    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription()
                        .genes({
                            GeneDescription().nodes({NodeDescription()}),
                            GeneDescription().numBranches(1).nodes({NodeDescription().referenceAngle(ConstructionAngle)}),
                        })
                        .frontAngle(FrontAngle))
            .cells({
                CellDescription().id(0).pos({101.0f, 100.0f}).angleToFront(FrontAngle),
                CellDescription()
                    .id(1)
                    .energy(getConstructorEnergy())
                    .cellTypeData(ConstructorDescription().geneIndex(1))
                    .pos({100.0f, 100.0f})
                    .angleToFront(FrontAngle - 180.0f),
                CellDescription().id(2).pos({100.0f, 101.0f}).angleToFront(FrontAngle - 90.0f),
            }),
    });
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(4, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCell({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(135.0f + FrontAngle - ConstructionAngle, newCell._angleToFront));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));
    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);
    EXPECT_TRUE(approxCompare(90.0f + 45.0f + ConstructionAngle, connection._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_1_2__concatenation_0_1__branch_0_0)
{
    auto const FrontAngle = 10.0f;

    auto genome = GenomeDescription()
                      .genes({
                          GeneDescription().nodes({NodeDescription(), NodeDescription()}),
                      })
                      .frontAngle(FrontAngle);
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
        }),

        // Offspring
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreature(1);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCell({0, 1});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(FrontAngle, newCell._angleToFront));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, newCell));
}

TEST_F(ConstructorTests, creature_1__gene_1__node_1_2__concatenation_0_1__branch_0_1)
{
    auto const FrontAngle = 10.0f;
    auto const LastAngle = 45.0f;
    auto genome = GenomeDescription()
                      .genes({
                          GeneDescription().nodes({NodeDescription()}),
                          GeneDescription().nodes({NodeDescription(), NodeDescription().referenceAngle(LastAngle)}).numBranches(1),
                      })
                      .frontAngle(FrontAngle);
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(1).currentNodeIndex(1).lastConstructedCellId(1))
                .pos({100.0f, 100.0f})
                .angleToFront(FrontAngle),
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }),

    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData._cells.size());
        ASSERT_EQ(1, actualData._creatures.size());
        EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

        auto creature = actualData.getCreature(0);
        ASSERT_EQ(3, creature._cells.size());

        auto hostCell = actualData.getCellRef(0);
        auto prevCell = actualData.getCellRef(1);
        auto newCell = actualData.getOtherCell({0, 1});
        EXPECT_EQ(CellState_Activating, newCell._cellState);
        EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
        EXPECT_TRUE(approxCompare(LastAngle + FrontAngle, newCell._angleToFront));
        EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
        EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
        EXPECT_TRUE(actualData.hasConnection(hostCell, newCell));
        EXPECT_TRUE(approxCompare(180.0f + LastAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));
    }

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();
        auto prevCell = actualData.getCellRef(1);
        EXPECT_EQ(CellState_Activating, prevCell._cellState);
        EXPECT_TRUE(approxCompare(LastAngle + FrontAngle - 180.0f, prevCell._angleToFront));

    }
}

TEST_F(ConstructorTests, creature_3__gene_1__node_1_2__concatenation_0_1__branch_0_1)
{
    auto const FrontAngle = 10.0f;
    auto const MiddleAngle = 5.0f;
    auto genome = GenomeDescription()
                      .genes({
                          GeneDescription().nodes({NodeDescription()}),
                          GeneDescription().nodes({NodeDescription(), NodeDescription().referenceAngle(MiddleAngle)}).numBranches(1),
                      })
                      .frontAngle(FrontAngle);
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription().id(0).pos({101.0f, 100.0f}).cellState(CellState_Constructing).angleToFront(FrontAngle),
            CellDescription()
                .id(1)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(1).currentNodeIndex(1).lastConstructedCellId(3))
                .pos({100.0f, 100.0f})
                .angleToFront(FrontAngle - 180.0f),
            CellDescription().id(2).pos({100.0f, 101.0f}).angleToFront(FrontAngle - 90.0f),
            CellDescription()
                .id(3)
                .pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * (1.0f + _parameters.constructorAdditionalOffspringDistance))
                .cellState(CellState_Constructing),
        }),

    });
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(5, creature._cells.size());

    auto hostCell = actualData.getCellRef(1);
    auto prevCell = actualData.getCellRef(3);
    auto newCell = actualData.getOtherCell({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(-45.0f + FrontAngle + MiddleAngle, newCell._angleToFront));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
    EXPECT_TRUE(actualData.hasConnection(hostCell, newCell));
    EXPECT_TRUE(approxCompare(180.0f + MiddleAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_3__gene_1__node_1_2__concatenation_0_1__branch_0_1__mirrored)
{
    auto const FrontAngle = 10.0f;
    auto const MiddleAngle = 5.0f;
    auto genome = GenomeDescription()
                      .genes({
                          GeneDescription().nodes({NodeDescription()}),
                          GeneDescription().nodes({NodeDescription(), NodeDescription().referenceAngle(MiddleAngle)}).numBranches(1),
                      })
                      .frontAngle(FrontAngle);
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription().id(0).pos({100.0f, 101.0f}).angleToFront(FrontAngle),
            CellDescription()
                .id(1)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(1).currentNodeIndex(1).lastConstructedCellId(3))
                .pos({100.0f, 100.0f})
                .angleToFront(FrontAngle - 180.0f),
            CellDescription().id(2).pos({101.0f, 100.0f}).cellState(CellState_Constructing).angleToFront(FrontAngle + 90.0f),
            CellDescription()
                .id(3)
                .pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * (1.0f + _parameters.constructorAdditionalOffspringDistance))
                .cellState(CellState_Constructing),
        }),

    });
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(5, creature._cells.size());

    auto hostCell = actualData.getCellRef(1);
    auto prevCell = actualData.getCellRef(3);
    auto newCell = actualData.getOtherCell({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(45.0f + FrontAngle + MiddleAngle, newCell._angleToFront));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
    EXPECT_TRUE(actualData.hasConnection(hostCell, newCell));
    EXPECT_TRUE(approxCompare(180.0f + MiddleAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_1__gene_0__node_1_3__concatenation_0_1__branch_0_0)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({
            NodeDescription().referenceAngle(0.0f), 
            NodeDescription().referenceAngle(45.0f), 
            NodeDescription().referenceAngle(0.0f)
        }),
    });
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
        }),

        // Offspring
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreature(1);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCell({0, 1});
    
    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(1, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(0, 1));
    EXPECT_TRUE(actualData.hasConnection(0, newCell._id));
    EXPECT_TRUE(approxCompare(180.0f + 45.0f, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(2, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_2_4__concatenation_0_1__branch_0_0__numAdditionalConnections_0)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({
            NodeDescription(),
            NodeDescription().referenceAngle(0),
            NodeDescription().referenceAngle(45.0f).numAdditionalConnections(0),
            NodeDescription(),
        }),
    });
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentNodeIndex(2).lastConstructedCellId(2))
                .pos({100.0f, 100.0f}),
        }),

        // Offspring
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 99.0f}).cellState(CellState_Constructing),
            CellDescription().id(2).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 0);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreature(1);
    ASSERT_EQ(3, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevPrevCell = actualData.getCellRef(1);
    auto prevCell = actualData.getCellRef(2);
    auto newCell = actualData.getOtherCell({0, 1, 2});

    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(prevPrevCell, prevCell));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_EQ(1, prevPrevCell._connections.size());
    EXPECT_EQ(2, prevCell._connections.size());
    EXPECT_EQ(2, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());
    EXPECT_TRUE(approxCompare(180.0f + 45.0f, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
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

TEST_P(ConstructorTests_AllAngleAlignments, creature_1__gene_0__node_2_4__concatenation_0_1__branch_0_0__numAdditionalConnections_1__angleAlignment)
{
    auto const NodeAngle = 5.0f;

    auto angleAlignment = GetParam();

    auto genome = GenomeDescription().genes({
        GeneDescription()
            .nodes({
                NodeDescription(),
                NodeDescription().referenceAngle(0),
                NodeDescription().referenceAngle(NodeAngle).numAdditionalConnections(1),
                NodeDescription(),
            })
            .angleAlignment(angleAlignment),
    });
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentNodeIndex(2).lastConstructedCellId(2))
                .pos({100.0f, 100.0f}),
        }),

        // Offspring
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 99.0f}).cellState(CellState_Constructing),
            CellDescription().id(2).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 0);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreature(1);
    ASSERT_EQ(3, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevPrevCell = actualData.getCellRef(1);
    auto prevCell = actualData.getCellRef(2);
    auto newCell = actualData.getOtherCell({0, 1, 2});

    EXPECT_EQ(CellState_Constructing, newCell._cellState);
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

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(3, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_0_1__concatenation_1_3__branch_0_1__concatenationAngle)
{
    auto const ConcatenationAngle = 20.0f;

    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription().referenceAngle(ConcatenationAngle)}).numConcatenations(3).numBranches(1),
    });
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentConcatenation(1).currentNodeIndex(0).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
        }),

        // Offspring
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreature(1);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCell({0, 1});

    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_EQ(1, prevCell._connections.size());
    EXPECT_EQ(2, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());
    EXPECT_TRUE(approxCompare(180.0f + ConcatenationAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(2, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_0_4__concatenation_1_2__branch_0_1__numAdditionalConnections_1)
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
            .numBranches(1)
            .angleAlignment(ConstructorAngleAlignment_90),
    });
    auto addDistance = _parameters.constructorAdditionalOffspringDistance;
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentConcatenation(1).currentNodeIndex(0).lastConstructedCellId(4))
                .pos({100.0f, 100.0f}),
        }),

        // Offspring
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(1).pos({100.0f + addDistance, 98.0f}).cellState(CellState_Constructing),
            CellDescription().id(2).pos({100.0f + addDistance, 99.0f}).cellState(CellState_Constructing),
            CellDescription().id(3).pos({101.0f + addDistance, 99.0f}).cellState(CellState_Constructing),
            CellDescription().id(4).pos({101.0f + addDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(0, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreature(1);
    ASSERT_EQ(5, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(4);
    auto prevPrevPrevCell = actualData.getCellRef(2);
    auto newCell = actualData.getOtherCell({0, 1, 2, 3, 4});

    EXPECT_EQ(CellState_Constructing, newCell._cellState);
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

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_0_1__concatenation_0_inf__branch_0_0)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numConcatenations(GeneDescription::NumConcatenations_Infinite).nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription().id(0).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0).currentNodeIndex(0)).pos({100.0f, 100.0f}),
        }),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCell({0});

    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__gene_0__node_0_1__concatenation_1_inf__branch_0_0)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numConcatenations(GeneDescription::NumConcatenations_Infinite).nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentNodeIndex(0).currentConcatenation(1).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
        }),

        // Offspring
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(1).pos({100.0f + _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Ready),
        }),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreature(0);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCell({0, 1});

    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, prevCell));
    EXPECT_EQ(1, prevCell._connections.size());
    EXPECT_EQ(2, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(2, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

// TODO Tests for different shape generators
// TODO Regression tests



//
//TEST_F(ConstructorTests, constructFurtherCell_connectToExistingCell_upperSide)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(1)}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.5f}).cellState(CellState_UnderConstruction),
//    });
//    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
//    data.addConnection(1, 2);
//    data.addConnection(2, 3, cell3_refPos);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//
//    ASSERT_EQ(4, actualData._cells.size());
//    auto actualHostCell = getCell(actualData, 1);
//    auto actualPrevConstructedCell = getCell(actualData, 2);
//    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
//
//    ASSERT_EQ(1, actualHostCell._connections.size());
//    ASSERT_EQ(3, actualConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//}
//
//TEST_F(ConstructorTests, constructFurtherCell_connectToExistingCell_bottomSide)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(1)}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 10.5f}).cellState(CellState_UnderConstruction),
//    });
//    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, -60.0f);
//    data.addConnection(1, 2);
//    data.addConnection(2, 3, cell3_refPos);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(4, actualData._cells.size());
//    auto actualHostCell = getCell(actualData, 1);
//    auto actualPrevConstructedCell = getCell(actualData, 2);
//    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
//
//    ASSERT_EQ(1, actualHostCell._connections.size());
//    ASSERT_EQ(3, actualConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//}
//
//TEST_F(ConstructorTests, constructFurtherCell_connectToExistingCell_bothSidesPresent)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(1)}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.5f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(4).pos({10.0f + getOffspringDistance(), 10.5f}).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(1, 2);
//    auto cell3_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
//    data.addConnection(2, 3, cell3_refPos);
//    auto cell4_refPos = RealVector2D(10.0f + getOffspringDistance(), 10.0f) + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f + 180.0f);
//    data.addConnection(2, 4, cell4_refPos);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(5, actualData._cells.size());
//    auto actualHostCell = getCell(actualData, 1);
//    auto actualPrevConstructedCell = getCell(actualData, 2);
//    auto actualUpperConstructedCell = getCell(actualData, 3);
//    auto actualLowerConstructedCell = getCell(actualData, 4);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});
//
//    ASSERT_EQ(1, actualHostCell._connections.size());
//    ASSERT_EQ(3, actualConstructedCell._connections.size());
//    ASSERT_EQ(3, actualPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualUpperConstructedCell._connections.size());
//    ASSERT_EQ(1, actualLowerConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualUpperConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualUpperConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualPrevConstructedCell, actualLowerConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualUpperConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualUpperConstructedCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualLowerConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//}
//
//
//TEST_F(ConstructorTests, constructFurtherCell_connectToExistingCell_threeCellsWithSmallAngles)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(2)}));
//
//    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f).cellState(CellState_UnderConstruction),
//        CellDescription().id(4).pos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(1, 2);
//    auto cell3_refPos = data._cells.at(1)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
//    data.addConnection(2, 3, cell3_refPos);
//    auto cell4_refPos = data._cells.at(2)._pos + Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);
//    data.addConnection(3, 4, cell4_refPos);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(5, actualData._cells.size());
//    auto actualHostCell = getCell(actualData, 1);
//    auto actualPrevConstructedCell = getCell(actualData, 2);
//    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
//    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});
//
//    ASSERT_EQ(1, actualHostCell._connections.size());
//    ASSERT_EQ(4, actualConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
//    ASSERT_EQ(3, actualPrevPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//}
//
//TEST_F(ConstructorTests, constructFurtherCell_connectToExistingCell_threeCellsWithSmallAngles2)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(2)}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({458.20f, 239.23f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({456.40f, 238.88f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos({455.96f, 239.75f})
//            .cellState(CellState_UnderConstruction),
//        CellDescription().id(4).pos({456.07f, 240.77f}).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(1, 2);
//    auto cell3_refPos = data._cells.at(1)._pos + Math::rotateClockwise(data._cells.at(0)._pos - data._cells.at(1)._pos, 120.0f);
//    data.addConnection(2, 3, cell3_refPos);
//    auto cell4_refPos = data._cells.at(2)._pos + Math::rotateClockwise(data._cells.at(1)._pos - data._cells.at(2)._pos, 120.0f);
//    data.addConnection(3, 4, cell4_refPos);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(5, actualData._cells.size());
//    auto actualHostCell = getCell(actualData, 1);
//    auto actualPrevConstructedCell = getCell(actualData, 2);
//    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
//    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});
//
//    ASSERT_EQ(1, actualHostCell._connections.size());
//    ASSERT_EQ(4, actualConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
//    ASSERT_EQ(3, actualPrevPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//}
//
//TEST_F(ConstructorTests, constructFurtherCell_connectToExistingCell_threeCellsWithSmallAngles_restrictAdditionalConnections)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(1)}));
//
//    auto offset = Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription()
//            .id(3)
//            .pos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f)
//            .cellState(CellState_UnderConstruction),
//        CellDescription().id(4).pos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(1, 2);
//    auto cell3_refPos = getCell(data, 2)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
//    data.addConnection(2, 3, cell3_refPos);
//    auto cell4_refPos = getCell(data, 3)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
//    data.addConnection(3, 4, cell4_refPos);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(5, actualData._cells.size());
//    auto actualHostCell = getCell(actualData, 1);
//    auto actualPrevConstructedCell = getCell(actualData, 2);
//    auto origPrevPrevConstructedCell = getCell(data, 3);
//    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
//    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});
//
//    ASSERT_EQ(1, actualHostCell._connections.size());
//    ASSERT_EQ(3, actualConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(300.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(60.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_EQ(origPrevPrevConstructedCell._connections, actualPrevPrevConstructedCell._connections);
//
//    EXPECT_TRUE(approxCompare(120.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(240.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//}
//
//TEST_F(ConstructorTests, constructFurtherCell_connectToExistingCell_90degAlignment)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false).angleAlignment(ConstructorAngleAlignment_90))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(1)}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(4).pos({10.0f + getOffspringDistance() - 1.0f, 9.0f - 0.2f}).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//    auto cell4_refPos = getCell(data, 3)._pos + RealVector2D(-1.0f, 0.0f);
//    data.addConnection(3, 4, cell4_refPos);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(5, actualData._cells.size());
//    auto actualHostCell = getCell(actualData, 1);
//    auto actualPrevConstructedCell = getCell(actualData, 2);
//    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
//    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});
//
//    ASSERT_EQ(1, actualHostCell._connections.size());
//    ASSERT_EQ(3, actualConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//}
//
//TEST_F(ConstructorTests, constructFurtherCell_connectToNoExistingCells_90degAlignment)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false).angleAlignment(ConstructorAngleAlignment_90))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(0)}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(4, actualData._cells.size());
//    auto actualHostCell = getCell(actualData, 1);
//    auto actualPrevConstructedCell = getCell(actualData, 2);
//    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
//
//    ASSERT_EQ(1, actualHostCell._connections.size());
//    ASSERT_EQ(2, actualConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
//    ASSERT_EQ(1, actualPrevPrevConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//}
//
//TEST_F(ConstructorTests, constructFurtherCell_connectToCellWithAngleSpace_90degAlignment)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false).angleAlignment(ConstructorAngleAlignment_90))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(1)}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 10.0f - 0.5f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(4).pos({10.0f + getOffspringDistance(), 10.0f - 1.0f}).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//    auto cell4_refPos = getCell(data, 3)._pos + RealVector2D(-1.0f, 0.0f);
//    data.addConnection(3, 4, cell4_refPos);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(5, actualData._cells.size());
//    auto actualHostCell = getCell(actualData, 1);
//    auto actualPrevConstructedCell = getCell(actualData, 2);
//    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
//    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});
//
//    ASSERT_EQ(1, actualHostCell._connections.size());
//    ASSERT_EQ(3, actualConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
//    ASSERT_EQ(2, actualPrevPrevPrevConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevPrevConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevPrevConstructedCell, actualPrevPrevPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(90.0f, getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(270.0f, getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
//}
//
//TEST_F(ConstructorTests, constructFurtherCell_onSpike)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numAdditionalConnections(0)}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription().id(1).pos({10.0f, 10.0f}),
//        CellDescription()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).genome(genome).lastConstructedCellId(3)),
//        CellDescription().id(3).pos({11.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(4, actualData._cells.size());
//    auto actualOtherCell = getCell(actualData, 1);
//    auto actualHostCell = getCell(actualData, 2);
//    auto actualPrevConstructedCell = getCell(actualData, 3);
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
//
//    ASSERT_EQ(1, actualOtherCell._connections.size());
//    ASSERT_EQ(2, actualHostCell._connections.size());
//    ASSERT_EQ(2, actualConstructedCell._connections.size());
//    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualOtherCell, actualHostCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualHostCell, actualOtherCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
//    EXPECT_TRUE(approxCompare(180.0f, getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));
//
//    EXPECT_TRUE(approxCompare(360.0f, getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
//}
//
//TEST_F(ConstructorTests, finishCreature_angleToFront_upperSide)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(true).frontAngle(45.0f))
//            .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy())}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(2).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(2, 3);
//    data.addConnection(1, 2);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(4);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(4, actualData._cells.size());
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
//    auto prevConstructedCell = getCell(actualData, 2);
//    auto prevPrevConstructedCell = getCell(actualData, 3);
//
//    EXPECT_EQ(CellState_Ready, actualConstructedCell._cellState);
//    EXPECT_TRUE(approxCompare(45.0f, actualConstructedCell._angleToFront));
//
//    EXPECT_EQ(CellState_Ready, prevConstructedCell._cellState);
//    EXPECT_TRUE(approxCompare(135.0f, prevConstructedCell._angleToFront));
//
//    EXPECT_EQ(CellState_Ready, prevPrevConstructedCell._cellState);
//    EXPECT_TRUE(approxCompare(-45.0f, prevPrevConstructedCell._angleToFront));
//}
//
//TEST_F(ConstructorTests, finishCreature_angleToFront_lowerSide)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(true).frontAngle(45.0f))
//            .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy())}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(2).autoTriggerInterval(1).genome(genome).lastConstructedCellId(2)),
//        CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_UnderConstruction),
//        CellDescription().id(3).pos({10.0f + getOffspringDistance(), 11.0f}).cellState(CellState_UnderConstruction),
//    });
//    data.addConnection(2, 3);
//    data.addConnection(1, 2);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(4);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(4, actualData._cells.size());
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
//    auto prevConstructedCell = getCell(actualData, 2);
//    auto prevPrevConstructedCell = getCell(actualData, 3);
//
//    EXPECT_EQ(CellState_Ready, actualConstructedCell._cellState);
//    EXPECT_TRUE(approxCompare(45.0f, actualConstructedCell._angleToFront));
//
//    EXPECT_EQ(CellState_Ready, prevConstructedCell._cellState);
//    EXPECT_TRUE(approxCompare(-45.0f, prevConstructedCell._angleToFront));
//
//    EXPECT_EQ(CellState_Ready, prevPrevConstructedCell._cellState);
//    EXPECT_TRUE(approxCompare(135.0f, prevPrevConstructedCell._angleToFront));
//}
//
//TEST_F(ConstructorTests, finishBodyPart_angleToFront_leftSide)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false).frontAngle(45.0f))
//            .cells({CellGenomeDescription()}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription().id(1).pos({10.0f, 10.0f}),
//        CellDescription()
//            .id(2)
//            .pos({9.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(0).autoTriggerInterval(1).genome(genome))
//            .angleToFront(-45.0f),
//        CellDescription().id(3).pos({9.0f, 11.0f}),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(2);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(4, actualData._cells.size());
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
//
//    EXPECT_EQ(CellState_Ready, actualConstructedCell._cellState);
//    EXPECT_TRUE(approxCompare(-90.0f, actualConstructedCell._angleToFront));
//}
//
//TEST_F(ConstructorTests, finishBodyPart_angleToFront_rightSide)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(false).frontAngle(-45.0f)).cells({CellGenomeDescription()}));
//
//    CollectionDescription data;
//    data.addCells({
//        CellDescription().id(1).pos({8.0f, 10.0f}),
//        CellDescription()
//            .id(2)
//            .pos({9.0f, 10.0f})
//            .energy(getConstructorEnergy())
//            .cellTypeData(ConstructorDescription().currentNodeIndex(0).autoTriggerInterval(1).genome(genome))
//            .angleToFront(45.0f),
//        CellDescription().id(3).pos({9.0f, 11.0f}),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(2);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(4, actualData._cells.size());
//    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});
//
//    EXPECT_EQ(CellState_Ready, actualConstructedCell._cellState);
//    EXPECT_TRUE(approxCompare(90.0f, actualConstructedCell._angleToFront));
//}
