#include <gtest/gtest.h>

#include "Base/Math.h"
#include "EngineInterface/NumberGenerator.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/ShapeGenerator.h"
#include "EngineInterface/SimulationFacade.h"

#include "EngineTestData/DescriptionTestDataFactory.h"

#include "TestFramework.h"

class ConstructorTests : public TestFramework
{
public:
    ConstructorTests()
        : TestFramework()
    {
        _descriptionTestDataFactory = &DescriptionTestDataFactory::get();
    }

    ~ConstructorTests() = default;

protected:
    
    float getConstructorEnergy() const { return _parameters.normalCellEnergy.value[0] * 2.5f; }
    float getOffspringDistance() const { return 1.0f + _parameters.constructorAdditionalOffspringDistance; }

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
        if (cell._signalRestriction._active != node._signalRestriction._active) {
            return false;
        }
        if (cell._signalRestriction._baseAngle != node._signalRestriction._baseAngle) {
            return false;
        }
        if (cell._signalRestriction._openingAngle != node._signalRestriction._openingAngle) {
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
            auto const& depot = std::get<DepotDescription>(cell._cellType);
            auto const& nodeDepot = std::get<DepotGenomeDescription>(node._cellType);
            if (depot._mode != nodeDepot._mode) {
                return false;
            }
        } break;
        case CellType_Constructor: {
            if (nodeType != CellTypeGenome_Constructor) {
                return false;
            }
            auto const& constructor = std::get<ConstructorDescription>(cell._cellType);
            auto const& nodeConstructor = std::get<ConstructorGenomeDescription>(node._cellType);
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
            auto const& sensor = std::get<SensorDescription>(cell._cellType);
            auto const& nodeSensor = std::get<SensorGenomeDescription>(node._cellType);
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
            auto const& generator = std::get<GeneratorDescription>(cell._cellType);
            auto const& nodeGenerator = std::get<GeneratorGenomeDescription>(node._cellType);
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
            auto const& injector = std::get<InjectorDescription>(cell._cellType);
            auto const& nodeInjector = std::get<InjectorGenomeDescription>(node._cellType);
            if (injector._mode != nodeInjector._mode) {
                return false;
            }
        } break;
        case CellType_Muscle: {
            if (nodeType != CellTypeGenome_Muscle) {
                return false;
            }
            auto const& muscle = std::get<MuscleDescription>(cell._cellType);
            auto const& nodeMuscle = std::get<MuscleGenomeDescription>(node._cellType);
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
                if (autoBending._forwardBackwardRatio != nodeAutoBending._forwardBackwardRatio) {
                    return false;
                }
            } break;
            case MuscleMode_ManualBending: {
                auto const& manualBending = std::get<ManualBendingDescription>(muscle._mode);
                auto const& nodeManualBending = std::get<ManualBendingGenomeDescription>(nodeMuscle._mode);
                if (manualBending._maxAngleDeviation != nodeManualBending._maxAngleDeviation) {
                    return false;
                }
                if (manualBending._forwardBackwardRatio != nodeManualBending._forwardBackwardRatio) {
                    return false;
                }
            } break;
            case MuscleMode_AngleBending: {
                auto const& angleBending = std::get<AngleBendingDescription>(muscle._mode);
                auto const& nodeAngleBending = std::get<AngleBendingGenomeDescription>(nodeMuscle._mode);
                if (angleBending._maxAngleDeviation != nodeAngleBending._maxAngleDeviation) {
                    return false;
                }
                if (angleBending._attractionRepulsionRatio != nodeAngleBending._attractionRepulsionRatio) {
                    return false;
                }
            } break;
            case MuscleMode_AutoCrawling: {
                auto const& autoCrawling = std::get<AutoCrawlingDescription>(muscle._mode);
                auto const& nodeAutoCrawling = std::get<AutoCrawlingGenomeDescription>(nodeMuscle._mode);
                if (autoCrawling._maxDistanceDeviation != nodeAutoCrawling._maxDistanceDeviation) {
                    return false;
                }
                if (autoCrawling._forwardBackwardRatio != nodeAutoCrawling._forwardBackwardRatio) {
                    return false;
                }
            } break;
            case MuscleMode_ManualCrawling: {
                auto const& manualCrawling = std::get<ManualCrawlingDescription>(muscle._mode);
                auto const& nodeManualCrawling = std::get<ManualCrawlingGenomeDescription>(nodeMuscle._mode);
                if (manualCrawling._maxDistanceDeviation != nodeManualCrawling._maxDistanceDeviation) {
                    return false;
                }
                if (manualCrawling._forwardBackwardRatio != nodeManualCrawling._forwardBackwardRatio) {
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
            auto const& defender = std::get<DefenderDescription>(cell._cellType);
            auto const& nodeDefender = std::get<DefenderGenomeDescription>(node._cellType);
            if (defender._mode != nodeDefender._mode) {
                return false;
            }
        } break;
        case CellType_Reconnector: {
            if (nodeType != CellTypeGenome_Reconnector) {
                return false;
            }
            auto const& reconnector = std::get<ReconnectorDescription>(cell._cellType);
            auto const& nodeReconnector = std::get<ReconnectorGenomeDescription>(node._cellType);
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
            auto const& detonator = std::get<DetonatorDescription>(cell._cellType);
            auto const& nodeDetonator = std::get<DetonatorGenomeDescription>(node._cellType);
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
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription().id(0).energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(1)).pos({100.0f, 100.0f}),
                CellDescription().id(1).pos({100.0f, 101.0f}),
            }),
    GenomeDescription().genes({
                GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
            }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, emptyGenome)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().geneIndex(0).currentBranch(0))
                    .pos({100.0f, 100.0f}),
            }),
    GenomeDescription());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, emptyGene)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().geneIndex(0).currentBranch(0))
                    .pos({100.0f, 100.0f}),
            }),
    GenomeDescription().genes({GeneDescription().separation(true)}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, nodeIndexOutOfRange)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().geneIndex(0).currentBranch(0).currentNodeIndex(1))
                    .pos({100.0f, 100.0f}),
            }),
    GenomeDescription().genes({GeneDescription().separation(true)}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, geneIndexOutOfRange)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().geneIndex(1).currentBranch(0).currentNodeIndex(0))
                    .pos({100.0f, 100.0f}),
            }),
    GenomeDescription().genes({GeneDescription().separation(true)}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, insufficientEnergy)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription()
                    .id(0)
                    .cellType(ConstructorDescription().geneIndex(0).currentBranch(0).currentNodeIndex(0))
                    .pos({100.0f, 100.0f}),
            }),
    GenomeDescription().genes({GeneDescription().separation(true).nodes({NodeDescription()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, lastConstructedCellNotFound)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().geneIndex(0).currentBranch(0).lastConstructedCellId(1))
                    .pos({100.0f, 100.0f}),
            }),
    GenomeDescription().genes({GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()})}));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, insufficientSpace)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentConcatenation(0).currentNodeIndex(1).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
        }), GenomeDescription().genes({         GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),     }));
    data.addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(1).pos({100.5f, 100.0f}).cellState(CellState_Constructing),
        }), GenomeDescription().genes({         GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),     }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
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

TEST_P(ConstructorTests_AllNodeTypes, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0)
{
    auto nodeParameter = GetParam();
    auto constexpr FrontAngleId = 5;

    auto randomNode = _descriptionTestDataFactory->createNonDefaultNodeDescription(nodeParameter);

    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({CellDescription().energy(getConstructorEnergy()).cellType(ConstructorDescription()).frontAngleId(FrontAngleId).pos({100.0f, 100.0f})}),
    GenomeDescription().genes({
                GeneDescription().separation(true).nodes({randomNode}),
            }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(newCell._isFrontAngleRefCell);
    EXPECT_EQ(FrontAngleId, newCell._frontAngleId);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(compare(newCell, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_P(ConstructorTests_AllNodeTypes, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0__preview)
{
    auto nodeParameter = GetParam();
    auto randomNode = _descriptionTestDataFactory->createNonDefaultNodeDescription(nodeParameter);

    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({CellDescription().energy(getConstructorEnergy()).cellType(ConstructorDescription()).pos({100.0f, 100.0f})}),
    GenomeDescription()
                        .genes({
                            GeneDescription().separation(true).nodes({randomNode}),
                        }));

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(1);

    auto actualData = _simulationFacade->getPreviewData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(newCell._isFrontAngleRefCell);
    EXPECT_TRUE(Math::length(hostCell._pos - newCell._pos) > 50.0f);  // Preview specific: Move seed far away from construction
    EXPECT_TRUE(compare(newCell, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_0__preview_detail)
{
    auto randomNode = _descriptionTestDataFactory->createNonDefaultNodeDescription(NodeParameter{CellTypeGenome_Base});

    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({CellDescription().energy(getConstructorEnergy()).cellType(ConstructorDescription()).pos({100.0f, 100.0f})}),
    GenomeDescription().genes({
                GeneDescription().separation(true).nodes({randomNode}),
            }));

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(1, true);

    auto actualData = _simulationFacade->getPreviewData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(newCell._isFrontAngleRefCell);
    EXPECT_TRUE(Math::length(hostCell._pos - newCell._pos) > 50.0f);  // Preview specific: Move seed far away from construction
    EXPECT_TRUE(compare(newCell, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_0__gene_1)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({CellDescription().energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(1)).pos({100.0f, 100.0f})}),
    GenomeDescription()
                        .genes({
                            GeneDescription().separation(true),
                            GeneDescription().separation(true).nodes({NodeDescription()}),
                        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_2_3__concatenation_0_1__branch_0_0__frontAngle_upperSide)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().currentNodeIndex(2).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2)),
        }), GenomeDescription()                       .genes({                           GeneDescription().separation(true).nodes(                               {NodeDescription(), NodeDescription(), NodeDescription().cellType(ConstructorGenomeDescription())}),                       }));
    data.addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing),
            CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).cellState(CellState_Constructing),
        }), GenomeDescription()                       .genes({                           GeneDescription().separation(true).nodes(                               {NodeDescription(), NodeDescription(), NodeDescription().cellType(ConstructorGenomeDescription())}),                       }));
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());
    ASSERT_EQ(3, newCreature._cells.size());

    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3});
    auto prevConstructedCell = actualData.getCellRef(2);
    auto prevPrevConstructedCell = actualData.getCellRef(3);

    EXPECT_EQ(CellState_Ready, actualConstructedCell._cellState);

    EXPECT_EQ(CellState_Ready, prevConstructedCell._cellState);

    EXPECT_EQ(CellState_Ready, prevPrevConstructedCell._cellState);
}

TEST_F(ConstructorTests, creature_1__node_2_3__concatenation_0_1__branch_0_0__frontAngle_lowerSide)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().currentNodeIndex(2).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2)),
        }), GenomeDescription()                       .genes({                           GeneDescription().separation(true).nodes(                               {NodeDescription(), NodeDescription(), NodeDescription().cellType(ConstructorGenomeDescription())}),                       }));
    data.addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing),
            CellDescription().id(3).pos({10.0f + getOffspringDistance(), 11.0f}).cellState(CellState_Constructing),
        }), GenomeDescription()                       .genes({                           GeneDescription().separation(true).nodes(                               {NodeDescription(), NodeDescription(), NodeDescription().cellType(ConstructorGenomeDescription())}),                       }));
    data.addConnection(2, 3);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());
    ASSERT_EQ(3, newCreature._cells.size());

    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3});
    auto prevConstructedCell = actualData.getCellRef(2);
    auto prevPrevConstructedCell = actualData.getCellRef(3);

    EXPECT_EQ(CellState_Ready, actualConstructedCell._cellState);

    EXPECT_EQ(CellState_Ready, prevConstructedCell._cellState);

    EXPECT_EQ(CellState_Ready, prevPrevConstructedCell._cellState);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_1__gene_0)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({CellDescription()
                        .id(0)
                        .energy(getConstructorEnergy())
                        .cellType(ConstructorDescription().geneIndex(0))
                        .pos({100.0f, 100.0f})}),
    GenomeDescription()
                        .genes({
                            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
                        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef(0);
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(newCell._isFrontAngleRefCell);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_0_1__gene_1)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({CellDescription()
                        .id(0)
                        .energy(getConstructorEnergy())
                        .cellType(ConstructorDescription().geneIndex(1))
                        .pos({100.0f, 100.0f})}),
    GenomeDescription()
                        .genes({
                            GeneDescription().separation(true),
                            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
                        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef(0);
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_2__concatenation_0_1__branch_0_1)
{
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .frontAngleId(InitialFrontAngleId)
            
            .cells({CellDescription().id(0).energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f})}),
    GenomeDescription().genes({
                GeneDescription().separation(false).numBranches(1).numConcatenations(1).nodes({NodeDescription(), NodeDescription()}),
            }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    EXPECT_EQ(InitialFrontAngleId, creature._frontAngleId);
    ASSERT_EQ(2, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef(0);
    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f + _parameters.constructorAdditionalOffspringDistance, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_2__branch_0_1)
{
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .frontAngleId(InitialFrontAngleId)
            
            .cells({CellDescription()
                        .id(0)
                        .energy(getConstructorEnergy())
                        .cellType(ConstructorDescription().geneIndex(0).currentConcatenation(0))
                        .pos({100.0f, 100.0f})}),
    GenomeDescription().genes({
                GeneDescription().separation(false).numBranches(1).numConcatenations(2).nodes({NodeDescription()}),
            }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    EXPECT_EQ(InitialFrontAngleId + 1, hostCreature._frontAngleId);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef(0);
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f + _parameters.constructorAdditionalOffspringDistance, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
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

    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .frontAngleId(InitialFrontAngleId)
            
            .cells({
                CellDescription().id(0).pos({100.0f, 100.0f}),
                CellDescription()
                    .id(1)
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().geneIndex(0).currentConcatenation(1).lastConstructedCellId(3))
                    .pos({100.0f, 101.0f}),
                CellDescription().id(2).pos({100.0f, 102.0f}),
                CellDescription()
                    .id(3)
                    .pos({100.0f + getOffspringDistance(), 101.0f}).cellType(MuscleDescription().mode(muscleMode)),
            }),
    GenomeDescription().genes({
                GeneDescription().separation(false).numBranches(1).numConcatenations(2).nodes({NodeDescription().cellType(MuscleGenomeDescription())}),
            }));
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    EXPECT_EQ(InitialFrontAngleId + 1, hostCreature._frontAngleId);
    ASSERT_EQ(5, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(1);
    auto prevCell = actualData.getCellRef(3);
    auto newCell = actualData.getOtherCellRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(hostCell._pos + RealVector2D(1.0f, 0.0f), newCell._pos));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);

    auto muscle = std::get<MuscleDescription>(prevCell._cellType);
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
    auto data = Description().addCreature(
        CreatureDescription().id(0).frontAngleId(InitialFrontAngleId).cells({
            CellDescription().id(0).energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(0)).pos({100.0f, 100.0f}),
        }),
    genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    EXPECT_EQ(InitialFrontAngleId + 1, hostCreature._frontAngleId);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef({0});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_1_2)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(2).nodes({NodeDescription()}),
    });
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(0).energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(1)).pos({100.0f, 100.0f}),
            CellDescription().id(1).pos({100.0f, 101.0f}),
        }),
    genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef({0, 1});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_FALSE(newCell._isFrontAngleRefCell);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(0.0f, 1.0f), newCell._pos));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(2, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_0_1__branch_1_2__firstBranchMissing)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(2).nodes({NodeDescription()}),
    });
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(0).energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentBranch(1)).pos({100.0f, 100.0f}),
        }),
    genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef({0});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
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
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(0).energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(0)).pos({100.0f, 100.0f}),
        }),
    genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef({0});

    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_FALSE(actualData.hasConnection(newCell, hostCell));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_0_0)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription().id(0).energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f}),
                CellDescription().id(1).pos({101.0f, 100.0f}),
            }),
    GenomeDescription().genes({
                GeneDescription().separation(true).nodes({NodeDescription()}),
            }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(hostCell._pos - RealVector2D(1.0f, 0.0f), newCell._pos));
    EXPECT_FALSE(actualData.hasConnection(0, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(1, newCell._id));
    EXPECT_TRUE(actualData.hasConnection(0, 1));
}

TEST_F(ConstructorTests, creature_2__node_0_1__concatenation_0_1__branch_0_1)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription().id(0).energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f}),
                CellDescription().id(1).pos({101.0f, 100.0f}),
            }),
    GenomeDescription().genes({
                GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
            }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef({0, 1});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
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
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(0).pos({101.0f, 100.0f}),
            CellDescription()
                .id(1)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentBranch(1).constructionAngle(10.0f))  // constructionAngle should be ignored for the second branch
                .pos({100.0f, 100.0f}),
            CellDescription().id(2).pos({99.0f, 100.0f}),
        }),
    genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCellRef({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
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

    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription().id(0).pos({101.0f, 100.0f}),
                CellDescription()
                    .id(1)
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().geneIndex(0))
                    .pos({100.0f, 100.0f}),
                CellDescription().id(2).pos({100.0f, 101.0f}),
            }),
    GenomeDescription()
                        .genes({
                            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription().referenceAngle(ConstructionAngle)}),
                        }));
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCellRef({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newCell._cellState);

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));
    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);
    EXPECT_TRUE(approxCompare(90.0f + 45.0f + ConstructionAngle, connection._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_1_2__concatenation_0_1__branch_0_0)
{
    auto const InitialFrontAngleId = 4;

    Description data;
    data.addCreature(CreatureDescription()
            .id(0)
            .cells({
                CellDescription()
                    .id(0)
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))
                    .pos({100.0f, 100.0f}),
            }), GenomeDescription().genes({         GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),     }));
    data.addCreature(CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId)
            
            .cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }), GenomeDescription().genes({         GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),     }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreatureRef(1);
    EXPECT_EQ(InitialFrontAngleId + 1, newCreature._frontAngleId);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCellRef({0, 1});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, newCell));
}

TEST_F(ConstructorTests, creature_1__node_1_2__concatenation_0_1__branch_0_1)
{
    auto const LastAngle = 45.0f;
    auto genome = GenomeDescription()
                      .genes({
                          GeneDescription().nodes({NodeDescription(), NodeDescription().referenceAngle(LastAngle)}).separation(false).numBranches(1),
                      });
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(0, actualData._cells.size());
        ASSERT_EQ(1, actualData._creatures.size());
        EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

        auto creature = actualData.getCreatureRef(0);
        ASSERT_EQ(3, creature._cells.size());

        auto hostCell = actualData.getCellRef(0);
        auto prevCell = actualData.getCellRef(1);
        auto newCell = actualData.getOtherCellRef({0, 1});
        EXPECT_EQ(CellState_Activating, newCell._cellState);
        EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
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

    }
}

TEST_F(ConstructorTests, creature_3__node_1_2__concatenation_0_1__branch_0_1)
{
    auto const MiddleAngle = 5.0f;
    auto const InitialFrontAngleId = 4;

    auto genome = GenomeDescription()
                      .genes({
                          GeneDescription().nodes({NodeDescription(), NodeDescription().referenceAngle(MiddleAngle)}).separation(false).numBranches(1),
                      });
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            
            .frontAngleId(InitialFrontAngleId)
            .cells({
                CellDescription().id(0).pos({101.0f, 100.0f}).cellState(CellState_Constructing),
                CellDescription()
                    .id(1)
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(3))
                    .pos({100.0f, 100.0f}),
                CellDescription().id(2).pos({100.0f, 101.0f}),
                CellDescription()
                    .id(3)
                    .pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * (1.0f + _parameters.constructorAdditionalOffspringDistance))
                    .cellState(CellState_Constructing),
            }),
    genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, creature._cells.size());
    EXPECT_EQ(InitialFrontAngleId + 1, creature._frontAngleId);

    auto hostCell = actualData.getCellRef(1);
    auto prevCell = actualData.getCellRef(3);
    auto newCell = actualData.getOtherCellRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(_parameters.constructorAdditionalOffspringDistance, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_FALSE(actualData.hasConnection(hostCell, prevCell));
    EXPECT_TRUE(actualData.hasConnection(hostCell, newCell));
    EXPECT_TRUE(approxCompare(180.0f + MiddleAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_3__node_1_2__concatenation_0_1__branch_0_1__mirrored)
{
    auto const MiddleAngle = 5.0f;
    auto genome = GenomeDescription()
                      .genes({
                          GeneDescription().nodes({NodeDescription(), NodeDescription().referenceAngle(MiddleAngle)}).separation(false).numBranches(1),
                      });
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(0).pos({100.0f, 101.0f}),
            CellDescription()
                .id(1)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(3))
                .pos({100.0f, 100.0f}),
            CellDescription().id(2).pos({101.0f, 100.0f}).cellState(CellState_Constructing),
            CellDescription()
                .id(3)
                .pos(RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(-45.0f) * (1.0f + _parameters.constructorAdditionalOffspringDistance))
                .cellState(CellState_Constructing),
        }),
    genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, creature._cells.size());

    auto hostCell = actualData.getCellRef(1);
    auto prevCell = actualData.getCellRef(3);
    auto newCell = actualData.getOtherCellRef({0, 1, 2, 3});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
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

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(1).pos({10.0f, 10.0f}),
            CellDescription()
                .id(2)
                .pos({11.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().currentNodeIndex(1).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(3)),
            CellDescription().id(3).pos({11.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing),
        }),
    genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, creature._cells.size());

    auto actualOtherCell = actualData.getCellRef(1);
    auto actualHostCell = actualData.getCellRef(2);
    auto actualPrevConstructedCell = actualData.getCellRef(3);
    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3});

    ASSERT_EQ(1, actualOtherCell._connections.size());
    ASSERT_EQ(2, actualHostCell._connections.size());
    ASSERT_EQ(2, actualConstructedCell._connections.size());
    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualOtherCell, actualHostCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualHostCell, actualOtherCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualHostCell, actualConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualHostCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, actualData.getConnection(actualConstructedCell, actualPrevConstructedCell)._angleFromPrevious));

    EXPECT_TRUE(approxCompare(360.0f, actualData.getConnection(actualPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_1__node_1_3__concatenation_0_1__branch_0_0)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(1).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
        }), GenomeDescription().genes({         GeneDescription().separation(true).nodes({             NodeDescription().referenceAngle(0.0f),              NodeDescription().referenceAngle(45.0f),              NodeDescription().referenceAngle(0.0f)         }),     }));
    data.addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }), GenomeDescription().genes({         GeneDescription().separation(true).nodes({             NodeDescription().referenceAngle(0.0f),              NodeDescription().referenceAngle(45.0f),              NodeDescription().referenceAngle(0.0f)         }),     }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCellRef({0, 1});
    
    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(1, newCell._id));
    EXPECT_FALSE(actualData.hasConnection(0, 1));
    EXPECT_TRUE(actualData.hasConnection(0, newCell._id));
    EXPECT_TRUE(approxCompare(180.0f + 45.0f, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(2, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_2_4__concatenation_0_1__branch_0_0__numAdditionalConnections_0)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(2).lastConstructedCellId(2))
                .pos({100.0f, 100.0f}),
        }), GenomeDescription().genes({         GeneDescription().separation(true).nodes({             NodeDescription(),             NodeDescription().referenceAngle(0),             NodeDescription().referenceAngle(45.0f).numAdditionalConnections(0),             NodeDescription(),         }),     }));
    data.addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 99.0f}).cellState(CellState_Constructing),
            CellDescription().id(2).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }), GenomeDescription().genes({         GeneDescription().separation(true).nodes({             NodeDescription(),             NodeDescription().referenceAngle(0),             NodeDescription().referenceAngle(45.0f).numAdditionalConnections(0),             NodeDescription(),         }),     }));
    data.addConnection(1, 2);
    data.addConnection(2, 0);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevPrevCell = actualData.getCellRef(1);
    auto prevCell = actualData.getCellRef(2);
    auto newCell = actualData.getOtherCellRef({0, 1, 2});

    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(prevPrevCell, prevCell));
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_EQ(1, prevPrevCell._connections.size());
    EXPECT_EQ(2, prevCell._connections.size());
    EXPECT_EQ(2, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());
    EXPECT_TRUE(approxCompare(180.0f + 45.0f, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
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
    data.addCreature(CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(2).lastConstructedCellId(2))
                .pos({100.0f, 100.0f}),
        }), GenomeDescription().genes({         GeneDescription()             .separation(true)             .nodes({                 NodeDescription(),                 NodeDescription().referenceAngle(0),                 NodeDescription().referenceAngle(NodeAngle).numAdditionalConnections(1),                 NodeDescription(),             })             .angleAlignment(angleAlignment),     }));
    data.addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 99.0f}).cellState(CellState_Constructing),
            CellDescription().id(2).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }), GenomeDescription().genes({         GeneDescription()             .separation(true)             .nodes({                 NodeDescription(),                 NodeDescription().referenceAngle(0),                 NodeDescription().referenceAngle(NodeAngle).numAdditionalConnections(1),                 NodeDescription(),             })             .angleAlignment(angleAlignment),     }));
    data.addConnection(1, 2);
    data.addConnection(2, 0);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevPrevCell = actualData.getCellRef(1);
    auto prevCell = actualData.getCellRef(2);
    auto newCell = actualData.getOtherCellRef({0, 1, 2});

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

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
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
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentConcatenation(1).currentNodeIndex(0).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCellRef({0, 1});

    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(prevCell, newCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_EQ(1, prevCell._connections.size());
    EXPECT_EQ(2, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());
    EXPECT_TRUE(approxCompare(180.0f + ConcatenationAngle, actualData.getConnection(newCell, hostCell)._angleFromPrevious));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
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
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentConcatenation(1).currentNodeIndex(0).lastConstructedCellId(4))
                .pos({100.0f, 100.0f}),
            CellDescription().id(1).pos({100.0f + addDistance, 98.0f}).cellState(CellState_Constructing),
            CellDescription().id(2).pos({100.0f + addDistance, 99.0f}).cellState(CellState_Constructing),
            CellDescription().id(3).pos({101.0f + addDistance, 99.0f}).cellState(CellState_Constructing),
            CellDescription().id(4).pos({101.0f + addDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(0, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(6, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(4);
    auto prevPrevPrevCell = actualData.getCellRef(2);
    auto newCell = actualData.getOtherCellRef({0, 1, 2, 3, 4});

    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_FALSE(newCell._isFrontAngleRefCell);
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

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(1, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_bothSidesPresent)
{
    auto data = Description().addCreature(
        CreatureDescription()
            .id(0)
            .cells({
                CellDescription()
                    .id(1)
                    .pos({10.0f, 10.0f})
                    .energy(getConstructorEnergy())
                    .cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).lastConstructedCellId(2)),
                CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing),
                CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.5f}).cellState(CellState_Constructing),
                CellDescription().id(4).pos({10.0f + getOffspringDistance(), 10.5f}).cellState(CellState_Constructing),
            }),
    GenomeDescription().genes({
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

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, hostCreature._cells.size());

    auto actualHostCell = actualData.getCellRef(1);
    auto actualPrevConstructedCell = actualData.getCellRef(2);
    auto actualUpperConstructedCell = actualData.getCellRef(3);
    auto actualLowerConstructedCell = actualData.getCellRef(4);
    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3, 4});

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

    auto data = Description().addCreature(CreatureDescription()
             .id(0)
             .cells({
                 CellDescription()
                     .id(1)
                     .pos({10.0f, 10.0f})
                     .energy(getConstructorEnergy())
                     .cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).lastConstructedCellId(2)),
                 CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing),
                 CellDescription().id(3).pos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f).cellState(CellState_Constructing),
                 CellDescription().id(4).pos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).cellState(CellState_Constructing),
             }),
             GenomeDescription().genes({
                 GeneDescription().separation(false).nodes(
                     {NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(2)}),
             }));
    data.addConnection(1, 2);
    auto cell3_refPos = data.getCellRef(2)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = data.getCellRef(3)._pos + Math::rotateClockwise({-1.0f, 0.0f}, 60.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto actualHostCell = actualData.getCellRef(1);
    auto actualPrevConstructedCell = actualData.getCellRef(2);
    auto actualPrevPrevConstructedCell = actualData.getCellRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getCellRef(4);
    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3, 4});

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

    EXPECT_TRUE(approxCompare(120.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(240.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_2__threeCellsWithSmallAngles__variant_2)
{
    auto data = Description().addCreature(CreatureDescription()
             .id(0)
             .cells({
                 CellDescription()
                     .id(1)
                     .pos({458.20f, 239.23f})
                     .energy(getConstructorEnergy())
                     .cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).lastConstructedCellId(2)),
                 CellDescription().id(2).pos({456.40f, 238.88f}).cellState(CellState_Constructing),
                 CellDescription().id(3).pos({455.96f, 239.75f}).cellState(CellState_Constructing),
                 CellDescription().id(4).pos({456.07f, 240.77f}).cellState(CellState_Constructing),
             }),
             GenomeDescription().genes({
                 GeneDescription().separation(false).nodes(
                     {NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(2)}),
             }));
    auto const& cell1 = data.getCellRef(1);
    auto const& cell2 = data.getCellRef(2);
    auto const& cell3 = data.getCellRef(3);

    data.addConnection(1, 2);
    auto cell3_refPos = cell2._pos + Math::rotateClockwise(cell1._pos - cell2._pos, 120.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = cell3._pos + Math::rotateClockwise(cell2._pos - cell3._pos, 120.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto actualHostCell = actualData.getCellRef(1);
    auto actualPrevConstructedCell = actualData.getCellRef(2);
    auto actualPrevPrevConstructedCell = actualData.getCellRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getCellRef(4);
    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3, 4});

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

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2)),
            CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing),
            CellDescription().id(3).pos(RealVector2D(10.0f + getOffspringDistance() + 0.2f, 10.0f) + offset * 0.1f).cellState(CellState_Constructing),
            CellDescription().id(4).pos(RealVector2D(10.0f + getOffspringDistance(), 10.0f) + offset * 0.2f).cellState(CellState_Constructing),
        }),
    genome);
    data.addConnection(1, 2);
    auto cell3_refPos = data.getCellRef(2)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(2, 3, cell3_refPos);
    auto cell4_refPos = data.getCellRef(3)._pos + Math::rotateClockwise({-0.5f, 0.0f}, 60.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, creature._cells.size());

    auto actualHostCell = actualData.getCellRef(1);
    auto actualPrevConstructedCell = actualData.getCellRef(2);
    auto origPrevPrevConstructedCell = data.getCellRef(3);
    auto actualPrevPrevConstructedCell = actualData.getCellRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getCellRef(4);
    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3, 4});

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

    EXPECT_TRUE(approxCompare(120.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualConstructedCell)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(240.0f, actualData.getConnection(actualPrevPrevPrevConstructedCell, actualPrevPrevConstructedCell)._angleFromPrevious));
}

TEST_F(ConstructorTests, creature_4__node_3_4__concatenation_0_1__branch_0_1__numAdditionalConnections_1__90degAlignment)
{
    auto genome = GenomeDescription().genes({
        GeneDescription()
            .separation(false)
            .angleAlignment(ConstructorAngleAlignment_90)
            .nodes({NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription().numAdditionalConnections(1)}),
    });

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2)),
            CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing),
            CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).cellState(CellState_Constructing),
            CellDescription().id(4).pos({10.0f + getOffspringDistance() - 1.0f, 9.0f - 0.2f}).cellState(CellState_Constructing),
        }),
    genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = data.getCellRef(3)._pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, creature._cells.size());

    auto actualHostCell = actualData.getCellRef(1);
    auto actualPrevConstructedCell = actualData.getCellRef(2);
    auto actualPrevPrevConstructedCell = actualData.getCellRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getCellRef(4);
    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3, 4});

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

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().currentNodeIndex(2).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2)),
            CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing),
            CellDescription().id(3).pos({10.0f + getOffspringDistance(), 9.0f}).cellState(CellState_Constructing),
        }),
    genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, creature._cells.size());

    auto actualHostCell = actualData.getCellRef(1);
    auto actualPrevConstructedCell = actualData.getCellRef(2);
    auto actualPrevPrevConstructedCell = actualData.getCellRef(3);
    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3});

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

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().currentNodeIndex(3).autoTriggerInterval(1).geneIndex(0).lastConstructedCellId(2)),
            CellDescription().id(2).pos({10.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing),
            CellDescription().id(3).pos({10.0f + getOffspringDistance(), 10.0f - 0.5f}).cellState(CellState_Constructing),
            CellDescription().id(4).pos({10.0f + getOffspringDistance(), 10.0f - 1.0f}).cellState(CellState_Constructing),
        }),
    genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    auto cell4_refPos = data.getCellRef(3)._pos + RealVector2D(-1.0f, 0.0f);
    data.addConnection(3, 4, cell4_refPos);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(5, creature._cells.size());

    auto actualHostCell = actualData.getCellRef(1);
    auto actualPrevConstructedCell = actualData.getCellRef(2);
    auto actualPrevPrevConstructedCell = actualData.getCellRef(3);
    auto actualPrevPrevPrevConstructedCell = actualData.getCellRef(4);
    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3, 4});

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
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(0).energy(getConstructorEnergy()).cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(0)).pos({100.0f, 100.0f}),
        }),
    genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(1, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCellRef({0});

    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, creature_1__node_0_1__concatenation_1_inf__branch_0_0)
{
    Description data;
    data.addCreature(CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(0).currentNodeIndex(0).currentConcatenation(1).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
        }), GenomeDescription().genes({         GeneDescription().separation(true).numConcatenations(GeneDescription::NumConcatenations_Infinite).nodes({NodeDescription()}),     }));
    data.addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(1).pos({101.0f + _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Ready),
        }), GenomeDescription().genes({         GeneDescription().separation(true).numConcatenations(GeneDescription::NumConcatenations_Infinite).nodes({NodeDescription()}),     }));
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto newCreature = actualData.getOtherCreatureRef(0);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCellRef({0, 1});

    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(newCell, hostCell));
    EXPECT_TRUE(actualData.hasConnection(newCell, prevCell));
    EXPECT_EQ(1, prevCell._connections.size());
    EXPECT_EQ(2, newCell._connections.size());
    EXPECT_EQ(1, hostCell._connections.size());

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellType);
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

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(0).pos({100.0f, 99.0f}),
            CellDescription()
                .id(1)
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().constructionAngle(ConstructionAngle).geneIndex(0).currentNodeIndex(0).autoTriggerInterval(100))
                .pos({100.0f, 100.0f}),
            CellDescription().id(2).pos({100.1f, 101.0f}),
        }),
    genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    auto hostCreature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, hostCreature._cells.size());

    auto const& hostCell = actualData.getCellRef(1);
    auto const& newCell = actualData.getOtherCellRef({0, 1, 2});

    auto angleSpan_cell2_cell0 = hostCell.getAngleSpan(2, 0);
    auto angleSpan_lastCell_and_cell0 = hostCell.getAngleSpan(newCell._id, 0);
    EXPECT_TRUE(approxCompare(Math::getNormalizedAngle(angleSpan_lastCell_and_cell0 + ConstructionAngle, 0.0f), angleSpan_cell2_cell0 / 2));
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1__frontAngle_leftSide)
{
    auto genome = GenomeDescription()
                      .genes({
                          GeneDescription().separation(false).nodes({NodeDescription()}),
                      });

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(1).pos({10.0f, 10.0f}),
            CellDescription()
                .id(2)
                .pos({9.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().currentNodeIndex(0).autoTriggerInterval(1).geneIndex(0)),
            CellDescription().id(3).pos({9.0f, 11.0f}),
        }),
    genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, creature._cells.size());

    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3});

    EXPECT_EQ(CellState_Ready, actualConstructedCell._cellState);
}

TEST_F(ConstructorTests, creature_3__node_0_1__concatenation_0_1__branch_0_1__frontAngle_rightSide)
{
    auto genome = GenomeDescription()
                      .genes({
                          GeneDescription().separation(false).nodes({NodeDescription()}),
                      });

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(1).pos({8.0f, 10.0f}),
            CellDescription()
                .id(2)
                .pos({9.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().currentNodeIndex(0).autoTriggerInterval(1).geneIndex(0)),
            CellDescription().id(3).pos({9.0f, 11.0f}),
        }),
    genome);
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(4, creature._cells.size());

    auto actualConstructedCell = actualData.getOtherCellRef({1, 2, 3});

    EXPECT_EQ(CellState_Ready, actualConstructedCell._cellState);
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

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(0).pos({100.0f, 99.0f}),
            CellDescription()
                .id(1)
                .energy(getConstructorEnergy() * n)
                .cellType(ConstructorDescription().constructionAngle(ConstructionAngle).geneIndex(0).currentNodeIndex(0).autoTriggerInterval(250))
                .pos({100.0f, 100.0f}),
            CellDescription().id(2).pos({100.1f, 101.0f}),
        }),
    genome);
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);

    // Construct offspring and record ids of constructed cells
    std::vector<uint64_t> createdCellIds;
    {
        for (int i = 0; i < n; ++i) {
            _simulationFacade->calcTimesteps(250);
            auto actualData = _simulationFacade->getSimulationData();

            ASSERT_EQ(0, actualData._cells.size());
            ASSERT_EQ(1, actualData._creatures.size());
            EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

            auto hostCreature = actualData.getCreatureRef(0);
            ASSERT_EQ(3 + i + 1, hostCreature._cells.size());

            auto hostCell = actualData.getCellRef(1);

            std::set<uint64_t> knownCellIds(createdCellIds.begin(), createdCellIds.end());
            knownCellIds.insert(0);
            knownCellIds.insert(1);
            knownCellIds.insert(2);
            auto newCell = actualData.getOtherCellRef(knownCellIds);
            createdCellIds.emplace_back(newCell._id);

            if (i < n - 1) {
                EXPECT_EQ(CellState_Constructing, newCell._cellState);
            } else {
                EXPECT_EQ(CellState_Ready, newCell._cellState);
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
            auto const& cell = actualData.getCellRef(createdCellIds.at(i));
            auto prevCellId = createdCellIds.at(i - 1);
            auto nextCellId = createdCellIds.at(i + 1);
            auto angle = cell.getAngleSpan(prevCellId, nextCellId);
            angle = Math::getNormalizedAngle(angle - 180.0f, -180.0f);
            EXPECT_EQ(shapeResult.angle, angle);
            int numPrevConnections = 0;
            for (auto const& connection : cell._connections) {
                if (connection._cellId < cell._id) {
                    ++numPrevConnections;
                }
            }
            EXPECT_EQ(shapeResult.numAdditionalConnections, numPrevConnections - 1);
        }
    }

    // Check angles for first node
    {
        auto const& hostCell = actualData.getCellRef(1);
        auto angleSpan_cell2_cell0 = hostCell.getAngleSpan(2, 0);
        auto angleSpan_lastCell_and_cell0 = hostCell._connections.at(0)._angleFromPrevious;
        EXPECT_TRUE(approxCompare(angleSpan_lastCell_and_cell0 + ConstructionAngle, angleSpan_cell2_cell0 / 2));
    }

    // Check angles for last node
    {
        auto const& cell = actualData.getCellRef(createdCellIds.back());
        auto prevCellId = createdCellIds.at(n - 2);
        auto nextCellId = 1;    // = id of hostCell
        auto angle = cell.getAngleSpan(prevCellId, nextCellId);
        angle = Math::getNormalizedAngle(angle - 180.0f, -180.0f);
        EXPECT_EQ(LastAngle, angle);
    }
}

TEST_F(ConstructorTests, avoidDeadlockByLockingNearCells)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({}),
        GeneDescription()
            .separation(false)
            .shape(ConstructorShape_Hexagon).nodes({NodeDescription(), NodeDescription(), NodeDescription(), NodeDescription()}),
    });

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription().id(1).pos({10.0f, 10.0f}),
            CellDescription()
                .id(2)
                .pos({10.0f, 9.0f})
                .energy(getConstructorEnergy())
                .cellType(ConstructorDescription().geneIndex(1).currentNodeIndex(2).lastConstructedCellId(6)),
            CellDescription().id(3).pos({11.0f, 9.0f}),
            CellDescription()
                .id(4)
                .pos({11.0f, 10.0f})
                .energy(getConstructorEnergy())
                .cellType(
                    ConstructorDescription().geneIndex(1).currentNodeIndex(2).lastConstructedCellId(8)),

            CellDescription().id(5).pos({10.0f, 9.0f - getOffspringDistance() - 1.0f}).cellState(CellState_Constructing).nodeIndex(0).geneIndex(1),
            CellDescription().id(6).pos({10.0f, 9.0f - getOffspringDistance()}).cellState(CellState_Constructing).nodeIndex(1).geneIndex(1),
            CellDescription().id(7).pos({11.0f + getOffspringDistance() + 1.0f, 10.0f}).cellState(CellState_Constructing).nodeIndex(0).geneIndex(1),
            CellDescription().id(8).pos({11.0f + getOffspringDistance(), 10.0f}).cellState(CellState_Constructing).nodeIndex(1).geneIndex(1),
        }),
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
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(10, creature._cells.size());
}

TEST_F(ConstructorTests, avoidConnectionsBetweenDifferentConstructions)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({}),
        GeneDescription().separation(false).shape(ConstructorShape_Hexagon).nodes({NodeDescription(), NodeDescription(), NodeDescription()}),
        GeneDescription().separation(false).shape(ConstructorShape_Hexagon).nodes({NodeDescription(), NodeDescription(), NodeDescription()}),
    });

    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .energy(getConstructorEnergy())
                .nodeIndex(0)
                .cellType(ConstructorDescription().geneIndex(1).currentNodeIndex(2).lastConstructedCellId(4)),
            CellDescription()
                .id(2)
                .pos({11.0f, 10.0f})
                .energy(getConstructorEnergy())
                .nodeIndex(1)
                .cellType(ConstructorDescription().geneIndex(2).currentNodeIndex(2).lastConstructedCellId(6)),

            CellDescription()
                .id(3)
                .pos({10.0f, 10.0f - getOffspringDistance() - 1.0f})
                .cellState(CellState_Constructing)
                .nodeIndex(0)
                .geneIndex(1)
                .parentNodeIndex(0),
            CellDescription()
                .id(4)
                .pos({10.0f, 10.0f - getOffspringDistance()})
                .cellState(CellState_Constructing)
                .nodeIndex(1)
                .geneIndex(1)
                .parentNodeIndex(0),
            CellDescription()
                .id(5)
                .pos({11.0f, 10.0f - getOffspringDistance() - 1.0f})
                .cellState(CellState_Constructing)
                .nodeIndex(0)
                .geneIndex(2)
                .parentNodeIndex(1),
            CellDescription().id(6).pos({11.0f, 10.0f - getOffspringDistance()}).cellState(CellState_Constructing).nodeIndex(1).geneIndex(2).parentNodeIndex(1),
        }),
    genome);
    data.addConnection(1, 2);

    data.addConnection(3, 4);
    data.addConnection(4, 1);

    data.addConnection(5, 6);
    data.addConnection(6, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(8, creature._cells.size());

    auto constructedCells = actualData.getOtherCells({1, 2, 3, 4, 5, 6});
    CellDescription cell1, cell2;
    for (auto const& cell : constructedCells) {
        if (cell._parentNodeIndex == 0) {
            cell1 = cell;
        } else {
            cell2 = cell;
        }
    }
    EXPECT_EQ(3, cell1._connections.size());
    EXPECT_TRUE(actualData.hasConnection(cell1._id, 1));
    EXPECT_TRUE(actualData.hasConnection(cell1._id, 3));
    EXPECT_TRUE(actualData.hasConnection(cell1._id, 4));

    EXPECT_EQ(3, cell2._connections.size());
    EXPECT_TRUE(actualData.hasConnection(cell2._id, 2));
    EXPECT_TRUE(actualData.hasConnection(cell2._id, 5));
    EXPECT_TRUE(actualData.hasConnection(cell2._id, 6));
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
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .pos({10.0f, 10.0f})
                .energy(constructorEnergy)
                .cellType(ConstructorDescription()
                              .provideEnergy(provideEnergy)
                              .geneIndex(0)
                              .currentNodeIndex(1)
                              .autoTriggerInterval(1)
                              .lastConstructedCellId(1)),
            CellDescription().id(1).pos({10.0f + getOffspringDistance(), 10.0f}),
        }),
    genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, creature._cells.size());

    auto actualConstructedCell = actualData.getOtherCellRef({0, 1});

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        EXPECT_TRUE(approxCompare(normalCellEnergy, actualData.getCellRef(0)._energy));
    } else {
        if (provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No) {
            EXPECT_TRUE(approxCompare(normalCellEnergy * (2 * 3 * 2 + 1), actualConstructedCell._energy));
        } else {
            EXPECT_TRUE(approxCompare(normalCellEnergy, actualConstructedCell._energy));
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
    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .pos({10.0f, 10.0f})
                .energy(constructorEnergy)
                .cellType(ConstructorDescription()
                              .provideEnergy(provideEnergy)
                              .geneIndex(0)
                              .currentNodeIndex(1)
                              .autoTriggerInterval(1)
                              .lastConstructedCellId(1)),
            CellDescription().id(1).pos({10.0f + getOffspringDistance(), 10.0f}),
        }),
    genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(2, creature._cells.size());
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


    auto data = Description().addCreature(
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(0)
                .pos({10.0f, 10.0f})
                .energy(constructorEnergy)
                .cellType(ConstructorDescription()
                              .provideEnergy(provideEnergy)
                              .geneIndex(0)
                              .currentNodeIndex(1)
                              .autoTriggerInterval(1)
                              .lastConstructedCellId(1)),
            CellDescription().id(1).pos({10.0f + getOffspringDistance(), 10.0f}),
        }),
    genome);
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto creature = actualData.getCreatureRef(0);
    ASSERT_EQ(3, creature._cells.size());

    auto actualConstructedCell = actualData.getOtherCellRef({0, 1});

    if (provideEnergy == ProvideEnergy_FreeGeneration) {
        EXPECT_TRUE(approxCompare(normalCellEnergy, actualData.getCellRef(0)._energy));
    } else {
        if (provideEnergy == ProvideEnergy_CellAndGene && separation == Separation::No) {
            EXPECT_TRUE(approxCompare(normalCellEnergy * (2 + 1), actualConstructedCell._energy));
        } else {
            EXPECT_TRUE(approxCompare(normalCellEnergy, actualConstructedCell._energy));
        }
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
    auto largeCreature = CreatureDescription();
    for (int i = 0; i < 50; ++i) {
        largeCreature._cells.emplace_back(
            CellDescription().id(i).pos({toFloat(i), 0.0f}).cellType(ConstructorDescription().geneIndex(0).autoTriggerInterval(30)));
    }
    Description largeCreatureData;
    largeCreatureData.addCreature(largeCreature, genome);
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
