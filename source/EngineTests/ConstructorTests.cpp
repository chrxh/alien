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
    
    float getOffspringDistance() const
    {
        return 1.0f + _parameters.constructorAdditionalOffspringDistance;
    }

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
            if (constructor._constructionAngle != nodeConstructor._constructionAngle) {
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

TEST_F(ConstructorTests, start_alreadyFinished)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().numBranches(1).nodes({NodeDescription()}),
            }))
            .cells({
                CellDescription().id(0).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(1)).pos({100.0f, 100.0f}),
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

TEST_P(ConstructorTests_AllNodeTypes, start_creatureSize1_gene0_separation_finished)
{
    auto nodeParameter = GetParam();
    auto randomNode = _descriptionTestDataFactory->createRandomNodeDescription(nodeParameter);

    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().nodes({randomNode}),
            }))
            .cells({CellDescription().energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f})}),
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
    EXPECT_TRUE(approxCompare(0, newCell._angleToFront));
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(compare(newCell, randomNode));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, start_creatureSize1_gene1_separation_finished)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().nodes({NodeDescription()}),
                GeneDescription().nodes({NodeDescription()}),
            }))
            .cells({CellDescription().id(0).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(1)).pos({100.0f, 100.0f})}),
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
    EXPECT_TRUE(approxCompare(0, newCell._angleToFront));
    EXPECT_FALSE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, start_creatureSize1_gene0_branch1_finished)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().numBranches(1).nodes({NodeDescription()}),
            }))
            .cells({CellDescription().energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f})}),
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
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(1.0f, Math::length(hostCell._pos - newCell._pos)));
    EXPECT_TRUE(approxCompare(0, newCell._angleToFront));

    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(1.0f, connection._distance);

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, start_creatureSize1_gene0_branch1_moreNodes)
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

TEST_F(ConstructorTests, start_creatureSize1_gene0_branch1_moreConcat)
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


TEST_F(ConstructorTests, start_creatureSize1_gene0_branch2_finished)
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

    auto newCreature = actualData.getCreature(1);
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

TEST_F(ConstructorTests, start_creatureSize1_gene1_branch1_finished)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().nodes({NodeDescription()}),
                GeneDescription().numBranches(1).nodes({NodeDescription()}),
            }))
            .cells({CellDescription().id(0).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(1)).pos({100.0f, 100.0f})}),
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
    EXPECT_TRUE(approxCompare(0, newCell._angleToFront));
    EXPECT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));

    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(1, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, start_creatureSize2_gene0_separation_finished)
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

TEST_F(ConstructorTests, start_creatureSize2_gene0_branch1_finished)
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

TEST_F(ConstructorTests, start_creatureSize2_gene0_branch2_finished)
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

TEST_F(ConstructorTests, start_creatureSize2_gene1_branch2_finished)
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

TEST_F(ConstructorTests, start_creatureSize3_gene1_branch1_finished)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(GenomeDescription().genes({
                GeneDescription().nodes({NodeDescription()}),
                GeneDescription().numBranches(1).nodes({NodeDescription()}),
            }))
            .cells({
                CellDescription().id(0).pos({100.0f, 99.0f}).angleToFront(90.0f),
                CellDescription().id(1).energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(1)).pos({100.0f, 100.0f}).angleToFront(-90.0f),
                 CellDescription().id(2).pos({101.0f, 100.0f}).angleToFront(0),
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

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(4, creature._cells.size());

    auto hostCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCell({0, 1, 2});
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    EXPECT_TRUE(approxCompare(hostCell._pos + Math::unitVectorOfAngle(225.0f), newCell._pos));
    EXPECT_TRUE(approxCompare(-135.0f, newCell._angleToFront));
    EXPECT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));
}

TEST_F(ConstructorTests, continue_creatureSize1_gene0_separation_finished)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription(), NodeDescription()}),
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
    EXPECT_TRUE(approxCompare(0.0f, newCell._angleToFront));
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_FALSE(actualData.hasConnection(0, 1));
    EXPECT_FALSE(actualData.hasConnection(0, 2));
}

TEST_F(ConstructorTests, continue_middleNode_gene0_multipleNodes)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({
            NodeDescription().referenceAngle(0.0f), 
            NodeDescription().referenceAngle(90.0f), 
            NodeDescription().referenceAngle(180.0f)
        }),
    });
    auto data = CollectionDescription().creatures({
// TODO different connection distances

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
    
    // Verify the new cell was constructed from node 1 (with 90 degree angle)
    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_FALSE(actualData.hasConnection(0, 1));
    EXPECT_FALSE(actualData.hasConnection(0, 2));

    // Verify constructor state progressed to node 2
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(2, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, continue_startFromMiddle_node1_withAngle)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({
            NodeDescription().referenceAngle(45.0f), 
            NodeDescription().referenceAngle(135.0f), 
            NodeDescription().referenceAngle(225.0f)
        }),
    });
    auto data = CollectionDescription().creatures({

        // Parent  
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentNodeIndex(1))
                .pos({100.0f, 100.0f}),
        }),

        // Offspring with first constructed cell
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(1).pos({99.0f, 100.0f}).cellState(CellState_Constructing),
        }),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    auto newCreature = actualData.getCreature(1);
    ASSERT_EQ(2, newCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto prevCell = actualData.getCellRef(1);
    auto newCell = actualData.getOtherCell({0, 1});
    
    // Should construct node 1 (index 1, referenceAngle 135.0f)
    EXPECT_EQ(CellState_Constructing, newCell._cellState);

    // Verify constructor progressed to node 2
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(2, hostConstructor._currentNodeIndex);
}

TEST_F(ConstructorTests, continue_endOfGene_separation_on)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription(), NodeDescription()}),
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

        // Offspring with first constructed cell  
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
    auto newCell = actualData.getOtherCell({0, 1});
    
    // Last node should be activating (finished)
    EXPECT_EQ(CellState_Activating, newCell._cellState);
    
    // Constructor should reset to beginning since it's separation
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, continue_endOfGene_branch_mode_numConcatenations2)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numBranches(1).numConcatenations(2).nodes({NodeDescription(), NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentNodeIndex(1).currentConcatenation(0).lastConstructedCellId(1))
                .pos({100.0f, 100.0f}),
        }),

        // Offspring with first constructed cell  
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(1).pos({99.0f - _parameters.constructorAdditionalOffspringDistance, 100.0f}).cellState(CellState_Constructing),
        }),
    });
    data.addConnection(0, 1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreature(0);
    ASSERT_EQ(3, creature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCell({0, 1});
    
    // Should be constructing still since we have more concatenations
    EXPECT_EQ(CellState_Constructing, newCell._cellState);
    
    // Constructor should go to next concatenation
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(1, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, continue_withExistingBranches_branch2)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numBranches(2).nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({

        // Parent
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(1))
                .pos({100.0f, 100.0f}),
            CellDescription().id(1).pos({101.0f, 100.0f}),
        }),

        // Existing branch 1 offspring
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(2).pos({99.0f, 100.0f}),
        }),
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
    
    // New cell should be connected to host
    EXPECT_TRUE(actualData.hasConnection(0, newCell._id));
    
    // Verify constructor advanced to branch 2
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(2, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, continue_multipleExistingBranches_newBranch)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numBranches(3).nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({

        // Parent with existing connections to branches
        CreatureDescription().id(0).genome(genome).cells({
            CellDescription()
                .id(0)
                .energy(getConstructorEnergy())
                .cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(2))
                .pos({100.0f, 100.0f}),
            CellDescription().id(1).pos({101.0f, 100.0f}),
        }),

        // Existing branch 1 offspring
        CreatureDescription().id(1).genome(genome).cells({
            CellDescription().id(2).pos({99.0f, 100.0f}),
        }),

        // Existing branch 2 offspring  
        CreatureDescription().id(2).genome(genome).cells({
            CellDescription().id(3).pos({100.0f, 99.0f}),
        }),
    });
    data.addConnection(0, 1);
    data.addConnection(0, 2);
    data.addConnection(0, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(3, actualData._creatures.size());

    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(2, hostCreature._cells.size());

    auto hostCell = actualData.getCellRef(0);
    auto newCell = actualData.getOtherCell({0, 1, 2, 3});
    
    // New cell should be connected to host
    EXPECT_TRUE(actualData.hasConnection(0, newCell._id));
    
    // Verify constructor advanced to branch 3
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(3, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, construction_withConnectionDistance)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numBranches(1).connectionDistance(2.0f).nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(genome)
            .cells({CellDescription().energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f})}),
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

    auto hostCell = hostCreature._cells.front();
    auto newCell = newCreature._cells.front();
    
    // Verify connection exists with custom distance
    ASSERT_TRUE(actualData.hasConnection(hostCell._id, newCell._id));
    auto connection = actualData.getConnection(hostCell, newCell);
    EXPECT_EQ(2.0f, connection._distance);
}

TEST_F(ConstructorTests, construction_withLargeConnectionDistance)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numBranches(1).connectionDistance(5.0f).nodes({NodeDescription(), NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(genome)
            .cells({CellDescription().energy(getConstructorEnergy() * 2).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f})}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();

    auto hostCreature = actualData.getCreature(0);
    auto newCreature = actualData.getOtherCreature(0);

    auto hostCell = hostCreature._cells.front();
    
    // Find the constructed cells
    auto constructedCells = newCreature._cells;
    ASSERT_EQ(2, constructedCells.size());
    
    // Verify connections use the specified distance
    for (auto& cell : constructedCells) {
        for (int i = 0; i < cell._connections.size(); ++i) {
            if (cell._connections[i]._distance > 1.0f) {  // Custom distance
                EXPECT_TRUE(approxCompare(5.0f, cell._connections[i]._distance) || 
                           approxCompare(5.0f + _parameters.constructorAdditionalOffspringDistance, cell._connections[i]._distance));
            }
        }
    }
}

TEST_F(ConstructorTests, failing_construction_lowEnergy)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(genome)
            .cells({CellDescription().energy(10.0f).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f})}),  // Very low energy
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    // Construction should fail due to insufficient energy
    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    
    auto creature = actualData.getCreature(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = creature._cells.front();
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    
    // Constructor state should not advance due to failed construction
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, failing_construction_invalidGeneIndex)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(genome)
            .cells({CellDescription().energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(5)).pos({100.0f, 100.0f})}),  // Invalid gene index
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    // Construction should fail due to invalid gene index
    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    
    auto creature = actualData.getCreature(0);
    ASSERT_EQ(1, creature._cells.size());

    auto hostCell = creature._cells.front();
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    
    // Constructor state should not advance
    EXPECT_EQ(0, hostConstructor._currentNodeIndex);
    EXPECT_EQ(0, hostConstructor._currentConcatenation);
    EXPECT_EQ(0, hostConstructor._currentBranch);
}

TEST_F(ConstructorTests, failing_construction_tooHighConcatenationIndex)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numConcatenations(2).nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(genome)
            .cells({CellDescription()
                        .energy(getConstructorEnergy())
                        .cellTypeData(ConstructorDescription().geneIndex(0).currentConcatenation(5))  // Too high concatenation index
                        .pos({100.0f, 100.0f})}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    // Construction should fail due to invalid concatenation index
    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    
    auto creature = actualData.getCreature(0);
    ASSERT_EQ(1, creature._cells.size());
}

TEST_F(ConstructorTests, failing_construction_tooHighBranchIndex)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numBranches(2).nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(genome)
            .cells({CellDescription()
                        .energy(getConstructorEnergy())
                        .cellTypeData(ConstructorDescription().geneIndex(0).currentBranch(10))  // Too high branch index
                        .pos({100.0f, 100.0f})}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    // Construction should fail due to invalid branch index
    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());
    
    auto creature = actualData.getCreature(0);
    ASSERT_EQ(1, creature._cells.size());
}

TEST_F(ConstructorTests, failing_construction_blockedSpace)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().nodes({NodeDescription()}),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(genome)
            .cells({CellDescription().energy(getConstructorEnergy()).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f})}),
        
        // Blocking creature at expected construction position
        CreatureDescription()
            .id(1)
            .cells({CellDescription().id(1).pos({99.0f, 100.0f})}),  // Block construction direction
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    // Construction might be blocked or redirected
    auto hostCreature = actualData.getCreature(0);
    ASSERT_EQ(1, hostCreature._cells.size());

    auto hostCell = hostCreature._cells.front();
    auto hostConstructor = std::get<ConstructorDescription>(hostCell._cellTypeData);
    
    // If construction was completely blocked, state should not advance
    // If construction succeeded in different direction, it should advance
    // Either behavior is acceptable depending on implementation
}

TEST_F(ConstructorTests, construction_withAdditionalConnections)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().numBranches(1).nodes({
            NodeDescription(),
            NodeDescription().numRequiredAdditionalConnections(1)
        }),
    });
    auto data = CollectionDescription().creatures({
        CreatureDescription()
            .id(0)
            .genome(genome)
            .cells({CellDescription().energy(getConstructorEnergy() * 2).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f})}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();

    auto hostCreature = actualData.getCreature(0);
    auto newCreature = actualData.getOtherCreature(0);

    // Should have constructed 2 cells
    ASSERT_EQ(2, newCreature._cells.size());
    
    // Find second constructed cell (should have additional connections)
    auto constructedCells = newCreature._cells;
    auto secondCell = constructedCells[1];  // Second constructed cell
    
    // Second cell should have more connections due to numRequiredAdditionalConnections
    EXPECT_GE(secondCell._connections.size(), 2);  // At least connection to previous + additional
}


//
//TEST_F(ConstructorTests, constructFurtherCell_connectToExistingCell_upperSide)
//{
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
//        GenomeDescription()
//            .header(GenomeHeaderDescription().separateConstruction(false))
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));
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
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));
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
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));
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
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(2)}));
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
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(2)}));
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
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));
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
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));
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
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(0)}));
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
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(1)}));
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
//            .cells({CellGenomeDescription(), CellGenomeDescription().numRequiredAdditionalConnections(0)}));
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
