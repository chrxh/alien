#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "EngineTestData/DescriptionTestDataFactory.h"

#include "IntegrationTestFramework.h"


class DataTransferTests_New : public IntegrationTestFramework
{
public:
    DataTransferTests_New()
        : IntegrationTestFramework()
    {
        _descriptionTestDataFactory = &DescriptionTestDataFactory::get();
    }

protected:
    DescriptionTestDataFactory* _descriptionTestDataFactory;
};

TEST_F(DataTransferTests_New, singleParticle)
{
    CollectionDescription data;
    data.addParticle(_descriptionTestDataFactory->createRandomParticleDescription());

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

using CellParameter = DescriptionTestDataFactory::CellParameter;
class DataTransferTests_AllCellTypes_New
    : public DataTransferTests_New
    , public testing::WithParamInterface<CellParameter>
{
};

INSTANTIATE_TEST_SUITE_P(
    DataTransferTests_AllCellTypes_New,
    DataTransferTests_AllCellTypes_New,
    ::testing::Values(
        CellParameter{CellType_Structure},
        CellParameter{CellType_Free},
        CellParameter{CellType_Base},
        CellParameter{CellType_Depot},
        CellParameter{CellType_Constructor},
        CellParameter{CellType_Sensor},
        CellParameter{CellType_Oscillator},
        CellParameter{CellType_Attacker},
        CellParameter{CellType_Injector},
        CellParameter{CellType_Muscle, MuscleMode_AutoBending},
        CellParameter{CellType_Muscle, MuscleMode_ManualBending},
        CellParameter{CellType_Muscle, MuscleMode_AngleBending},
        CellParameter{CellType_Muscle, MuscleMode_AutoCrawling},
        CellParameter{CellType_Muscle, MuscleMode_ManualCrawling},
        CellParameter{CellType_Muscle, MuscleMode_DirectMovement},
        CellParameter{CellType_Defender},
        CellParameter{CellType_Reconnector},
        CellParameter{CellType_Detonator}));

TEST_P(DataTransferTests_AllCellTypes_New, singleCellWithoutCreature)
{
    auto cellParameter = GetParam();

    CollectionDescription data;
    data.addCell(_descriptionTestDataFactory->createRandomCellDescription(cellParameter));

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

using NodeParameter = DescriptionTestDataFactory::NodeParameter;
class DataTransferTests_AllNodeTypes_New
    : public DataTransferTests_New
    , public testing::WithParamInterface<NodeParameter>
{
};

INSTANTIATE_TEST_SUITE_P(
    DataTransferTests_AllNodeTypes_New,
    DataTransferTests_AllNodeTypes_New,
    ::testing::Values(
        NodeParameter{CellTypeGenome_Base},
        NodeParameter{CellTypeGenome_Depot},
        NodeParameter{CellTypeGenome_Constructor},
        NodeParameter{CellTypeGenome_Sensor},
        NodeParameter{CellTypeGenome_Oscillator},
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

TEST_P(DataTransferTests_AllNodeTypes_New, singleCellWithCreature_oneGene_oneNode)
{
    auto nodeParameter = GetParam();

    auto data = CollectionDescription().addCreature(_descriptionTestDataFactory->createRandomCreatureDescription(nodeParameter), {CellDescription()});

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests_New, multipleCells_genome_multipleGenes_multiple_Nodes)
{
    auto hexagon = DescriptionEditService::get().createHex(DescriptionEditService::CreateHexParameters().center({100.0f, 100.0f}));
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().genes(
            {GeneDescription().nodes({NodeDescription(), NodeDescription()}),
             GeneDescription().nodes({NodeDescription(), NodeDescription(), NodeDescription()})}),
        hexagon._cells);

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests_New, createCellIds_differentIdsOnDescription)
{
    auto data = CollectionDescription().cells({CellDescription().id(0), CellDescription().id(1)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& cell : actualData._cells) {
        ids.insert(cell._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createCellIds_sameIdsOnDescription)
{
    auto data1 = CollectionDescription().cells({CellDescription().id(0), CellDescription().id(0)});
    _simulationFacade->addAndSelectSimulationData(std::move(data1));
    auto data2 = CollectionDescription().cells({CellDescription().id(0), CellDescription().id(0)});
    _simulationFacade->addAndSelectSimulationData(std::move(data2));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& cell : actualData._cells) {
        ids.insert(cell._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createParticleIds_differentIdsOnDescription)
{
    auto data = CollectionDescription().particles({ParticleDescription().id(0), ParticleDescription().id(1)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& cell : actualData._particles) {
        ids.insert(cell._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createParticleIds_sameIdsOnDescription)
{
    auto data1 = CollectionDescription().particles({ParticleDescription().id(0), ParticleDescription().id(0)});
    _simulationFacade->addAndSelectSimulationData(std::move(data1));

    auto data2 = CollectionDescription().particles({ParticleDescription().id(0), ParticleDescription().id(0)});
    _simulationFacade->addAndSelectSimulationData(std::move(data2));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& cell : actualData._particles) {
        ids.insert(cell._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createGenomeIds)
{
    CollectionDescription data;
    data.addCreature(CreatureDescription(), {CellDescription()});
    data.addCreature(CreatureDescription(), {CellDescription()});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& genome : actualData._creatures) {
        ids.insert(genome._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, keepObjectIdsStable)
{
    auto data =
        CollectionDescription().cells({CellDescription().id(0), CellDescription().id(1)}).particles({ParticleDescription().id(2), ParticleDescription().id(3)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->setSimulationData(data);

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> expectedCellIds;
    for (auto const& cell : data._cells) {
        expectedCellIds.insert(cell._id);
    }
    std::unordered_set<uint64_t> actualCellIds;
    for (auto const& cell : actualData._cells) {
        actualCellIds.insert(cell._id);
    }
    EXPECT_EQ(expectedCellIds, actualCellIds);

    std::unordered_set<uint64_t> expectedParticleIds;
    for (auto const& particle : data._particles) {
        expectedParticleIds.insert(particle._id);
    }
    std::unordered_set<uint64_t> actualParticleIds;
    for (auto const& particle : actualData._particles) {
        actualParticleIds.insert(particle._id);
    }
    EXPECT_EQ(expectedParticleIds, actualParticleIds);
}

TEST_F(DataTransferTests_New, createCreatureIds_differentIds)
{
    auto data = CollectionDescription()
                    .cells({
                        CellDescription().creatureId(0),
                        CellDescription().creatureId(1),
                    })
                    .creatures({
                        CreatureDescription().id(0),
                        CreatureDescription().id(1),
                    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& creature : actualData._creatures) {
        ids.insert(creature._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests_New, createCreatureIds_sameIds)
{
    auto data = CollectionDescription()
                    .cells({
                        CellDescription().creatureId(0),
                        CellDescription().creatureId(0),
                    })
                    .creatures({
                        CreatureDescription().id(0),
                    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& creature : actualData._creatures) {
        ids.insert(creature._id);
    }

    EXPECT_EQ(2, ids.size());
}

TEST_F(DataTransferTests_New, changeGenome_successful)
{
    auto data = CollectionDescription().addCreature(CreatureDescription(), {CellDescription()});
    auto creatureId = data._cells.at(0)._creatureId;

    _simulationFacade->setSimulationData(data);

    auto newCreature = CreatureDescription().id(creatureId.value()).genes({GeneDescription().nodes({NodeDescription(), NodeDescription()})});
    auto result = _simulationFacade->changeCreature(newCreature);
    ASSERT_TRUE(result);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto cell = actualData._cells.front();
    EXPECT_EQ(creatureId, cell._creatureId);

    auto creature = actualData._creatures.front();
    EXPECT_EQ(cell._creatureId, creature._id);

    ASSERT_EQ(1, creature._genes.size());

    auto gene = creature._genes.front();
    EXPECT_EQ(2, gene._nodes.size());
}

TEST_F(DataTransferTests_New, changeGenome_failed)
{
    auto data = CollectionDescription().addCreature(CreatureDescription(), {CellDescription()});
    auto creatureId = data._cells.at(0)._creatureId;

    _simulationFacade->setSimulationData(data);

    auto newGenome = CreatureDescription().genes({GeneDescription().nodes({NodeDescription(), NodeDescription()})});
    auto result = _simulationFacade->changeCreature(newGenome);
    ASSERT_FALSE(result);
}

TEST_F(DataTransferTests_New, getInspectedSimulationData)
{
    auto data =
        CollectionDescription()
            .addCreature(CreatureDescription().genes({GeneDescription().nodes({NodeDescription(), NodeDescription()})}), {CellDescription(), CellDescription()})
            .addCreature(CreatureDescription(), {CellDescription()});
    auto cellId1 = data._cells.at(0)._id;
    auto cellId2 = data._cells.at(1)._id;

    _simulationFacade->setSimulationData(data);

    auto inspectedData = _simulationFacade->getInspectedSimulationData({cellId1, cellId2});
    ASSERT_EQ(2, inspectedData._cells.size());
    ASSERT_EQ(1, inspectedData._creatures.size());

    auto genome = inspectedData._creatures.front();
    auto cell1 = getCell(inspectedData, cellId1);
    EXPECT_EQ(genome._id, cell1._creatureId);

    auto cell2 = getCell(inspectedData, cellId2);
    EXPECT_EQ(genome._id, cell2._creatureId);

    ASSERT_EQ(1, genome._genes.size());

    auto gene = genome._genes.front();
    EXPECT_EQ(2, gene._nodes.size());
}
