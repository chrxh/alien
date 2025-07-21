#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "EngineTestData/DescriptionTestDataFactory.h"

#include "IntegrationTestFramework.h"


class DataTransferTests : public IntegrationTestFramework
{
public:
    DataTransferTests()
        : IntegrationTestFramework()
    {
        _descriptionTestDataFactory = &DescriptionTestDataFactory::get();
    }

protected:
    DescriptionTestDataFactory* _descriptionTestDataFactory;
};

TEST_F(DataTransferTests, singleParticle)
{
    CollectionDescription data;
    data.addParticle(_descriptionTestDataFactory->createRandomParticleDescription());

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

using CellParameter = DescriptionTestDataFactory::CellParameter;
class DataTransferTests_AllCellTypes
    : public DataTransferTests
    , public testing::WithParamInterface<CellParameter>
{
};

INSTANTIATE_TEST_SUITE_P(
    DataTransferTests_AllCellTypes,
    DataTransferTests_AllCellTypes,
    ::testing::Values(
        CellParameter{CellType_Structure},
        CellParameter{CellType_Free},
        CellParameter{CellType_Base},
        CellParameter{CellType_Depot},
        CellParameter{CellType_Constructor},
        CellParameter{CellType_Sensor},
        CellParameter{CellType_Generator},
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

TEST_P(DataTransferTests_AllCellTypes, cellsWithoutCreature)
{
    auto cellParameter = GetParam();

    CollectionDescription data;
    data.addCell(_descriptionTestDataFactory->createRandomCellDescription(cellParameter));
    data.addCell(_descriptionTestDataFactory->createRandomCellDescription(cellParameter));

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

using NodeParameter = DescriptionTestDataFactory::NodeParameter;
class DataTransferTests_AllNodeTypes
    : public DataTransferTests
    , public testing::WithParamInterface<NodeParameter>
{
};

INSTANTIATE_TEST_SUITE_P(
    DataTransferTests_AllNodeTypes,
    DataTransferTests_AllNodeTypes,
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

TEST_P(DataTransferTests_AllNodeTypes, cellsWithCreatures_oneNode)
{
    auto nodeParameter = GetParam();

    auto data = CollectionDescription().creatures({
        _descriptionTestDataFactory->createRandomCreatureDescription(nodeParameter).cells({CellDescription()}),
        _descriptionTestDataFactory->createRandomCreatureDescription(nodeParameter).cells({CellDescription()}),
    });

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, multipleCells_genome_multipleGenes_multipleNodes)
{
    auto hexagon = DescriptionEditService::get().createHex(DescriptionEditService::CreateHexParameters().center({100.0f, 100.0f}).cellType(BaseDescription()));
    CollectionDescription data;
    data.creatures({
        CreatureDescription()
            .genome(GenomeDescription().genes({
                GeneDescription().nodes({NodeDescription(), NodeDescription()}),
                GeneDescription().nodes({NodeDescription(), NodeDescription(), NodeDescription()}),
            }))
            .cells(hexagon._cells),
    });

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, createCellIds_differentIdsOnDescription)
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

TEST_F(DataTransferTests, createCellIds_sameIdsOnDescription)
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

TEST_F(DataTransferTests, createParticleIds_differentIdsOnDescription)
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

TEST_F(DataTransferTests, createParticleIds_sameIdsOnDescription)
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

TEST_F(DataTransferTests, createGenomeIds)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().genome(GenomeDescription()).cells({CellDescription()}),
        CreatureDescription().genome(GenomeDescription()).cells({CellDescription()})
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> ids;
    for (auto const& genome : actualData._creatures) {
        ids.insert(genome._id);
    }

    EXPECT_EQ(4, ids.size());
}

TEST_F(DataTransferTests, keepObjectIdsStable)
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

TEST_F(DataTransferTests, createCreatureIds_differentIds)
{
    auto data = CollectionDescription()
                    .creatures({
                        CreatureDescription().id(0).cells({CellDescription()}),
                        CreatureDescription().id(1).cells({CellDescription()}),
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

TEST_F(DataTransferTests, createCreatureIds_sameIds)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).cells({CellDescription(), CellDescription()}),
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

TEST_F(DataTransferTests, changeGenome_successful)
{
    auto const CreatureId = 1;
    auto data = CollectionDescription().creatures({CreatureDescription().id(CreatureId).genome(GenomeDescription()).cells({CellDescription()})});

    _simulationFacade->setSimulationData(data);

    auto newGenome = GenomeDescription().genes({GeneDescription().nodes({NodeDescription(), NodeDescription()})});
    auto result = _simulationFacade->changeCreature(CreatureId, newGenome);
    ASSERT_TRUE(result);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(1, creature._cells.size());
    EXPECT_EQ(CreatureId, creature._id);

    ASSERT_EQ(1, creature._genome._genes.size());

    auto gene = creature._genome._genes.front();
    EXPECT_EQ(2, gene._nodes.size());
}

TEST_F(DataTransferTests, changeGenome_failed)
{
    auto const CreatureId = 1;
    auto const WrongCreatureId = 2;
    auto data = CollectionDescription().creatures({CreatureDescription().id(CreatureId).genome(GenomeDescription()).cells({CellDescription()})});

    _simulationFacade->setSimulationData(data);

    auto newGenome = GenomeDescription().genes({GeneDescription().nodes({NodeDescription(), NodeDescription()})});
    auto result = _simulationFacade->changeCreature(WrongCreatureId, newGenome);
    ASSERT_FALSE(result);
}

TEST_F(DataTransferTests, getInspectedSimulationData)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().genome(GenomeDescription().genes({GeneDescription().nodes({NodeDescription(), NodeDescription()})})).cells({
            CellDescription().id(1), 
            CellDescription().id(2)
        }),
        CreatureDescription().genome(GenomeDescription()).cells({
            CellDescription().id(3)
        })
    });
    uint64_t cellId1 = 1;
    uint64_t cellId2 = 2;

    _simulationFacade->setSimulationData(data);

    auto inspectedData = _simulationFacade->getInspectedSimulationData({cellId1, cellId2});
    ASSERT_EQ(1, inspectedData._creatures.size());

    auto creature = inspectedData._creatures.front();
    EXPECT_EQ(2, creature._cells.size());

    auto cell1 = inspectedData.getCellRef(cellId1);
    auto cell2 = inspectedData.getCellRef(cellId2);

    ASSERT_EQ(1, creature._genome._genes.size());

    auto gene = creature._genome._genes.front();
    EXPECT_EQ(2, gene._nodes.size());
}
