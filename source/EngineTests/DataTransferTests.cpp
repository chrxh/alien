#include <algorithm>
#include <ranges>

#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
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
    Description data;
    data._particles.emplace_back(_descriptionTestDataFactory->createNonDefaultParticleDescription());

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

using CellParameter = DescriptionTestDataFactory::CellParameter;
class DataTransferTests_AllCellTypes
    : public DataTransferTests
    , public testing::WithParamInterface<CellParameter>
{};

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

    Description data;
    data._cells.emplace_back(_descriptionTestDataFactory->createNonDefaultCellDescription(cellParameter));
    data._cells.emplace_back(_descriptionTestDataFactory->createNonDefaultCellDescription(cellParameter));

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_P(DataTransferTests_AllCellTypes, cellsWithoutCreature_preview)
{
    auto cellParameter = GetParam();

    Description data;
    data._cells.emplace_back(_descriptionTestDataFactory->createNonDefaultCellDescription(cellParameter));
    data._cells.emplace_back(_descriptionTestDataFactory->createNonDefaultCellDescription(cellParameter));

    _simulationFacade->setPreviewData(data);
    auto actualData = _simulationFacade->getPreviewData();

    EXPECT_TRUE(compare(data, actualData));
}

using NodeParameter = DescriptionTestDataFactory::NodeParameter;
class DataTransferTests_AllNodeTypes
    : public DataTransferTests
    , public testing::WithParamInterface<NodeParameter>
{};

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

    auto data = Description().creatures({
        _descriptionTestDataFactory->createNonDefaultCreatureDescription(nodeParameter).cells({CellDescription()}),
        _descriptionTestDataFactory->createNonDefaultCreatureDescription(nodeParameter).cells({CellDescription()}),
    });

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_P(DataTransferTests_AllNodeTypes, cellsWithCreatures_oneNode_preview)
{
    auto nodeParameter = GetParam();

    auto data = Description().creatures({
        _descriptionTestDataFactory->createNonDefaultCreatureDescription(nodeParameter).cells({CellDescription()}),
        _descriptionTestDataFactory->createNonDefaultCreatureDescription(nodeParameter).cells({CellDescription()}),
    });

    _simulationFacade->setPreviewData(data);
    auto actualData = _simulationFacade->getPreviewData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, multipleCells_genome_multipleGenes_multipleNodes)
{
    auto hexagon = DescriptionEditService::get().createHex(DescriptionEditService::CreateHexParameters().center({100.0f, 100.0f}).cellType(BaseDescription()));
    Description data;
    data.creatures({
        CreatureDescription()
            .genome(GenomeDescription().genes({
                GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),
                GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription(), NodeDescription()}),
            }))
            .cells(hexagon._cells),
    });

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, setSimulationData_keepIdsStable)
{
    auto data = Description()
                    .cells({CellDescription().id(0), CellDescription().id(1)})
                    .particles({ParticleDescription().id(2), ParticleDescription().id(3)})
                    .creatures({CreatureDescription().id(4).cells({CellDescription().id(5)}), CreatureDescription().id(5).cells({CellDescription().id(6)})});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->setSimulationData(data);

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> expectedCellIds;
    data.forEachCell([&](auto const& cell) { expectedCellIds.insert(cell._id); });
    std::unordered_set<uint64_t> actualCellIds;
    actualData.forEachCell([&](auto const& cell) { actualCellIds.insert(cell._id); });
    EXPECT_EQ(expectedCellIds, actualCellIds);

    std::unordered_set<uint64_t> expectedCreatureIds;
    for (auto const& creature: data._creatures) {
        expectedCreatureIds.insert(creature._id);
    }
    std::unordered_set<uint64_t> actualCreatureIds;
    for (auto const& creature : actualData._creatures) {
        actualCreatureIds.insert(creature._id);
    }
    EXPECT_EQ(expectedCreatureIds, actualCreatureIds);

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

TEST_F(DataTransferTests, addAndSelectSimulationData_assignNewIds)
{
    auto data = Description()
                    .cells({CellDescription().id(0), CellDescription().id(1)})
                    .particles({ParticleDescription().id(2), ParticleDescription().id(3)})
                    .creatures({CreatureDescription().id(4).cells({CellDescription().id(5)}), CreatureDescription().id(5).cells({CellDescription().id(6)})});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> actualCellIds;
    actualData.forEachCell([&](auto const& cell) { actualCellIds.insert(cell._id); });
    EXPECT_EQ(2 * 4, actualCellIds.size());

    std::unordered_set<uint64_t> actualCreatureIds;
    for (auto const& creature : actualData._creatures) {
        actualCreatureIds.insert(creature._id);
    }
    EXPECT_EQ(2 * 2, actualCreatureIds.size());

    std::unordered_set<uint64_t> actualParticleIds;
    for (auto const& particle : actualData._particles) {
        actualParticleIds.insert(particle._id);
    }
    EXPECT_EQ(2 * 2, actualParticleIds.size());
}

TEST_F(DataTransferTests, changeGenome_successful)
{
    auto const CreatureId = 1;
    auto data = Description().creatures({CreatureDescription().id(CreatureId).genome(GenomeDescription()).cells({CellDescription()})});

    _simulationFacade->setSimulationData(data);

    auto newGenome = GenomeDescription().genes({GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()})});
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
    auto data = Description().creatures({CreatureDescription().id(CreatureId).genome(GenomeDescription()).cells({CellDescription()})});

    _simulationFacade->setSimulationData(data);

    auto newGenome = GenomeDescription().genes({GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()})});
    auto result = _simulationFacade->changeCreature(WrongCreatureId, newGenome);
    ASSERT_FALSE(result);
}

TEST_F(DataTransferTests, getInspectedSimulationData)
{
    auto data = Description().creatures(
        {CreatureDescription()
             .genome(GenomeDescription().genes({GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()})}))
             .cells({CellDescription().id(1), CellDescription().id(2)}),
         CreatureDescription().genome(GenomeDescription()).cells({CellDescription().id(3)})});
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
