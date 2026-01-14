#include <algorithm>
#include <ranges>

#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

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

TEST_F(DataTransferTests, twoCreaturesSharingOneGenome)
{
    auto genome = GenomeDescription().genes(
        {GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),
         GeneDescription().separation(false).nodes({NodeDescription(), NodeDescription(), NodeDescription()})});

    Description data;
    data.addCreature(
        CreatureDescription().cells(
            {CellDescription().id(1).pos({10.0f, 10.0f}).cellType(BaseDescription()), CellDescription().id(2).pos({11.0f, 10.0f}).cellType(BaseDescription())}),
        genome);
    data.addCreature(
        CreatureDescription().cells(
            {CellDescription().id(3).pos({20.0f, 20.0f}).cellType(BaseDescription()), CellDescription().id(4).pos({21.0f, 20.0f}).cellType(BaseDescription())}),
        genome);
    data.addConnection(1, 2);
    data.addConnection(3, 4);

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
    ::testing::ValuesIn(DescriptionTestDataFactory::get().getAllCellParameters()));

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
    ::testing::ValuesIn(DescriptionTestDataFactory::get().getAllNodeParameters()));

TEST_P(DataTransferTests_AllNodeTypes, cellsWithCreatures_oneNode)
{
    auto nodeParameter = GetParam();

    auto [creature1, genome1] = _descriptionTestDataFactory->createNonDefaultCreatureDescription(nodeParameter);
    auto [creature2, genome2] = _descriptionTestDataFactory->createNonDefaultCreatureDescription(nodeParameter);

    Description data;
    data.addCreature(creature1, {CellDescription()}, genome1);
    data.addCreature(creature2, {CellDescription()}, genome2);

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_P(DataTransferTests_AllNodeTypes, cellsWithCreatures_oneNode_preview)
{
    auto nodeParameter = GetParam();

    auto [creature, genome] = _descriptionTestDataFactory->createNonDefaultCreatureDescription(nodeParameter);

    Description data;
    data.addCreature(creature, {CellDescription()}, genome);

    _simulationFacade->setPreviewData(data);
    auto actualData = _simulationFacade->getPreviewData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, multipleCells_genome_multipleGenes_multipleNodes)
{
    auto hexagon = DescriptionEditService::get().createHex(DescriptionEditService::CreateHexParameters().center({100.0f, 100.0f}).cellType(BaseDescription()));


    auto data = Description().addCreature(
        CreatureDescription().cells(hexagon._cells),
        GenomeDescription().genes({
            GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),
            GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription(), NodeDescription()}),
        }));

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, setSimulationData_keepIdsStable)
{
    auto data = Description().cells({CellDescription().id(0), CellDescription().id(1)}).particles({ParticleDescription().id(2), ParticleDescription().id(3)});
    data.addCreature(CreatureDescription().id(4), {CellDescription().id(5)}), GenomeDescription());
    data.addCreature(CreatureDescription().id(5), {CellDescription().id(6)}), GenomeDescription());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->setSimulationData(data);

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> expectedCellIds;
    data.forEachCell([&](auto const& cell) { expectedCellIds.insert(cell._id); });
    std::unordered_set<uint64_t> actualCellIds;
    actualData.forEachCell([&](auto const& cell) { actualCellIds.insert(cell._id); });
    EXPECT_EQ(expectedCellIds, actualCellIds);

    std::unordered_set<uint64_t> expectedCreatureIds;
    for (auto const& creature : data._creatures) {
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
    auto data = Description().cells({CellDescription().id(0), CellDescription().id(1)}).particles({ParticleDescription().id(2), ParticleDescription().id(3)});
    data.addCreature(CreatureDescription().id(4), {CellDescription().id(5)}), GenomeDescription());
    data.addCreature(CreatureDescription().id(5), {CellDescription().id(6)}), GenomeDescription());

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

    auto data = Description().addCreature(CreatureDescription().id(CreatureId), {CellDescription()}), GenomeDescription());

    _simulationFacade->setSimulationData(data);

    auto newGenome = GenomeDescription().genes({GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()})});
    auto result = _simulationFacade->changeCreature(CreatureId, newGenome);
    ASSERT_TRUE(result);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(1, actualData.getCellsForCreature(creature._id).size());
    EXPECT_EQ(CreatureId, creature._id);

    // Find the genome for this creature
    auto genomeIt = std::find_if(actualData._genomes.begin(), actualData._genomes.end(), [&creature](auto const& g) { return g._id == creature._genomeId; });
    ASSERT_NE(genomeIt, actualData._genomes.end());
    ASSERT_EQ(1, genomeIt->_genes.size());

    auto gene = genomeIt->_genes.front();
    EXPECT_EQ(2, gene._nodes.size());
}

TEST_F(DataTransferTests, changeGenome_failed)
{
    auto constexpr CreatureId = 1;
    auto constexpr WrongCreatureId = 2;

    auto data = Description().addCreature(CreatureDescription().id(CreatureId), {CellDescription()}), GenomeDescription());

    _simulationFacade->setSimulationData(data);

    auto newGenome = GenomeDescription().genes({GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()})});
    auto result = _simulationFacade->changeCreature(WrongCreatureId, newGenome);
    ASSERT_FALSE(result);
}

TEST_F(DataTransferTests, getGenomeOfCreature_successful)
{
    auto constexpr CreatureId = 1;

    auto genome =
        GenomeDescription().name("Test Genome").genes({GeneDescription().name("Gene1").separation(true).nodes({NodeDescription(), NodeDescription()})});
    auto data = Description().addCreature(
        CreatureDescription().id(CreatureId), {CellDescription(), CellDescription(), CellDescription(), CellDescription()}), genome);

    _simulationFacade->setSimulationData(data);

    auto retrievedGenome = _simulationFacade->getGenomeOfCreature(CreatureId);

    ASSERT_TRUE(retrievedGenome.has_value());
    EXPECT_EQ("Test Genome", retrievedGenome->_name);
    ASSERT_EQ(1, retrievedGenome->_genes.size());
    EXPECT_EQ("Gene1", retrievedGenome->_genes.front()._name);
    EXPECT_TRUE(retrievedGenome->_genes.front()._separation);
    EXPECT_EQ(2, retrievedGenome->_genes.front()._nodes.size());
}

TEST_F(DataTransferTests, getGenomeOfCreature_nonexistentCreature)
{
    auto constexpr CreatureId = 1;
    auto constexpr WrongCreatureId = 2;

    auto data = Description().addCreature(CreatureDescription().id(CreatureId), {CellDescription()}), GenomeDescription());

    _simulationFacade->setSimulationData(data);

    auto result = _simulationFacade->getGenomeOfCreature(WrongCreatureId);
    EXPECT_FALSE(result.has_value());
}

TEST_F(DataTransferTests, getInspectedSimulationData)
{
    auto constexpr CreatureId1 = 1;
    auto constexpr CreatureId2 = 2;
    auto genome = GenomeDescription().genes({GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()})});
    auto genome2 = GenomeDescription();

    Description data;
    data.addCreature(CreatureDescription().id(CreatureId1), {CellDescription().id(1), CellDescription().id(2)}), genome);
    data.addCreature(CreatureDescription().id(CreatureId2), {CellDescription().id(3)}), genome2);

    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);

    auto inspectedData = _simulationFacade->getInspectedSimulationData({1, 2});
    ASSERT_EQ(2, inspectedData._creatures.size());

    auto creature = inspectedData.getCreatureRef(CreatureId1);

    EXPECT_EQ(2, actualData.getCellsForCreature(creature._id).size());

    auto cell1 = inspectedData.getCellRef(1);
    auto cell2 = inspectedData.getCellRef(2);

    // Find the genome for this creature
    auto genomeIt =
        std::find_if(inspectedData._genomes.begin(), inspectedData._genomes.end(), [&creature](auto const& g) { return g._id == creature._genomeId; });
    ASSERT_NE(genomeIt, inspectedData._genomes.end());
    EXPECT_EQ(genome, *genomeIt);
}

TEST_F(DataTransferTests, adaptIdGenerator_cells)
{
    auto constexpr HighId = 1000000;
    auto data = Description().cells({CellDescription().id(HighId)});
    _simulationFacade->setSimulationData(data);

    NumberGenerator::get().setIds({1});
    auto actualData = _simulationFacade->getSimulationData();

    auto newId = NumberGenerator::get().createId();
    EXPECT_TRUE(newId > HighId);
}

TEST_F(DataTransferTests, adaptIdGenerator_particles)
{
    auto constexpr HighId = 1000000;
    auto data = Description().particles({ParticleDescription().id(HighId)});
    _simulationFacade->setSimulationData(data);

    NumberGenerator::get().setIds({1});
    auto actualData = _simulationFacade->getSimulationData();

    auto newId = NumberGenerator::get().createId();
    EXPECT_TRUE(newId > HighId);
}

TEST_F(DataTransferTests, adaptIdGenerator_creatures)
{
    auto constexpr HighId = 1000000;
    auto data = Description().addCreature(CreatureDescription().id(HighId), {CellDescription().id(1)});
    _simulationFacade->setSimulationData(data);

    NumberGenerator::get().setIds({1});
    auto actualData = _simulationFacade->getSimulationData();

    auto newId = NumberGenerator::get().createId();
    EXPECT_TRUE(newId > HighId);
}

TEST_F(DataTransferTests, adaptIdGenerator_genomes)
{
    auto constexpr HighId = 1000000;
    auto data = Description().addCreature(CreatureDescription().id(1), {CellDescription().id(2)}), GenomeDescription().id(HighId));
    _simulationFacade->setSimulationData(data);

    NumberGenerator::get().setIds({1});
    auto actualData = _simulationFacade->getSimulationData();

    auto newId = NumberGenerator::get().createId();
    EXPECT_TRUE(newId > HighId);
}
