#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <Base/Definitions.h>

#include <Data/Descs.h>
#include <Data/GenomeDesc.h>

#include <Persister/Interface/GenomeMassEditService.h>

class GenomeMassEditServiceTests : public ::testing::Test
{
protected:
    struct CreatureParameters
    {
        int numCells = 1;
        int color = 0;
        RealVector2D pos;
        int lineageId = 0;
    };

    void addCreature(ContentDesc& content, uint64_t creatureId, GenomeDesc const& genome, CreatureParameters const& parameters = {})
    {
        std::vector<ObjectDesc> cells;
        for (auto index : std::views::iota(0, parameters.numCells)) {
            cells.emplace_back(
                ObjectDesc().id(_nextObjectId++).pos({parameters.pos.x + toFloat(index), parameters.pos.y}).color(parameters.color).type(CellDesc()));
        }
        content.addCreature(cells, CreatureDesc().id(creatureId).lineageId(parameters.lineageId), genome);
    }

    GenomeDesc createGenome(uint64_t genomeId, std::vector<NodeDesc> const& nodes) const { return GenomeDesc().id(genomeId).genes({GeneDesc().nodes(nodes)}); }

    std::vector<NodeDesc> const& getNodes(ContentDesc const& content, uint64_t genomeId, int geneIndex = 0) const
    {
        return content.getGenomeRef(genomeId)._genes.at(geneIndex)._nodes;
    }

    GenomeEditResult edit(ContentDesc& content, GenomeFilter const& filter, GenomeEdit const& genomeEdit) const
    {
        return GenomeMassEditService::get().edit(content, filter, {genomeEdit});
    }

    std::string getErrorMessage(ContentDesc content, GenomeEdit const& genomeEdit) const
    {
        try {
            edit(content, {}, genomeEdit);
        } catch (std::invalid_argument const& exception) {
            return exception.what();
        }
        return {};
    }

private:
    uint64_t _nextObjectId = 1000;
};

TEST_F(GenomeMassEditServiceTests, patchMutationRatesOfGenomesWithNodeColor)
{
    auto genome1 = createGenome(1, {NodeDesc().color(3)});
    genome1._mutationRates._deleteNodeMutation._nodeProbability = 0.2f;
    auto genome2 = createGenome(2, {NodeDesc().color(1)});
    ContentDesc content;
    addCreature(content, 10, genome1);
    addCreature(content, 20, genome2);

    auto result = edit(
        content,
        GenomeFilter{.nodeColors = std::vector{3}},
        GenomeEdit{.operation = GenomePatchOperation{.patch = boost::json::parse(R"({"mutationRates": {"addNodeMutation": {"nodeProbability": 0.01}}})")}});

    EXPECT_EQ(1, result.numGenomes);
    EXPECT_EQ(1, result.numModifiedGenomes);
    EXPECT_EQ(0, result.numCorrectedGenomes);
    auto const& mutationRates = content.getGenomeRef(1)._mutationRates;
    EXPECT_EQ(0.01f, mutationRates._addNodeMutation._nodeProbability);
    EXPECT_EQ(0.2f, mutationRates._deleteNodeMutation._nodeProbability);
    EXPECT_TRUE(content.getGenomeRef(2) == genome2);
}

TEST_F(GenomeMassEditServiceTests, clampSensorRanges)
{
    ContentDesc content;
    addCreature(
        content,
        10,
        createGenome(
            1,
            {NodeDesc().cellType(SensorGenomeDesc().maxRange(80)),
             NodeDesc().cellType(SensorGenomeDesc().maxRange(30)),
             NodeDesc().cellType(MuscleGenomeDesc())}));

    auto result = edit(
        content,
        {},
        GenomeEdit{
            .level = GenomeEditLevel::Node,
            .nodeFilter = {.cellTypeNames = std::vector<std::string>{"sensor"}},
            .operation = GenomeClampOperation{.path = JsonPath("cellType.sensor.maxRange"), .max = 40}});

    auto const& nodes = getNodes(content, 1);
    EXPECT_EQ(40, std::get<SensorGenomeDesc>(nodes.at(0)._cellType)._maxRange);
    EXPECT_EQ(30, std::get<SensorGenomeDesc>(nodes.at(1)._cellType)._maxRange);
    EXPECT_TRUE(std::holds_alternative<MuscleGenomeDesc>(nodes.at(2)._cellType));
    auto const& statistics = result.editStatistics.at(0);
    EXPECT_EQ(1, statistics.numGenomesWithTargets);
    EXPECT_EQ(2, statistics.numTargets);
    EXPECT_EQ(2, statistics.numValues);
    EXPECT_EQ(1, statistics.numChangedValues);
    EXPECT_EQ(0, result.numCorrectedGenomes);
}

TEST_F(GenomeMassEditServiceTests, pathSkipsOtherCellTypes)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc().cellType(SensorGenomeDesc().maxRange(80)), NodeDesc().cellType(MuscleGenomeDesc())}));

    auto result = edit(
        content, {}, GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeSetOperation{.path = JsonPath("cellType.sensor.maxRange"), .value = 50}});

    EXPECT_EQ(50, std::get<SensorGenomeDesc>(getNodes(content, 1).at(0)._cellType)._maxRange);
    EXPECT_EQ(2, result.editStatistics.at(0).numTargets);
    EXPECT_EQ(1, result.editStatistics.at(0).numValues);
}

TEST_F(GenomeMassEditServiceTests, scaleKeepsIntegers)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc().cellType(SensorGenomeDesc().maxRange(80))}));

    edit(
        content,
        {},
        GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeScaleOperation{.path = JsonPath("cellType.sensor.maxRange"), .factor = 0.33}});

    EXPECT_EQ(26, std::get<SensorGenomeDesc>(getNodes(content, 1).at(0)._cellType)._maxRange);
}

TEST_F(GenomeMassEditServiceTests, scaleGeneStiffness)
{
    ContentDesc content;
    addCreature(content, 10, GenomeDesc().id(1).genes({GeneDesc().stiffness(0.5f).nodes({NodeDesc()}), GeneDesc().stiffness(0.8f).nodes({NodeDesc()})}));

    edit(
        content,
        {},
        GenomeEdit{
            .level = GenomeEditLevel::Gene,
            .geneFilter = {.indices = std::vector{1}},
            .operation = GenomeScaleOperation{.path = JsonPath("stiffness"), .factor = 0.5}});

    auto const& genes = content.getGenomeRef(1)._genes;
    EXPECT_EQ(0.5f, genes.at(0)._stiffness);
    EXPECT_EQ(0.4f, genes.at(1)._stiffness);
    EXPECT_EQ(1, genes.at(1)._nodes.size());
}

TEST_F(GenomeMassEditServiceTests, copyGenomeSharedWithOtherCreatures)
{
    auto genome = createGenome(1, {NodeDesc().color(2)});
    ContentDesc content;
    addCreature(content, 10, genome);
    addCreature(content, 20, genome);
    ASSERT_EQ(1, content._genomes.size());

    auto result = edit(
        content,
        GenomeFilter{.creatureIds = std::vector<uint64_t>{10}},
        GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeSetOperation{.path = JsonPath("color"), .value = 5}});

    EXPECT_EQ(1, result.numCopiedGenomes);
    ASSERT_EQ(2, content._genomes.size());
    EXPECT_TRUE(content.hasUniqueIds());
    auto const& creature1 = content._creatures.at(0);
    auto const& creature2 = content._creatures.at(1);
    EXPECT_NE(1, creature1._genomeId);
    EXPECT_EQ(1, creature2._genomeId);
    EXPECT_EQ(5, getNodes(content, creature1._genomeId).at(0)._color);
    EXPECT_EQ(2, getNodes(content, 1).at(0)._color);
    EXPECT_EQ(creature1._genomeId, result.exampleGenomeId.value_or(0));
}

TEST_F(GenomeMassEditServiceTests, remapColors)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc().color(2), NodeDesc().color(3), NodeDesc().color(2)}));

    edit(
        content,
        {},
        GenomeEdit{
            .level = GenomeEditLevel::Node, .operation = GenomeRemapOperation{.path = JsonPath("color"), .entries = {GenomeRemapEntry{.from = 2, .to = 5}}}});

    auto const& nodes = getNodes(content, 1);
    EXPECT_EQ(5, nodes.at(0)._color);
    EXPECT_EQ(3, nodes.at(1)._color);
    EXPECT_EQ(5, nodes.at(2)._color);
}

TEST_F(GenomeMassEditServiceTests, randomizeColorPerGenome)
{
    ContentDesc content;
    for (auto genomeId : std::views::iota(1, 6)) {
        addCreature(content, 10 + genomeId, createGenome(genomeId, {NodeDesc(), NodeDesc(), NodeDesc()}));
    }

    edit(
        content,
        {},
        GenomeEdit{
            .level = GenomeEditLevel::Node,
            .operation = GenomeRandomizeOperation{.path = JsonPath("color"), .values = {6, 7, 8}, .granularity = GenomeEditGranularity::Genome}});

    std::set const allowedColors{6, 7, 8};
    for (auto const& genome : content._genomes) {
        auto const& nodes = genome._genes.at(0)._nodes;
        EXPECT_TRUE(allowedColors.contains(nodes.at(0)._color));
        EXPECT_EQ(nodes.at(0)._color, nodes.at(1)._color);
        EXPECT_EQ(nodes.at(0)._color, nodes.at(2)._color);
    }
}

TEST_F(GenomeMassEditServiceTests, randomizeIntegerRange)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, std::vector<NodeDesc>(20, NodeDesc().cellType(SensorGenomeDesc()))));

    edit(
        content,
        {},
        GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeRandomizeOperation{.path = JsonPath("cellType.sensor.maxRange"), .min = 10, .max = 12}});

    for (auto const& node : getNodes(content, 1)) {
        auto maxRange = std::get<SensorGenomeDesc>(node._cellType)._maxRange;
        EXPECT_GE(maxRange, 10);
        EXPECT_LE(maxRange, 12);
    }
}

TEST_F(GenomeMassEditServiceTests, nodeConditions)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc().cellType(SensorGenomeDesc().maxRange(80)), NodeDesc().cellType(SensorGenomeDesc().maxRange(30))}));

    edit(
        content,
        {},
        GenomeEdit{
            .level = GenomeEditLevel::Node,
            .nodeFilter = {.conditions = {JsonCondition{.path = JsonPath("cellType.sensor.maxRange"), .op = JsonConditionOperator::Greater, .value = 50}}},
            .operation = GenomeSetOperation{.path = JsonPath("cellType.sensor.maxRange"), .value = 50}});

    auto const& nodes = getNodes(content, 1);
    EXPECT_EQ(50, std::get<SensorGenomeDesc>(nodes.at(0)._cellType)._maxRange);
    EXPECT_EQ(30, std::get<SensorGenomeDesc>(nodes.at(1)._cellType)._maxRange);
}

TEST_F(GenomeMassEditServiceTests, existsIgnoresMissingConstructor)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc(), NodeDesc().constructor(ConstructorGenomeDesc())}));

    edit(
        content,
        {},
        GenomeEdit{
            .level = GenomeEditLevel::Node,
            .nodeFilter = {.conditions = {JsonCondition{.path = JsonPath("constructor"), .op = JsonConditionOperator::Exists, .value = true}}},
            .operation = GenomeSetOperation{.path = JsonPath("color"), .value = 4}});

    auto const& nodes = getNodes(content, 1);
    EXPECT_EQ(0, nodes.at(0)._color);
    EXPECT_EQ(4, nodes.at(1)._color);
}

TEST_F(GenomeMassEditServiceTests, patchNodeSwitchesCellType)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc().color(4).cellType(MuscleGenomeDesc())}));

    edit(
        content,
        {},
        GenomeEdit{
            .level = GenomeEditLevel::Node, .operation = GenomePatchOperation{.patch = boost::json::parse(R"({"cellType": {"sensor": {"maxRange": 20}}})")}});

    auto const& node = getNodes(content, 1).at(0);
    ASSERT_TRUE(std::holds_alternative<SensorGenomeDesc>(node._cellType));
    EXPECT_EQ(20, std::get<SensorGenomeDesc>(node._cellType)._maxRange);
    EXPECT_EQ(4, node._color);
}

TEST_F(GenomeMassEditServiceTests, setReplacesObjects)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc().cellType(SensorGenomeDesc().minRange(10).maxRange(80))}));

    edit(
        content,
        {},
        GenomeEdit{
            .level = GenomeEditLevel::Node,
            .operation = GenomeSetOperation{.path = JsonPath("cellType"), .value = boost::json::parse(R"({"sensor": {"maxRange": 50}})")}});

    auto const& sensor = std::get<SensorGenomeDesc>(getNodes(content, 1).at(0)._cellType);
    EXPECT_EQ(50, sensor._maxRange);
    EXPECT_EQ(SensorGenomeDesc()._minRange, sensor._minRange);
}

TEST_F(GenomeMassEditServiceTests, jitterBiasesKeepsWeights)
{
    auto node = NodeDesc().cellType(MuscleGenomeDesc());
    node._neuralNetwork.weight(1, 2, 0.5f);
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {node}));

    edit(
        content, {}, GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeJitterOperation{.path = JsonPath("neuralNetwork.biases[*]"), .sigma = 0.1}});

    auto const& neuralNetwork = getNodes(content, 1).at(0)._neuralNetwork;
    EXPECT_TRUE(neuralNetwork._weights == node._neuralNetwork._weights);
    EXPECT_FALSE(neuralNetwork._biases == node._neuralNetwork._biases);
}

TEST_F(GenomeMassEditServiceTests, removeNodes)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc(), NodeDesc().cellType(InjectorGenomeDesc()), NodeDesc(), NodeDesc().cellType(InjectorGenomeDesc())}));

    auto result = edit(
        content,
        {},
        GenomeEdit{
            .level = GenomeEditLevel::Node, .nodeFilter = {.cellTypeNames = std::vector<std::string>{"injector"}}, .operation = GenomeRemoveNodesOperation()});

    auto const& nodes = getNodes(content, 1);
    ASSERT_EQ(2, nodes.size());
    EXPECT_TRUE(std::holds_alternative<BaseGenomeDesc>(nodes.at(0)._cellType));
    EXPECT_TRUE(std::holds_alternative<BaseGenomeDesc>(nodes.at(1)._cellType));
    EXPECT_EQ(2, result.editStatistics.at(0).numTargets);
}

TEST_F(GenomeMassEditServiceTests, insertNode)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc(), NodeDesc()}));

    edit(content, {}, GenomeEdit{.level = GenomeEditLevel::Gene, .operation = GenomeInsertNodeOperation{.node = NodeDesc().color(4), .index = 0}});

    auto const& nodes = getNodes(content, 1);
    ASSERT_EQ(3, nodes.size());
    EXPECT_EQ(4, nodes.at(0)._color);
}

TEST_F(GenomeMassEditServiceTests, insertGeneShiftsReferences)
{
    ContentDesc content;
    addCreature(
        content,
        10,
        GenomeDesc().id(1).genes({
            GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
            GeneDesc().nodes({NodeDesc()}),
        }));

    edit(
        content,
        {},
        GenomeEdit{
            .operation = GenomeInsertGeneOperation{.gene = GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}), .index = 1}});

    auto const& genes = content.getGenomeRef(1)._genes;
    ASSERT_EQ(3, genes.size());
    EXPECT_EQ(2, genes.at(0)._nodes.at(0)._constructor->_geneIndex);
    EXPECT_EQ(2, genes.at(1)._nodes.at(0)._constructor->_geneIndex);
}

TEST_F(GenomeMassEditServiceTests, insertUnreferencedGeneReportsIssue)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc()}));

    auto result = edit(content, {}, GenomeEdit{.operation = GenomeInsertGeneOperation{.gene = GeneDesc().nodes({NodeDesc()})}});

    EXPECT_EQ(2, content.getGenomeRef(1)._genes.size());
    EXPECT_EQ(1, result.numGenomesWithNewIssues);
    ASSERT_TRUE(result.exampleIssue.has_value());
    EXPECT_NE(std::string::npos, result.exampleIssue->find("unreachable"));
}

TEST_F(GenomeMassEditServiceTests, removeGenes)
{
    ContentDesc content;
    addCreature(
        content,
        10,
        GenomeDesc().id(1).genes({
            GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)), NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2))}),
            GeneDesc().nodes({NodeDesc()}),
            GeneDesc().nodes({NodeDesc()}),
        }));

    edit(content, {}, GenomeEdit{.level = GenomeEditLevel::Gene, .geneFilter = {.indices = std::vector{1}}, .operation = GenomeRemoveGenesOperation()});

    auto const& genes = content.getGenomeRef(1)._genes;
    ASSERT_EQ(2, genes.size());
    EXPECT_FALSE(genes.at(0)._nodes.at(0)._constructor.has_value());
    EXPECT_EQ(1, genes.at(0)._nodes.at(1)._constructor->_geneIndex);
}

TEST_F(GenomeMassEditServiceTests, removingAllGenesIsRejected)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc()}));
    auto origContent = content;
    auto removeAllGenes = GenomeEdit{.level = GenomeEditLevel::Gene, .operation = GenomeRemoveGenesOperation()};

    EXPECT_THROW(edit(content, {}, removeAllGenes), std::invalid_argument);
    EXPECT_TRUE(origContent == content);
}

TEST_F(GenomeMassEditServiceTests, replaceGene)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc()}));

    edit(
        content,
        {},
        GenomeEdit{.level = GenomeEditLevel::Gene, .operation = GenomeReplaceGeneOperation{.gene = GeneDesc().name("leg").nodes({NodeDesc(), NodeDesc()})}});

    auto const& gene = content.getGenomeRef(1)._genes.at(0);
    EXPECT_EQ("leg", gene._name);
    EXPECT_EQ(2, gene._nodes.size());
}

TEST_F(GenomeMassEditServiceTests, validationCorrectsValues)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc()}));

    auto result = edit(content, {}, GenomeEdit{.operation = GenomeSetOperation{.path = JsonPath("mutationRates.addNodeMutation.nodeProbability"), .value = 5}});

    EXPECT_EQ(1, result.numCorrectedGenomes);
    EXPECT_FALSE(result.exampleCorrections.firstChanges.empty());
    EXPECT_EQ(1.0f, content.getGenomeRef(1)._mutationRates._addNodeMutation._nodeProbability);
}

TEST_F(GenomeMassEditServiceTests, exampleChanges)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc().color(2)}));

    auto result = edit(content, {}, GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeSetOperation{.path = JsonPath("color"), .value = 6}});

    EXPECT_EQ(1, result.exampleGenomeId);
    ASSERT_EQ(1, result.exampleChanges.firstChanges.size());
    auto const& change = result.exampleChanges.firstChanges.front();
    EXPECT_EQ("genes[0].nodes[0].color", change.path);
    EXPECT_EQ(2, change.before->as_int64());
    EXPECT_EQ(6, change.after->as_int64());
}

TEST_F(GenomeMassEditServiceTests, similarGenomes)
{
    auto referenceGenome = createGenome(1, {NodeDesc().cellType(SensorGenomeDesc()), NodeDesc().cellType(MuscleGenomeDesc())});
    ContentDesc content;
    addCreature(content, 10, referenceGenome);
    addCreature(content, 20, createGenome(2, {NodeDesc().color(3).cellType(SensorGenomeDesc().maxRange(10)), NodeDesc().cellType(MuscleGenomeDesc())}));
    addCreature(content, 30, createGenome(3, {NodeDesc().cellType(SensorGenomeDesc()), NodeDesc().cellType(AttackerGenomeDesc())}));

    auto summary = GenomeMassEditService::get().summarize(content, GenomeFilter{.similarTo = referenceGenome}, {});
    EXPECT_EQ(2, summary.numGenomes);

    summary = GenomeMassEditService::get().summarize(content, GenomeFilter{.similarTo = referenceGenome, .maxNodeDifferences = 1}, {});
    EXPECT_EQ(3, summary.numGenomes);
}

TEST_F(GenomeMassEditServiceTests, creatureFilters)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc()}), {.numCells = 3, .pos = {10, 10}});
    addCreature(content, 20, createGenome(2, {NodeDesc()}), {.numCells = 1, .color = 4, .pos = {100, 100}, .lineageId = 7});

    auto getNumGenomes = [&](GenomeFilter const& filter) { return GenomeMassEditService::get().summarize(content, filter, {}).numGenomes; };
    EXPECT_EQ(2, getNumGenomes({}));
    EXPECT_EQ(1, getNumGenomes({.minCells = 2}));
    EXPECT_EQ(1, getNumGenomes({.cellColors = std::vector{4}}));
    EXPECT_EQ(1, getNumGenomes({.lineageIds = std::vector{7}}));
    EXPECT_EQ(1, getNumGenomes({.area = RealRect{.topLeft = {0, 0}, .bottomRight = {20, 20}}}));
    EXPECT_EQ(1, getNumGenomes({.objectIds = std::vector<uint64_t>{content._objects.back()._id}}));
    EXPECT_EQ(0, getNumGenomes({.cellColors = std::vector{4}, .minCells = 2}));
}

TEST_F(GenomeMassEditServiceTests, hasNodeFilter)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc().cellType(AttackerGenomeDesc())}));
    addCreature(content, 20, createGenome(2, {NodeDesc().cellType(SensorGenomeDesc())}));

    auto summary =
        GenomeMassEditService::get().summarize(content, GenomeFilter{.hasNode = GenomeNodeFilter{.cellTypeNames = std::vector<std::string>{"attacker"}}}, {});

    EXPECT_EQ(1, summary.numGenomes);
    ASSERT_EQ(1, summary.numNodesByCellType.size());
    EXPECT_EQ("attacker", summary.numNodesByCellType.front().first);
}

TEST_F(GenomeMassEditServiceTests, nothingMatches)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc().color(1)}));
    auto origContent = content;

    auto result = edit(
        content,
        GenomeFilter{.nodeColors = std::vector{5}},
        GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeSetOperation{.path = JsonPath("color"), .value = 2}});

    EXPECT_EQ(0, result.numGenomes);
    EXPECT_EQ(0, result.numModifiedGenomes);
    EXPECT_TRUE(origContent == content);
}

TEST_F(GenomeMassEditServiceTests, summarizeFields)
{
    ContentDesc content;
    addCreature(
        content, 10, createGenome(1, {NodeDesc().cellType(SensorGenomeDesc().maxRange(80)), NodeDesc().cellType(MuscleGenomeDesc())}), {.lineageId = 3});
    addCreature(content, 20, createGenome(2, {NodeDesc().cellType(SensorGenomeDesc().maxRange(40))}), {.lineageId = 3});

    auto summary =
        GenomeMassEditService::get().summarize(content, {}, {GenomeFieldQuery{.level = GenomeEditLevel::Node, .path = JsonPath("cellType.sensor.maxRange")}});

    EXPECT_EQ(2, summary.numCreatures);
    EXPECT_EQ(2, summary.numGenomes);
    EXPECT_EQ(1, summary.numLineages);
    EXPECT_EQ(1, summary.genesPerGenome.max);
    EXPECT_EQ(1, summary.nodesPerGenome.min);
    EXPECT_EQ(2, summary.nodesPerGenome.max);
    ASSERT_EQ(1, summary.fields.size());
    auto const& field = summary.fields.front();
    EXPECT_EQ(2, field.numValues);
    EXPECT_EQ(40, field.min);
    EXPECT_EQ(80, field.max);
    EXPECT_EQ(60, field.mean);
    EXPECT_EQ(2, field.frequentValues.size());
    EXPECT_EQ(2, summary.exampleCellIds.size());
}

TEST_F(GenomeMassEditServiceTests, invalidEdits)
{
    ContentDesc content;
    addCreature(content, 10, createGenome(1, {NodeDesc()}));

    auto message = getErrorMessage(
        content, GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeSetOperation{.path = JsonPath("cellType.sensor.maxrange"), .value = 1}});
    EXPECT_NE(std::string::npos, message.find("maxRange"));

    message =
        getErrorMessage(content, GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeScaleOperation{.path = JsonPath("cellType"), .factor = 2}});
    EXPECT_NE(std::string::npos, message.find("numbers"));

    message = getErrorMessage(content, GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeSetOperation{.path = JsonPath("color"), .value = "red"}});
    EXPECT_NE(std::string::npos, message.find("color"));

    message = getErrorMessage(content, GenomeEdit{.operation = GenomePatchOperation{.patch = boost::json::parse(R"({"genes": []})")}});
    EXPECT_NE(std::string::npos, message.find("gene"));

    message = getErrorMessage(content, GenomeEdit{.level = GenomeEditLevel::Node, .operation = GenomeInsertGeneOperation{}});
    EXPECT_NE(std::string::npos, message.find("genome"));

    message = getErrorMessage(
        content,
        GenomeEdit{
            .level = GenomeEditLevel::Node,
            .nodeFilter = {.cellTypeNames = std::vector<std::string>{"teleporter"}},
            .operation = GenomeRemoveNodesOperation()});
    EXPECT_NE(std::string::npos, message.find("teleporter"));
}
