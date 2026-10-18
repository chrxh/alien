#include "McpGenomeEditTools.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <format>

#include <boost/algorithm/string/join.hpp>
#include <boost/json.hpp>

#include <Base/Definitions.h>
#include <Base/ExitScopeGuard.h>
#include <Base/StringHelper.h>

#include <Data/EngineConstants.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Persister/Interface/SerializerService.h>

#include <Network/McpArguments.h>
#include <Network/McpJson.h>
#include <Network/McpSchema.h>

#include "GenomeMassEditService.h"
#include "McpGenomeTools.h"

namespace
{
    size_t constexpr MaxValueTextLength = 100;

    std::vector<std::pair<std::string, std::vector<std::string>>> const OperationKeys = {
        {"patch", {"value"}},
        {"set", {"path", "value"}},
        {"add", {"path", "value"}},
        {"scale", {"path", "factor"}},
        {"clamp", {"path", "min", "max"}},
        {"randomize", {"path", "min", "max", "values", "per"}},
        {"jitter", {"path", "sigma", "per"}},
        {"remap", {"path", "map"}},
        {"insert_node", {"node", "from", "index"}},
        {"remove_nodes", {}},
        {"insert_gene", {"gene", "from", "index"}},
        {"replace_gene", {"gene", "from"}},
        {"remove_genes", {}},
    };

    std::vector<std::pair<std::string, JsonConditionOperator>> const ConditionOperators = {
        {"eq", JsonConditionOperator::Equal},
        {"ne", JsonConditionOperator::NotEqual},
        {"lt", JsonConditionOperator::Less},
        {"le", JsonConditionOperator::LessEqual},
        {"gt", JsonConditionOperator::Greater},
        {"ge", JsonConditionOperator::GreaterEqual},
        {"in", JsonConditionOperator::In},
        {"exists", JsonConditionOperator::Exists},
    };

    auto constexpr PathConventions =
        "Paths and patches use the JSON format of genomes, see get_json_format with kind 'genome'. They are relative to the level and do not include "
        "the child levels, i.e. the genome level has no 'genes' and the gene level no 'nodes'. '[n]' and '[*]' select array elements and '*' any field "
        "or alternative, e.g. 'cellType.*.mode.*.restrictToColors'.";

    boost::json::object withDescription(boost::json::object schema, std::string const& description)
    {
        schema["description"] = description;
        return schema;
    }

    boost::json::object colorList(std::string const& description)
    {
        return McpSchema::array(description, McpSchema::integer("Color index, see get_simulation_info", 0, MAX_COLORS - 1), 1);
    }

    boost::json::object conditionsSchema(std::string const& level)
    {
        auto condition = McpSchema::object(
            {
                {"path", McpSchema::string(std::format("Path relative to the {}", level))},
                {"eq", McpSchema::any("Equal to this value")},
                {"ne", McpSchema::any("The path exists and no value is equal to this value")},
                {"lt", McpSchema::number("Less than")},
                {"le", McpSchema::number("Less than or equal to")},
                {"gt", McpSchema::number("Greater than")},
                {"ge", McpSchema::number("Greater than or equal to")},
                {"in", McpSchema::array("Equal to one of these values", McpSchema::any("Value"), 1)},
                {"exists",
                 McpSchema::boolean(
                     "Whether the path exists with a value other than null, e.g. 'cellType.sensor' exists only for sensor nodes and 'constructor' only "
                     "for nodes with a constructor")},
            },
            {"path"});
        return McpSchema::array(
            std::format(
                "Conditions on the fields of the {}, all of them must hold. Each condition has a path and exactly one comparison, which holds if one of "
                "the values at the path fulfills it.",
                level),
            condition,
            1);
    }

    boost::json::object geneFilterSchema(std::string const& description)
    {
        return withDescription(
            McpSchema::object({
                {"indices", McpSchema::array("Gene indices, 0 is the root gene", McpSchema::integer("Gene index", 0), 1)},
                {"min_nodes", McpSchema::integer("Minimum number of nodes of a gene", 0)},
                {"max_nodes", McpSchema::integer("Maximum number of nodes of a gene", 0)},
                {"names", McpSchema::array("Gene names", McpSchema::string("Gene name"), 1)},
                {"has_node", McpSchema::jsonObject("Genes containing a node that matches this node filter, see 'has_node' of the tool's 'where'")},
                {"conditions", conditionsSchema("gene")},
            }),
            description);
    }

    boost::json::object nodeFilterSchema(std::string const& description)
    {
        return withDescription(
            McpSchema::object({
                {"gene", geneFilterSchema("Nodes in genes matching this gene filter")},
                {"indices", McpSchema::array("Node indices within the gene", McpSchema::integer("Node index", 0), 1)},
                {"cell_types", McpSchema::array("Cell types", McpSchema::enumeration("Cell type", GenomeMassEditService::get().getCellTypeNames()), 1)},
                {"colors", colorList("Node colors")},
                {"has_constructor", McpSchema::boolean("Whether the node has a constructor")},
                {"conditions", conditionsSchema("node")},
            }),
            description);
    }

    boost::json::object whereSchema()
    {
        return withDescription(
            McpSchema::object({
                {"creature_ids", McpSchema::ids("Creature ids", 1)},
                {"object_ids", McpSchema::ids("Creatures to which these cells belong", 1)},
                {"genome_ids", McpSchema::ids("Genome ids", 1)},
                {"lineage_ids", McpSchema::array("Lineage ids", McpSchema::integer("Lineage id"), 1)},
                {"cell_colors", colorList("Creatures with at least one cell of these colors")},
                {"node_colors", colorList("Genomes with at least one node of these colors")},
                {"min_cells", McpSchema::integer("Minimum number of cells of a creature", 0)},
                {"max_cells", McpSchema::integer("Maximum number of cells of a creature", 0)},
                {"min_generation", McpSchema::integer("Minimum generation", 0)},
                {"max_generation", McpSchema::integer("Maximum generation", 0)},
                {"mutation_states",
                 McpSchema::array(
                     "Mutation states: with 'not_mutated' the genome is copied and mutated before the creature constructs its next offspring, with "
                     "'mutated' the offspring receive the current genome",
                     McpSchema::enumeration("Mutation state", {"not_mutated", "mutated"}),
                     1)},
                {"area",
                 withDescription(
                     McpSchema::object(
                         {
                             {"x1", McpSchema::number("X coordinate of one corner")},
                             {"y1", McpSchema::number("Y coordinate of one corner")},
                             {"x2", McpSchema::number("X coordinate of the opposite corner")},
                             {"y2", McpSchema::number("Y coordinate of the opposite corner")},
                         },
                         {"x1", "y1", "x2", "y2"}),
                     "Creatures with at least one cell inside this rectangle")},
                {"genome_names", McpSchema::array("Genome names", McpSchema::string("Genome name"), 1)},
                {"has_node", nodeFilterSchema("Genomes containing a node that matches this node filter")},
                {"has_gene", geneFilterSchema("Genomes containing a gene that matches this gene filter")},
                {"similar_to_object_id",
                 McpSchema::id(
                     "Genomes with the same structure as the genome of the creature of this cell: the same number of genes and nodes and at each node "
                     "position the same cell type and whether there is a constructor")},
                {"max_node_differences", McpSchema::integer("Number of node positions that may differ for similar_to_object_id. Default: 0", 0)},
                {"conditions", conditionsSchema("genome")},
            }),
            "Selects the creatures and thereby their genomes. All given criteria must hold. Without 'where' all creatures in the scope are selected.");
    }

    std::vector<std::string> getOperationNames()
    {
        std::vector<std::string> result;
        for (auto const& [name, keys] : OperationKeys) {
            result.emplace_back(name);
        }
        return result;
    }

    boost::json::object editSchema()
    {
        return McpSchema::object(
            {
                {"level", McpSchema::enumeration("Level the edit works on", {"genome", "gene", "node"})},
                {"where",
                 McpSchema::jsonObject(
                     "Restricts the edit: for the level 'gene' a gene filter as in 'where.has_gene', for the level 'node' a node filter as in "
                     "'where.has_node'. Default: all genes or nodes")},
                {"op", McpSchema::enumeration("Operation, see the tool description", getOperationNames())},
                {"path",
                 McpSchema::string(
                     "Path relative to the level, e.g. 'cellType.sensor.maxRange' for a node or 'mutationRates.addNodeMutation.nodeProbability' for a "
                     "genome")},
                {"value", McpSchema::any("For 'set' the new value, for 'add' the number to add and for 'patch' an object in the JSON format of the level")},
                {"factor", McpSchema::number("For 'scale'")},
                {"min", McpSchema::number("For 'clamp' and 'randomize'")},
                {"max", McpSchema::number("For 'clamp' and 'randomize'")},
                {"values", McpSchema::array("For 'randomize': values to choose from instead of 'min' and 'max'", McpSchema::any("Value"), 1)},
                {"per",
                 McpSchema::enumeration(
                     "For 'randomize' and 'jitter': draw a random number for each value (default) or once per node, gene or genome",
                     {"value", "node", "gene", "genome"})},
                {"sigma", McpSchema::number("For 'jitter': standard deviation of the added noise", 0)},
                {"map",
                 McpSchema::array(
                     "For 'remap': replacements, e.g. [{\"from\": 2, \"to\": 5}]",
                     McpSchema::object({{"from", McpSchema::any("Value to replace")}, {"to", McpSchema::any("New value")}}, {"from", "to"}),
                     1)},
                {"node", McpSchema::jsonObject("For 'insert_node': the node in JSON format, omitted fields take their default values")},
                {"gene", McpSchema::jsonObject("For 'insert_gene' and 'replace_gene': the gene in JSON format, omitted fields take their default values")},
                {"from",
                 withDescription(
                     McpSchema::object({
                         {"genome", McpSchema::jsonObject("Template genome in JSON format")},
                         {"genome_file_path", McpSchema::string("Absolute path of a template genome file (*.genome)")},
                         {"genome_of_object_id", McpSchema::id("Id of a cell whose creature's genome is the template")},
                         {"gene_index", McpSchema::integer("Gene of the template. Default: 0", 0)},
                         {"node_index", McpSchema::integer("Node of the template gene for 'insert_node'. Default: 0", 0)},
                     }),
                     "Instead of 'node' or 'gene': takes the node or gene from a template genome given by exactly one of 'genome', 'genome_file_path' "
                     "and 'genome_of_object_id'")},
                {"index",
                 McpSchema::integer(
                     "For 'insert_node' and 'insert_gene': position of the inserted node or gene, at least 1 for genes. Default: at the end", 0)},
            },
            {"level", "op"});
    }

    boost::json::object fieldSchema()
    {
        return McpSchema::object(
            {
                {"level", McpSchema::enumeration("Level of the field", {"genome", "gene", "node"})},
                {"where", McpSchema::jsonObject("For the level 'gene' a gene filter, for the level 'node' a node filter, as in mass_edit_genomes")},
                {"path", McpSchema::string("Path relative to the level, e.g. 'cellType.sensor.maxRange' for a node")},
            },
            {"level", "path"});
    }
}

std::vector<McpTool> McpGenomeEditTools::getTools(McpToolContext& context)
{
    _context = &context;

    auto restrictToSelectionSchema = McpSchema::boolean(
        "Only consider the creatures in the selection including the connected cell networks. Otherwise all creatures in the world are considered. "
        "Default: true");

    return {
        McpTool{
            .name = "mass_edit_genomes",
            .description =
                std::string("Edits the genomes of many creatures at once. 'where' selects the creatures and thereby their genomes, 'edits' lists the "
                            "changes, which are applied in order to each selected genome. Each edit works on a level: 'genome' (fields like "
                            "'applyMetaMutations' or 'mutationRates'), 'gene' (fields like 'stiffness' or 'shape') or 'node' (fields like 'color', "
                            "'cellType', 'constructor' or 'neuralNetwork'). ")
                + PathConventions
                + " A path that does not exist in a gene or node, e.g. 'cellType.sensor.maxRange' in a muscle node, is skipped.\n"
                  "Operations: 'patch' merges 'value' into the genome, gene or node, fields not given stay unchanged and another alternative, e.g. "
                  "another cell type, replaces the current one. 'set' sets the value at 'path' to 'value'. 'add' adds the number 'value'. 'scale' "
                  "multiplies by 'factor'. 'clamp' limits to 'min' and/or 'max'. 'randomize' draws a number between 'min' and 'max' or one of "
                  "'values'. 'jitter' adds normally distributed noise with the standard deviation 'sigma'. 'remap' replaces values according to 'map'. "
                  "Integer fields stay integers. 'insert_node' (level 'gene') inserts 'node' at 'index'. 'remove_nodes' (level 'node') removes the "
                  "selected nodes. 'insert_gene' (level 'genome') inserts 'gene' at 'index'. The gene references (geneIndex) to genes behind it are "
                  "shifted, the references inside the inserted gene are taken over unchanged. 'replace_gene' (level 'gene') replaces the selected "
                  "genes. 'remove_genes' (level 'gene') removes the selected genes and the constructors building them, but at least one gene "
                  "must remain. Instead of 'node' or 'gene', 'from' takes a node or gene from a template genome.\n"
                  "The modified genomes are validated and corrected. Genomes that are shared with creatures not selected by 'where' are copied first, "
                  "so that only the selected creatures change. Cells that are already constructed keep their properties, the changes take effect in "
                  "cells constructed afterwards, e.g. in offspring. Inserting or removing genes and nodes renumbers them, while cells that are "
                  "already built keep their gene and node indices, so a construction in progress may continue differently. Run with dry_run first "
                  "to see what would change. With restrict_to_selection, the "
                  "selection including the connected cell networks is replaced by the modified objects as in apply_mass_operations, otherwise the "
                  "whole world is modified, which resets the selection.",
            .inputSchema = McpSchema::object(
                {
                    {"restrict_to_selection", restrictToSelectionSchema},
                    {"where", whereSchema()},
                    {"edits", McpSchema::array("Edits applied in this order", editSchema(), 1)},
                    {"dry_run", McpSchema::boolean("Only report what would change without changing anything. Default: false")},
                    {"max_genomes", McpSchema::integer("Change nothing if more genomes would be modified", 0)},
                },
                {"edits"}),
            .handler = [this](boost::json::object const& arguments) { return massEditGenomes(arguments); },
        },
        McpTool{
            .name = "summarize_genomes",
            .description =
                std::string("Summarizes the genomes of the creatures selected by 'where', which accepts the same filters as in mass_edit_genomes: the "
                            "numbers of creatures, genomes and lineages, the largest lineages, the numbers of genes and nodes, the cell types and "
                            "colors of the nodes and statistics of the values at the given 'fields'. By default each genome counts once in these "
                            "statistics, regardless of how many creatures share it, which describes the genetic variety. With count_per_creature each "
                            "creature counts, which describes shares of the population. Useful to explore a population and to check a filter before "
                            "mass_edit_genomes. ")
                + PathConventions,
            .inputSchema = McpSchema::object({
                {"restrict_to_selection", restrictToSelectionSchema},
                {"where", whereSchema()},
                {"fields", McpSchema::array("Fields whose values are evaluated", fieldSchema(), 1)},
                {"count_per_creature",
                 McpSchema::boolean("Count each creature instead of each genome in the statistics of genes, nodes and fields. Default: false")},
            }),
            .handler = [this](boost::json::object const& arguments) { return summarizeGenomes(arguments); },
        },
    };
}

namespace
{
    void checkKeys(boost::json::object const& object, std::vector<std::string> const& allowedKeys)
    {
        for (auto const& [key, value] : object) {
            if (!std::ranges::contains(allowedKeys, std::string(key))) {
                throw std::invalid_argument(
                    std::format("Unknown argument '{}', allowed are: {}.", std::string(key), StringHelper::formatEnumeration(allowedKeys)));
            }
        }
    }

    template <typename Func>
    auto withContext(std::string const& context, Func const& func)
    {
        try {
            return func();
        } catch (std::invalid_argument const& exception) {
            throw std::invalid_argument(std::format("{}: {}", context, exception.what()));
        }
    }

    boost::json::object const& getObject(boost::json::object const& arguments, std::string_view key)
    {
        auto const& value = McpArguments::getValue(arguments, key);
        if (!value.is_object()) {
            throw std::invalid_argument(std::format("'{}' must be an object.", key));
        }
        return value.as_object();
    }

    std::optional<std::vector<std::string>> getOptionalStrings(boost::json::object const& arguments, std::string_view key)
    {
        if (!arguments.contains(key)) {
            return std::nullopt;
        }
        auto const& value = McpArguments::getValue(arguments, key);
        auto invalidStrings = std::invalid_argument(std::format("'{}' must be a non-empty array of strings.", key));
        if (!value.is_array() || value.as_array().empty()) {
            throw invalidStrings;
        }
        std::vector<std::string> result;
        for (auto const& element : value.as_array()) {
            if (!element.is_string()) {
                throw invalidStrings;
            }
            result.emplace_back(element.as_string());
        }
        return result;
    }

    double getNumber(boost::json::object const& arguments, std::string_view key)
    {
        auto const& value = McpArguments::getValue(arguments, key);
        if (!value.is_number()) {
            throw std::invalid_argument(std::format("'{}' must be a number.", key));
        }
        return value.to_number<double>();
    }

    std::optional<double> getOptionalNumber(boost::json::object const& arguments, std::string_view key)
    {
        if (!arguments.contains(key)) {
            return std::nullopt;
        }
        return getNumber(arguments, key);
    }

    JsonCondition parseCondition(boost::json::object const& object)
    {
        std::vector<std::string> allowedKeys = {"path"};
        for (auto const& [name, op] : ConditionOperators) {
            allowedKeys.emplace_back(name);
        }
        checkKeys(object, allowedKeys);

        JsonCondition result{.path = JsonPath(McpArguments::getString(object, "path"))};
        auto numOperators = 0;
        for (auto const& [name, op] : ConditionOperators) {
            if (auto value = object.if_contains(name)) {
                result.op = op;
                result.value = *value;
                ++numOperators;
            }
        }
        if (numOperators != 1) {
            throw std::invalid_argument("A condition needs exactly one of 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'in' and 'exists'.");
        }
        return result;
    }

    std::vector<JsonCondition> parseConditions(boost::json::object const& filter)
    {
        std::vector<JsonCondition> result;
        if (filter.contains("conditions")) {
            for (auto const& object : McpArguments::getObjects(filter, "conditions", 1)) {
                result.emplace_back(parseCondition(object));
            }
        }
        return result;
    }

    GenomeNodeFilter parseNodeFilter(boost::json::object const& object);

    GenomeGeneFilter parseGeneFilter(boost::json::object const& object)
    {
        checkKeys(object, {"indices", "names", "min_nodes", "max_nodes", "has_node", "conditions"});
        GenomeGeneFilter result;
        result.indices = McpArguments::getOptionalInts(object, "indices", 1, 0);
        result.names = getOptionalStrings(object, "names");
        result.minNodes = McpArguments::getOptionalInt(object, "min_nodes", 0);
        result.maxNodes = McpArguments::getOptionalInt(object, "max_nodes", 0);
        if (object.contains("has_node")) {
            result.hasNode = std::make_shared<GenomeNodeFilter const>(parseNodeFilter(getObject(object, "has_node")));
        }
        result.conditions = parseConditions(object);
        return result;
    }

    GenomeNodeFilter parseNodeFilter(boost::json::object const& object)
    {
        checkKeys(object, {"gene", "indices", "cell_types", "colors", "has_constructor", "conditions"});
        GenomeNodeFilter result;
        if (object.contains("gene")) {
            result.gene = parseGeneFilter(getObject(object, "gene"));
        }
        result.indices = McpArguments::getOptionalInts(object, "indices", 1, 0);
        result.cellTypeNames = getOptionalStrings(object, "cell_types");
        result.colors = McpArguments::getOptionalInts(object, "colors", 1, 0, MAX_COLORS - 1);
        result.hasConstructor = McpArguments::getOptionalBool(object, "has_constructor");
        result.conditions = parseConditions(object);
        return result;
    }

    MutationState parseMutationState(std::string const& name)
    {
        if (name == "not_mutated") {
            return MutationState_NotMutated;
        }
        if (name == "mutated") {
            return MutationState_Mutated;
        }
        throw std::invalid_argument(std::format("Unknown mutation state '{}', allowed are: not_mutated and mutated.", name));
    }

    GenomeFilter parseFilter(boost::json::object const& arguments)
    {
        if (!arguments.contains("where")) {
            return {};
        }
        auto const& where = getObject(arguments, "where");
        checkKeys(
            where,
            {"creature_ids",
             "object_ids",
             "genome_ids",
             "lineage_ids",
             "cell_colors",
             "node_colors",
             "min_cells",
             "max_cells",
             "min_generation",
             "max_generation",
             "mutation_states",
             "area",
             "genome_names",
             "has_node",
             "has_gene",
             "similar_to_object_id",
             "max_node_differences",
             "conditions"});

        GenomeFilter result;
        if (where.contains("creature_ids")) {
            result.creatureIds = McpArguments::getIds(where, "creature_ids", 1);
        }
        if (where.contains("object_ids")) {
            result.objectIds = McpArguments::getIds(where, "object_ids", 1);
        }
        if (where.contains("genome_ids")) {
            result.genomeIds = McpArguments::getIds(where, "genome_ids", 1);
        }
        result.lineageIds = McpArguments::getOptionalInts(where, "lineage_ids", 1);
        result.cellColors = McpArguments::getOptionalInts(where, "cell_colors", 1, 0, MAX_COLORS - 1);
        result.nodeColors = McpArguments::getOptionalInts(where, "node_colors", 1, 0, MAX_COLORS - 1);
        result.minCells = McpArguments::getOptionalInt(where, "min_cells", 0);
        result.maxCells = McpArguments::getOptionalInt(where, "max_cells", 0);
        result.minGeneration = McpArguments::getOptionalInt(where, "min_generation", 0);
        result.maxGeneration = McpArguments::getOptionalInt(where, "max_generation", 0);
        if (auto mutationStates = getOptionalStrings(where, "mutation_states")) {
            result.mutationStates.emplace();
            for (auto const& mutationState : *mutationStates) {
                result.mutationStates->emplace_back(parseMutationState(mutationState));
            }
        }
        if (where.contains("area")) {
            auto const& area = getObject(where, "area");
            checkKeys(area, {"x1", "y1", "x2", "y2"});
            auto x1 = McpArguments::getFloat(area, "x1");
            auto y1 = McpArguments::getFloat(area, "y1");
            auto x2 = McpArguments::getFloat(area, "x2");
            auto y2 = McpArguments::getFloat(area, "y2");
            result.area = RealRect{.topLeft = {std::min(x1, x2), std::min(y1, y2)}, .bottomRight = {std::max(x1, x2), std::max(y1, y2)}};
        }
        result.genomeNames = getOptionalStrings(where, "genome_names");
        if (where.contains("has_node")) {
            result.hasNode = withContext("has_node", [&] { return parseNodeFilter(getObject(where, "has_node")); });
        }
        if (where.contains("has_gene")) {
            result.hasGene = withContext("has_gene", [&] { return parseGeneFilter(getObject(where, "has_gene")); });
        }
        if (auto objectId = McpArguments::getOptionalId(where, "similar_to_object_id")) {
            result.similarTo = McpGenomeTools::get().getGenomeOfObject(*objectId);
        }
        if (where.contains("max_node_differences")) {
            if (!result.similarTo) {
                throw std::invalid_argument("'max_node_differences' requires 'similar_to_object_id'.");
            }
            result.maxNodeDifferences = McpArguments::getInt(where, "max_node_differences", 0);
        }
        result.conditions = parseConditions(where);
        return result;
    }

    std::string getLevelName(GenomeEditLevel level)
    {
        switch (level) {
        case GenomeEditLevel::Gene:
            return "gene";
        case GenomeEditLevel::Node:
            return "node";
        default:
            return "genome";
        }
    }

    GenomeEditLevel parseLevel(boost::json::object const& object)
    {
        auto name = McpArguments::getString(object, "level");
        for (auto level : {GenomeEditLevel::Genome, GenomeEditLevel::Gene, GenomeEditLevel::Node}) {
            if (name == getLevelName(level)) {
                return level;
            }
        }
        throw std::invalid_argument(std::format("Unknown level '{}', allowed are: genome, gene and node.", name));
    }

    GenomeEditGranularity parseGranularity(boost::json::object const& object)
    {
        auto name = McpArguments::getOptionalString(object, "per").value_or("value");
        if (name == "value") {
            return GenomeEditGranularity::Value;
        }
        if (name == "node") {
            return GenomeEditGranularity::Node;
        }
        if (name == "gene") {
            return GenomeEditGranularity::Gene;
        }
        if (name == "genome") {
            return GenomeEditGranularity::Genome;
        }
        throw std::invalid_argument(std::format("Unknown value '{}' of 'per', allowed are: value, node, gene and genome.", name));
    }

    void parseLevelFilter(boost::json::object const& object, GenomeEditLevel level, GenomeGeneFilter& geneFilter, GenomeNodeFilter& nodeFilter)
    {
        if (!object.contains("where")) {
            return;
        }
        auto const& where = getObject(object, "where");
        switch (level) {
        case GenomeEditLevel::Genome:
            throw std::invalid_argument("The level 'genome' has no 'where', whole genomes are selected by the 'where' of the tool.");
        case GenomeEditLevel::Gene:
            geneFilter = withContext("where", [&] { return parseGeneFilter(where); });
            break;
        case GenomeEditLevel::Node:
            nodeFilter = withContext("where", [&] { return parseNodeFilter(where); });
            break;
        }
    }

    GenomeDesc getTemplateGenome(boost::json::object const& edit, std::vector<std::string> allowedKeys)
    {
        auto const& from = getObject(edit, "from");
        allowedKeys.insert(allowedKeys.end(), {"genome", "genome_file_path", "genome_of_object_id", "gene_index"});
        checkKeys(from, allowedKeys);
        return McpGenomeTools::get().getGenomeFromArguments(from);
    }

    GeneDesc getTemplateGene(GenomeDesc const& genome, boost::json::object const& from)
    {
        auto geneIndex = McpArguments::getOptionalInt(from, "gene_index", 0).value_or(0);
        if (geneIndex >= toInt(genome._genes.size())) {
            throw std::invalid_argument(std::format("The template genome has no gene {}.", geneIndex));
        }
        return genome._genes.at(geneIndex);
    }

    GeneDesc parseGene(boost::json::object const& edit)
    {
        if (edit.contains("gene") == edit.contains("from")) {
            throw std::invalid_argument("Give exactly one of 'gene' and 'from'.");
        }
        if (edit.contains("gene")) {
            GeneDesc result;
            SerializerService::get().deserializeFromJson(result, getObject(edit, "gene"));
            return result;
        }
        return getTemplateGene(getTemplateGenome(edit, {}), getObject(edit, "from"));
    }

    NodeDesc parseNode(boost::json::object const& edit)
    {
        if (edit.contains("node") == edit.contains("from")) {
            throw std::invalid_argument("Give exactly one of 'node' and 'from'.");
        }
        if (edit.contains("node")) {
            NodeDesc result;
            SerializerService::get().deserializeFromJson(result, getObject(edit, "node"));
            return result;
        }
        auto gene = getTemplateGene(getTemplateGenome(edit, {"node_index"}), getObject(edit, "from"));
        auto nodeIndex = McpArguments::getOptionalInt(getObject(edit, "from"), "node_index", 0).value_or(0);
        if (nodeIndex >= toInt(gene._nodes.size())) {
            throw std::invalid_argument(std::format("The template gene has no node {}.", nodeIndex));
        }
        return gene._nodes.at(nodeIndex);
    }

    GenomeEditOperation parseOperation(boost::json::object const& object, std::string const& op)
    {
        auto getPath = [&] { return JsonPath(McpArguments::getString(object, "path")); };
        if (op == "patch") {
            return GenomePatchOperation{.patch = getObject(object, "value")};
        }
        if (op == "set") {
            return GenomeSetOperation{.path = getPath(), .value = McpArguments::getValue(object, "value")};
        }
        if (op == "add") {
            return GenomeAddOperation{.path = getPath(), .summand = getNumber(object, "value")};
        }
        if (op == "scale") {
            return GenomeScaleOperation{.path = getPath(), .factor = getNumber(object, "factor")};
        }
        if (op == "clamp") {
            return GenomeClampOperation{.path = getPath(), .min = getOptionalNumber(object, "min"), .max = getOptionalNumber(object, "max")};
        }
        if (op == "randomize") {
            GenomeRandomizeOperation result{
                .path = getPath(), .min = getOptionalNumber(object, "min"), .max = getOptionalNumber(object, "max"), .granularity = parseGranularity(object)};
            if (object.contains("values")) {
                auto const& values = McpArguments::getValue(object, "values");
                if (!values.is_array() || values.as_array().empty()) {
                    throw std::invalid_argument("'values' must be a non-empty array.");
                }
                result.values.assign(values.as_array().begin(), values.as_array().end());
            }
            return result;
        }
        if (op == "jitter") {
            return GenomeJitterOperation{.path = getPath(), .sigma = getNumber(object, "sigma"), .granularity = parseGranularity(object)};
        }
        if (op == "remap") {
            GenomeRemapOperation result{.path = getPath()};
            for (auto const& entry : McpArguments::getObjects(object, "map", 1)) {
                checkKeys(entry, {"from", "to"});
                result.entries.emplace_back(GenomeRemapEntry{.from = McpArguments::getValue(entry, "from"), .to = McpArguments::getValue(entry, "to")});
            }
            return result;
        }
        if (op == "insert_node") {
            return GenomeInsertNodeOperation{.node = parseNode(object), .index = McpArguments::getOptionalInt(object, "index", 0)};
        }
        if (op == "remove_nodes") {
            return GenomeRemoveNodesOperation();
        }
        if (op == "insert_gene") {
            return GenomeInsertGeneOperation{.gene = parseGene(object), .index = McpArguments::getOptionalInt(object, "index", 0)};
        }
        if (op == "replace_gene") {
            return GenomeReplaceGeneOperation{.gene = parseGene(object)};
        }
        return GenomeRemoveGenesOperation();
    }

    struct ParsedEdit
    {
        GenomeEdit edit;
        std::string description;
    };

    ParsedEdit parseEdit(boost::json::object const& object)
    {
        auto level = parseLevel(object);
        auto op = McpArguments::getString(object, "op");
        auto operationKeys = std::ranges::find(OperationKeys, op, &std::pair<std::string, std::vector<std::string>>::first);
        if (operationKeys == OperationKeys.end()) {
            throw std::invalid_argument(std::format("Unknown operation '{}', allowed are: {}.", op, StringHelper::formatEnumeration(getOperationNames())));
        }
        std::vector<std::string> allowedKeys = {"level", "where", "op"};
        allowedKeys.insert(allowedKeys.end(), operationKeys->second.begin(), operationKeys->second.end());
        checkKeys(object, allowedKeys);

        ParsedEdit result;
        result.edit.level = level;
        parseLevelFilter(object, level, result.edit.geneFilter, result.edit.nodeFilter);
        result.edit.operation = parseOperation(object, op);
        auto path = object.contains("path") ? std::format(" '{}'", McpArguments::getString(object, "path")) : std::string();
        result.description = std::format("{} {}{}", getLevelName(level), op, path);
        return result;
    }

    struct ParsedField
    {
        GenomeFieldQuery field;
        std::string description;
    };

    ParsedField parseField(boost::json::object const& object)
    {
        checkKeys(object, {"level", "where", "path"});
        ParsedField result;
        result.field.level = parseLevel(object);
        parseLevelFilter(object, result.field.level, result.field.geneFilter, result.field.nodeFilter);
        auto path = McpArguments::getString(object, "path");
        result.field.path = JsonPath(path);
        result.description = std::format("{} '{}'", getLevelName(result.field.level), path);
        return result;
    }

    std::string formatCount(int count, std::string const& noun)
    {
        return std::format("{} {}{}", StringHelper::format(static_cast<uint64_t>(std::max(0, count))), noun, count == 1 ? "" : "s");
    }

    std::string formatValue(boost::json::value const& value)
    {
        return StringHelper::truncate(McpJson::serialize(value), MaxValueTextLength, 1);
    }

    std::string formatChanges(GenomeChanges const& changes)
    {
        std::string result;
        for (auto const& change : changes.firstChanges) {
            if (change.arraySizes) {
                result += std::format("\n  {}: {} -> {} entries", change.path, change.arraySizes->first, change.arraySizes->second);
            } else if (!change.before) {
                result += std::format("\n  {}: added {}", change.path, formatValue(change.after.value()));
            } else if (!change.after) {
                result += std::format("\n  {}: removed {}", change.path, formatValue(*change.before));
            } else {
                result += std::format("\n  {}: {} -> {}", change.path, formatValue(*change.before), formatValue(*change.after));
            }
        }
        auto numMoreChanges = changes.numChanges - toInt(changes.firstChanges.size());
        if (numMoreChanges > 0) {
            result += std::format("\n  ... and {} more", formatCount(numMoreChanges, "change"));
        }
        return result;
    }

    std::string describeEditStatistics(GenomeEdit const& edit, GenomeEditStatistics const& statistics)
    {
        auto targets = formatCount(statistics.numTargets, getLevelName(edit.level));
        auto inGenomes = edit.level == GenomeEditLevel::Genome ? std::string() : " in " + formatCount(statistics.numGenomesWithTargets, "genome");
        if (statistics.numTargets == 0) {
            return std::format("no matching {}s.", getLevelName(edit.level));
        }
        return std::visit(
            [&]<typename Operation>(Operation const&) -> std::string {
                if constexpr (std::is_same_v<Operation, GenomePatchOperation>) {
                    return std::format(
                        "{} patched{}, {} of them changed.", targets, inGenomes, StringHelper::format(static_cast<uint64_t>(statistics.numChangedValues)));
                } else if constexpr (std::is_same_v<Operation, GenomeInsertNodeOperation>) {
                    return std::format("a node inserted into {}{}.", targets, inGenomes);
                } else if constexpr (std::is_same_v<Operation, GenomeInsertGeneOperation>) {
                    return std::format("a gene inserted into {}.", targets);
                } else if constexpr (std::is_same_v<Operation, GenomeReplaceGeneOperation>) {
                    return std::format("{} replaced{}.", targets, inGenomes);
                } else if constexpr (std::is_same_v<Operation, GenomeRemoveNodesOperation> || std::is_same_v<Operation, GenomeRemoveGenesOperation>) {
                    return std::format("{} removed{}.", targets, inGenomes);
                } else {
                    auto result = std::format(
                        "{}{} matched, {} found at the path, {} changed.",
                        targets,
                        inGenomes,
                        formatCount(statistics.numValues, "value"),
                        StringHelper::format(static_cast<uint64_t>(statistics.numChangedValues)));
                    if (statistics.numValues == 0) {
                        result += " The path does not exist in the matching " + getLevelName(edit.level) + "s.";
                    }
                    if (statistics.numNonNumericValues > 0) {
                        result += std::format(" {} skipped because they are not numbers.", formatCount(statistics.numNonNumericValues, "value"));
                    }
                    return result;
                }
            },
            edit.operation);
    }

    std::string formatEditResult(
        GenomeEditResult const& result,
        std::vector<ParsedEdit> const& parsedEdits,
        bool dryRun,
        bool restrictToSelection,
        std::optional<int> maxGenomes)
    {
        auto scope = restrictToSelection ? "selection" : "whole world";
        auto matching = std::format(
            "{} with {} in the {} match the filter.", formatCount(result.numCreatures, "creature"), formatCount(result.numGenomes, "genome"), scope);
        std::string text;
        if (dryRun) {
            text = std::format("Dry run, nothing was changed. {} The edits would modify {}.", matching, formatCount(result.numModifiedGenomes, "genome"));
            if (result.numCopiedGenomes > 0) {
                text += std::format(" {} would be copied first since creatures outside the filter share them.", formatCount(result.numCopiedGenomes, "genome"));
            }
            if (maxGenomes && result.numModifiedGenomes > *maxGenomes) {
                text += std::format(" This exceeds max_genomes ({}).", *maxGenomes);
            }
        } else if (result.numModifiedGenomes == 0) {
            text = std::format("No genome was changed. {}", matching);
        } else {
            text = std::format("Modified {}. {}", formatCount(result.numModifiedGenomes, "genome"), matching);
            if (result.numCopiedGenomes > 0) {
                text += std::format(" {} were copied first since creatures outside the filter share them.", formatCount(result.numCopiedGenomes, "genome"));
            }
            if (restrictToSelection) {
                text += " The modified objects are selected.";
            }
        }

        for (auto const& [editIndex, parsedEdit, statistics] : std::views::zip(std::views::iota(0), parsedEdits, result.editStatistics)) {
            text += std::format("\nedits[{}] ({}): {}", editIndex, parsedEdit.description, describeEditStatistics(parsedEdit.edit, statistics));
        }
        if (result.numCorrectedGenomes > 0) {
            text += std::format(
                "\nThe validation corrected {}, e.g.:{}", formatCount(result.numCorrectedGenomes, "genome"), formatChanges(result.exampleCorrections));
        }
        if (result.numGenomesWithNewIssues > 0) {
            text += std::format(
                "\n{} {} new genome issues as shown in the genome editor, e.g. {}. Check whether this is intended.",
                formatCount(result.numGenomesWithNewIssues, "modified genome"),
                result.numGenomesWithNewIssues == 1 ? "has" : "have",
                result.exampleIssue.value_or("unknown"));
        }
        if (result.exampleGenomeId) {
            if (!dryRun && restrictToSelection) {
                text += "\nChanges of one genome:";
            } else {
                text += dryRun ? std::string("\nChanges of the genome") : std::format("\nChanges of genome {}", *result.exampleGenomeId);
                if (result.exampleCellId) {
                    text += std::format(" of the creature with the cell {}", *result.exampleCellId);
                }
                text += ":";
            }
            text += formatChanges(result.exampleChanges);
        }
        return text;
    }
}

McpToolResult McpGenomeEditTools::massEditGenomes(boost::json::object const& arguments) const
{
    auto restrictToSelection = McpArguments::getOptionalBool(arguments, "restrict_to_selection").value_or(true);
    auto dryRun = McpArguments::getOptionalBool(arguments, "dry_run").value_or(false);
    auto maxGenomes = McpArguments::getOptionalInt(arguments, "max_genomes", 0);
    auto filter = withContext("where", [&] { return parseFilter(arguments); });
    std::vector<ParsedEdit> parsedEdits;
    auto editObjects = McpArguments::getObjects(arguments, "edits", 1);
    for (auto const& [editIndex, object] : editObjects | std::views::enumerate) {
        parsedEdits.emplace_back(withContext(std::format("edits[{}]", editIndex), [&] { return parseEdit(object); }));
    }
    std::vector<GenomeEdit> edits;
    for (auto const& parsedEdit : parsedEdits) {
        edits.emplace_back(parsedEdit.edit);
    }

    auto pausedSimulation = !dryRun && _SimulationFacade::get()->isSimulationRunning();
    if (pausedSimulation) {
        _SimulationFacade::get()->pauseSimulation();
    }
    ExitScopeGuard resumeSimulation([&] {
        if (pausedSimulation) {
            _SimulationFacade::get()->runSimulation();
        }
    });

    auto content = getContent(restrictToSelection);
    auto result = GenomeMassEditService::get().edit(content, filter, edits);
    if (!dryRun && maxGenomes && result.numModifiedGenomes > *maxGenomes) {
        throw std::invalid_argument(std::format(
            "The edits would modify {}, more than max_genomes ({}). Nothing was changed.", formatCount(result.numModifiedGenomes, "genome"), *maxGenomes));
    }
    if (!dryRun && result.numModifiedGenomes > 0) {
        if (restrictToSelection) {
            replaceSelection(std::move(content));
        } else {
            replaceWorldContent(content);
        }
        _context->onSelectionChanged();
        _context->showMessage(std::format("{} edited", formatCount(result.numModifiedGenomes, "genome")));
    }
    return {.text = formatEditResult(result, parsedEdits, dryRun, restrictToSelection, maxGenomes)};
}

namespace
{
    std::string formatNumber(double value)
    {
        return McpJson::serialize(boost::json::value(value));
    }

    std::string formatMean(double value)
    {
        return std::format("{:.4g}", value);
    }

    std::string formatCountStatistics(GenomeCountStatistics const& statistics)
    {
        return std::format("{} to {}, mean {}", statistics.min, statistics.max, formatMean(statistics.mean));
    }

    std::string formatFieldStatistics(GenomeFieldStatistics const& statistics)
    {
        if (statistics.numValues == 0) {
            return "no values, the path does not exist in the matching genomes, genes or nodes.";
        }
        auto result = formatCount(statistics.numValues, "value");
        if (statistics.numNumbers > 0) {
            result += std::format(
                ", {} from {} to {}, mean {}",
                formatCount(statistics.numNumbers, "number"),
                formatNumber(statistics.min),
                formatNumber(statistics.max),
                formatMean(statistics.mean));
        }
        result += ".";
        if (!statistics.frequentValues.empty()) {
            std::vector<std::string> frequentValues;
            for (auto const& [value, count] : statistics.frequentValues) {
                frequentValues.emplace_back(std::format("{} ({})", formatValue(value), StringHelper::format(static_cast<uint64_t>(count))));
            }
            result += " Most frequent: " + boost::algorithm::join(frequentValues, ", ") + ".";
        }
        return result;
    }

    template <typename Key, typename Formatter>
    std::string formatFrequencies(std::vector<std::pair<Key, int>> const& frequencies, Formatter const& formatEntry)
    {
        std::vector<std::string> entries;
        for (auto const& [key, count] : frequencies) {
            entries.emplace_back(formatEntry(key, StringHelper::format(static_cast<uint64_t>(count))));
        }
        return entries.empty() ? std::string("none") : boost::algorithm::join(entries, ", ");
    }

    std::string formatSummary(GenomeSummary const& summary, std::vector<ParsedField> const& parsedFields, bool restrictToSelection, bool countPerCreature)
    {
        auto text = std::format(
            "{} with {} in {} match the filter in the {}. {}",
            formatCount(summary.numCreatures, "creature"),
            formatCount(summary.numGenomes, "genome"),
            formatCount(summary.numLineages, "lineage"),
            restrictToSelection ? "selection" : "whole world",
            countPerCreature ? "Each creature counts once in the following statistics, i.e. a genome counts as often as creatures share it."
                             : "Each genome counts once in the following statistics.");
        if (summary.numGenomes == 0) {
            return text;
        }
        text += "\nLargest lineages: " + formatFrequencies(summary.largestLineages, [](int lineageId, std::string const& count) {
                    return std::format("{} ({} creatures)", lineageId, count);
                });
        text += "\nGenes per genome: " + formatCountStatistics(summary.genesPerGenome);
        text += "\nNodes per genome: " + formatCountStatistics(summary.nodesPerGenome);
        text += "\nNodes per cell type: " + formatFrequencies(summary.numNodesByCellType, [](std::string const& cellType, std::string const& count) {
                    return std::format("{} {}", cellType, count);
                });
        text += "\nNodes per color: "
            + formatFrequencies(summary.numNodesByColor, [](int color, std::string const& count) { return std::format("color {}: {}", color, count); });
        for (auto const& [fieldIndex, parsedField, statistics] : std::views::zip(std::views::iota(0), parsedFields, summary.fields)) {
            text += std::format("\nfields[{}] ({}): {}", fieldIndex, parsedField.description, formatFieldStatistics(statistics));
        }
        if (!summary.exampleCellIds.empty()) {
            std::vector<std::string> cellIds;
            for (auto cellId : summary.exampleCellIds) {
                cellIds.emplace_back(std::to_string(cellId));
            }
            text += "\nCells of example creatures with different genomes, e.g. for get_genome: " + boost::algorithm::join(cellIds, ", ");
        }
        return text;
    }
}

McpToolResult McpGenomeEditTools::summarizeGenomes(boost::json::object const& arguments) const
{
    auto restrictToSelection = McpArguments::getOptionalBool(arguments, "restrict_to_selection").value_or(true);
    auto countPerCreature = McpArguments::getOptionalBool(arguments, "count_per_creature").value_or(false);
    auto filter = withContext("where", [&] { return parseFilter(arguments); });
    std::vector<ParsedField> parsedFields;
    if (arguments.contains("fields")) {
        auto fieldObjects = McpArguments::getObjects(arguments, "fields", 1);
        for (auto const& [fieldIndex, object] : fieldObjects | std::views::enumerate) {
            parsedFields.emplace_back(withContext(std::format("fields[{}]", fieldIndex), [&] { return parseField(object); }));
        }
    }
    std::vector<GenomeFieldQuery> fields;
    for (auto const& parsedField : parsedFields) {
        fields.emplace_back(parsedField.field);
    }

    auto content = getContent(restrictToSelection);
    auto summary = GenomeMassEditService::get().summarize(content, filter, fields, countPerCreature);
    return {.text = formatSummary(summary, parsedFields, restrictToSelection, countPerCreature)};
}

ContentDesc McpGenomeEditTools::getContent(bool restrictToSelection) const
{
    if (!restrictToSelection) {
        return _SimulationFacade::get()->getSimulationData();
    }
    auto selection = _SimulationFacade::get()->getSelectionShallowData();
    if (selection.numObjects == 0) {
        throw std::invalid_argument("Nothing is selected. Select creatures with select_area or select_at first, or set restrict_to_selection to false.");
    }
    return _SimulationFacade::get()->getSelectedSimulationData(true);
}

void McpGenomeEditTools::replaceSelection(ContentDesc&& content) const
{
    _SimulationFacade::get()->removeSelectedObjects(true);
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
}

void McpGenomeEditTools::replaceWorldContent(ContentDesc const& content) const
{
    _SimulationFacade::get()->setSimulationData(content);
}
