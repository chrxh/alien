#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <boost/json/value.hpp>

#include <Base/MathTypes.h>
#include <Base/Singleton.h>

#include <Data/Descs.h>
#include <Data/GenomeDesc.h>

#include "JsonPath.h"

// Empty filters match everything. Conditions refer to the JSON format of the filtered level without its child levels.
struct GenomeNodeFilter;

struct GenomeGeneFilter
{
    std::optional<std::vector<int>> indices;
    std::optional<std::vector<std::string>> names;
    std::shared_ptr<GenomeNodeFilter const> hasNode;
    std::vector<JsonCondition> conditions;
};

struct GenomeNodeFilter
{
    GenomeGeneFilter gene;
    std::optional<std::vector<int>> indices;
    std::optional<std::vector<std::string>> cellTypeNames;
    std::optional<std::vector<int>> colors;
    std::optional<bool> hasConstructor;
    std::vector<JsonCondition> conditions;
};

struct GenomeFilter
{
    std::optional<std::vector<uint64_t>> creatureIds;
    std::optional<std::vector<uint64_t>> objectIds;
    std::optional<std::vector<uint64_t>> genomeIds;
    std::optional<std::vector<int>> lineageIds;
    std::optional<std::vector<int>> cellColors;
    std::optional<std::vector<int>> nodeColors;
    std::optional<int> minCells;
    std::optional<int> maxCells;
    std::optional<int> minGeneration;
    std::optional<int> maxGeneration;
    std::optional<std::vector<MutationState>> mutationStates;
    std::optional<RealRect> area;
    std::optional<std::vector<std::string>> genomeNames;
    std::optional<GenomeNodeFilter> hasNode;
    std::optional<GenomeGeneFilter> hasGene;
    std::optional<GenomeDesc> similarTo;
    int maxNodeDifferences = 0;
    std::vector<JsonCondition> conditions;
};

enum class GenomeEditLevel
{
    Genome,
    Gene,
    Node
};

enum class GenomeEditGranularity
{
    Value,
    Node,
    Gene,
    Genome
};

struct GenomePatchOperation
{
    boost::json::value patch;  // An object in the JSON format of the level
};

struct GenomeSetOperation
{
    JsonPath path;
    boost::json::value value;
};

struct GenomeAddOperation
{
    JsonPath path;
    double summand = 0;
};

struct GenomeScaleOperation
{
    JsonPath path;
    double factor = 1;
};

struct GenomeClampOperation
{
    JsonPath path;
    std::optional<double> min;
    std::optional<double> max;
};

struct GenomeRandomizeOperation
{
    JsonPath path;
    std::optional<double> min;
    std::optional<double> max;
    std::vector<boost::json::value> values;
    GenomeEditGranularity granularity = GenomeEditGranularity::Value;
};

struct GenomeJitterOperation
{
    JsonPath path;
    double sigma = 0;
    GenomeEditGranularity granularity = GenomeEditGranularity::Value;
};

struct GenomeRemapEntry
{
    boost::json::value from;
    boost::json::value to;
};

struct GenomeRemapOperation
{
    JsonPath path;
    std::vector<GenomeRemapEntry> entries;
};

struct GenomeInsertNodeOperation
{
    NodeDesc node;
    std::optional<int> index;  // std::nullopt = append
};

struct GenomeRemoveNodesOperation
{};

struct GenomeInsertGeneOperation
{
    GeneDesc gene;
    std::optional<int> index;  // std::nullopt = append
};

struct GenomeReplaceGeneOperation
{
    GeneDesc gene;
};

struct GenomeRemoveGenesOperation
{};

using GenomeEditOperation = std::variant<
    GenomePatchOperation,
    GenomeSetOperation,
    GenomeAddOperation,
    GenomeScaleOperation,
    GenomeClampOperation,
    GenomeRandomizeOperation,
    GenomeJitterOperation,
    GenomeRemapOperation,
    GenomeInsertNodeOperation,
    GenomeRemoveNodesOperation,
    GenomeInsertGeneOperation,
    GenomeReplaceGeneOperation,
    GenomeRemoveGenesOperation>;

struct GenomeEdit
{
    GenomeEditLevel level = GenomeEditLevel::Genome;
    GenomeGeneFilter geneFilter;
    GenomeNodeFilter nodeFilter;
    GenomeEditOperation operation;
};

struct GenomeEditStatistics
{
    int numGenomesWithTargets = 0;
    int numTargets = 0;
    int numValues = 0;
    int numChangedValues = 0;
    int numNonNumericValues = 0;
};

struct GenomeChange
{
    std::string path;
    std::optional<boost::json::value> before;
    std::optional<boost::json::value> after;
    std::optional<std::pair<int, int>> arraySizes;
};

struct GenomeChanges
{
    std::vector<GenomeChange> firstChanges;
    int numChanges = 0;
};

struct GenomeEditResult
{
    int numCreatures = 0;
    int numGenomes = 0;
    int numModifiedGenomes = 0;
    int numCopiedGenomes = 0;
    int numCorrectedGenomes = 0;
    int numGenomesWithNewIssues = 0;
    std::vector<GenomeEditStatistics> editStatistics;

    std::optional<uint64_t> exampleGenomeId;
    std::optional<uint64_t> exampleCellId;
    GenomeChanges exampleChanges;
    GenomeChanges exampleCorrections;
    std::optional<std::string> exampleIssue;
};

struct GenomeFieldQuery
{
    GenomeEditLevel level = GenomeEditLevel::Genome;
    GenomeGeneFilter geneFilter;
    GenomeNodeFilter nodeFilter;
    JsonPath path;
};

struct GenomeFieldStatistics
{
    int numValues = 0;
    int numNumbers = 0;
    double min = 0;
    double max = 0;
    double mean = 0;
    std::vector<std::pair<boost::json::value, int>> frequentValues;  // Most frequent first, empty if there are too many different values
};

struct GenomeCountStatistics
{
    int min = 0;
    int max = 0;
    double mean = 0;
};

struct GenomeSummary
{
    int numCreatures = 0;
    int numGenomes = 0;
    int numLineages = 0;
    std::vector<std::pair<int, int>> largestLineages;  // Lineage id and number of creatures
    GenomeCountStatistics genesPerGenome;
    GenomeCountStatistics nodesPerGenome;
    std::vector<std::pair<std::string, int>> numNodesByCellType;
    std::vector<std::pair<int, int>> numNodesByColor;
    std::vector<GenomeFieldStatistics> fields;
    std::vector<uint64_t> exampleCellIds;
};

// Selects genomes via their creatures and edits them in the JSON format of genomes. Genomes shared by matching and non-matching creatures
// are copied before editing, the example genome id refers to the copy. Invalid arguments throw std::invalid_argument and leave the content
// unchanged, as do edits that would remove all genes of a genome.
class GenomeMassEditService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GenomeMassEditService);

public:
    std::vector<std::string> const& getCellTypeNames() const;

    GenomeEditResult edit(ContentDesc& content, GenomeFilter const& filter, std::vector<GenomeEdit> const& edits) const;
    GenomeSummary summarize(ContentDesc const& content, GenomeFilter const& filter, std::vector<GenomeFieldQuery> const& fields) const;

private:
    GenomeMassEditService();

    boost::json::value const& getFormat(GenomeEditLevel level) const;
    void checkFilter(GenomeFilter const& filter) const;
    void checkGeneFilter(GenomeGeneFilter const& filter) const;
    void checkNodeFilter(GenomeNodeFilter const& filter) const;
    void checkEdit(GenomeEdit const& edit) const;
    void checkFieldQuery(GenomeFieldQuery const& field) const;

    boost::json::value _genomeFormat;
    boost::json::value _geneFormat;
    boost::json::value _nodeFormat;
    std::vector<std::string> _cellTypeNames;
};
