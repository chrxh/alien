#include "GenomeIssue.h"

#include <ranges>

std::vector<GenomeIssue> GenomeIssue::filterByGene(std::vector<GenomeIssue> const& issues, int geneIndex)
{
    auto result = issues | std::views::filter([&](auto const& issue) { return issue.geneIndex == geneIndex; });
    return std::vector(result.begin(), result.end());
}

std::set<int> GenomeIssue::getRemovedGeneIndices(std::vector<GenomeIssue> const& issues)
{
    std::set<int> result;
    for (auto const& issue : issues) {
        if (issue.removesGene) {
            result.insert(issue.geneIndex);
        }
    }
    return result;
}
