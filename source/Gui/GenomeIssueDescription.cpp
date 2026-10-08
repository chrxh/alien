#include "GenomeIssueDescription.h"

#include <algorithm>
#include <ranges>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Definitions.h>

#include "StyleService.h"

namespace
{
    struct IssueText
    {
        std::string title;
        std::string problem;
        std::string explanation;
        std::string consequence;
    };

    std::string toEnumeration(std::vector<std::string> const& items)
    {
        if (items.size() <= 1) {
            return items.empty() ? std::string() : items.front();
        }
        auto leadingItems = std::vector(items.begin(), items.end() - 1);
        return boost::algorithm::join(leadingItems, ", ") + " and " + items.back();
    }

    std::string toGeneList(std::vector<int> const& geneIndices)
    {
        auto numbers = geneIndices | std::views::transform([](int geneIndex) { return std::to_string(geneIndex); });
        return (geneIndices.size() == 1 ? "gene " : "genes ") + toEnumeration(std::vector(numbers.begin(), numbers.end()));
    }

    // Consecutive indices are combined into ranges, e.g. "0-2, 5"
    std::string toIndexRanges(std::vector<int> const& indices)
    {
        std::vector<std::pair<int, int>> ranges;
        for (auto index : indices) {
            if (!ranges.empty() && ranges.back().second + 1 == index) {
                ranges.back().second = index;
            } else {
                ranges.emplace_back(index, index);
            }
        }
        auto parts = ranges | std::views::transform([](auto const& range) {
                         return range.first == range.second ? std::to_string(range.first) : std::to_string(range.first) + "-" + std::to_string(range.second);
                     });
        return boost::algorithm::join(std::vector(parts.begin(), parts.end()), ", ");
    }

    std::string toNodeList(std::vector<int> const& nodeIndices)
    {
        return (nodeIndices.size() == 1 ? "node " : "nodes ") + toIndexRanges(nodeIndices);
    }

    std::string getVoidBoundaryNodeProblem(GenomeIssue const& issue)
    {
        return issue.nodeIndex == 0 ? "The first node is void." : "The last node is void.";
    }

    std::string getNodesCutOffProblem(GenomeIssue const& issue, GenomeDesc const& genome)
    {
        if (issue.nodeIndex.has_value()) {
            return getVoidBoundaryNodeProblem(issue);
        }
        std::vector<int> voidNodeIndices;
        for (auto const& [nodeIndex, node] : genome._genes.at(issue.geneIndex)._nodes | boost::adaptors::indexed(0)) {
            if (node.getCellType() == CellType_Void) {
                voidNodeIndices.emplace_back(toInt(nodeIndex));
            }
        }
        return "Void " + toNodeList(voidNodeIndices) + (voidNodeIndices.size() == 1 ? " cuts " : " cut ") + toNodeList(issue.voidedNodeIndices)
            + " off from the last node.";
    }

    IssueText getIssueText(GenomeIssue const& issue, GenomeDesc const& genome)
    {
        switch (issue.type) {
        case GenomeIssueType::VoidBoundaryNode:
            return IssueText{
                .title = "Void node at gene boundary",
                .problem = getVoidBoundaryNodeProblem(issue),
                .explanation = "The first and the last node of a gene must not be void.",
                .consequence = "The node gets a random cell type."};
        case GenomeIssueType::NodesCutOffFromLastNode: {
            auto isSingleNode = issue.voidedNodeIndices.size() == 1;
            return IssueText{
                .title = "Nodes cut off from last node",
                .problem = getNodesCutOffProblem(issue, genome),
                .explanation = "A construction stays attached to its constructor at the last node of a gene. Void cells die right after their "
                               "construction, so nodes that are connected to the last node only via void nodes are lost. If the first node is among "
                               "them, the whole gene is lost.",
                .consequence = issue.removesGene
                    ? "The gene is removed."
                    : (isSingleNode ? "Node " : "Nodes ") + toIndexRanges(issue.voidedNodeIndices) + (isSingleNode ? " becomes void." : " become void.")};
        }
        case GenomeIssueType::CycleAvoidingRootGene: {
            auto genes = issue.relatedGeneIndices | std::views::transform([](int geneIndex) { return "gene " + std::to_string(geneIndex); });
            return IssueText{
                .title = "Cycle avoiding the root gene",
                .problem = "Closes the cycle " + boost::algorithm::join(std::vector(genes.begin(), genes.end()), " " ICON_FA_LONG_ARROW_ALT_RIGHT " ") + ".",
                .explanation = "Constructors may only form cycles through the root gene, since constructing the root gene starts a new creature. A "
                               "cycle avoiding the root gene would construct cells endlessly within the same creature.",
                .consequence = "The constructor is turned off."};
        }
        case GenomeIssueType::GeneUnreachableFromRoot:
            return IssueText{
                .title = "Gene unreachable from root",
                .problem = issue.causeIssueIndex.has_value() ? "Becomes unreachable from the root gene." : "Not reachable from the root gene.",
                .explanation = issue.causeIssueIndex.has_value()
                    ? "Another correction removes the last chain of constructors that leads from the root gene to this gene."
                    : "No chain of constructors leads from the root gene to this gene, so it is never constructed.",
                .consequence = "The gene is removed."};
        case GenomeIssueType::TooManyGenesWithSeparation: {
            auto const& constructor = genome._genes.at(issue.geneIndex)._nodes.at(issue.nodeIndex.value())._constructor;
            auto targetGene = constructor.has_value() ? std::to_string(constructor->_geneIndex) : std::string("?");
            return IssueText{
                .title = "Too many genes with separation",
                .problem = "Constructs gene " + targetGene + " with separation in addition to " + toGeneList(issue.relatedGeneIndices) + ".",
                .explanation = "At most 2 different genes can be constructed with separation. They are counted in construction order starting at the "
                               "root gene.",
                .consequence = "Separation is turned off, so gene " + targetGene + " becomes part of the same creature."};
        }
        case GenomeIssueType::ConstructsRemovedGene:
            return IssueText{
                .title = "Constructs a removed gene",
                .problem = "Constructs " + toGeneList(issue.relatedGeneIndices) + ", which is removed.",
                .explanation = "The constructed gene is removed because of another issue.",
                .consequence = "The constructor is turned off."};
        case GenomeIssueType::InjectsRemovedGene:
            return IssueText{
                .title = "Injects a removed gene",
                .problem = "Injects " + toGeneList(issue.relatedGeneIndices) + ", which is removed.",
                .explanation = "The injected gene is removed because of another issue.",
                .consequence = "The injector uses the first remaining gene instead."};
        default:
            return IssueText();
        }
    }
}

char const* GenomeIssueDescription::getIcon(GenomeIssue const& issue)
{
    if (issue.causeIssueIndex.has_value()) {
        return ICON_FA_LONG_ARROW_ALT_RIGHT;
    }
    return issue.getSeverity() == GenomeIssueSeverity::Error ? ICON_FA_TIMES_CIRCLE : ICON_FA_EXCLAMATION_TRIANGLE;
}

ImColor GenomeIssueDescription::getColor(GenomeIssue const& issue)
{
    if (issue.causeIssueIndex.has_value()) {
        return Const::TextDecentColor;
    }
    return issue.getSeverity() == GenomeIssueSeverity::Error ? Const::DangerColor : Const::WarningColor;
}

namespace
{
    int getMarkerPriority(GenomeIssue const& issue)
    {
        if (issue.causeIssueIndex.has_value()) {
            return 0;
        }
        return issue.getSeverity() == GenomeIssueSeverity::Error ? 2 : 1;
    }
}

std::optional<GenomeIssue> GenomeIssueDescription::findMarkerIssue(std::vector<GenomeIssue> const& issues)
{
    if (issues.empty()) {
        return std::nullopt;
    }
    return *std::ranges::max_element(issues, {}, getMarkerPriority);
}

namespace
{
    std::string getLocation(GenomeIssue const& issue)
    {
        auto result = "gene " + std::to_string(issue.geneIndex);
        if (issue.nodeIndex.has_value()) {
            result += ", node " + std::to_string(issue.nodeIndex.value());
        }
        return result;
    }

    std::string getIssueTooltip(GenomeIssue const& issue, GenomeDesc const& genome)
    {
        auto text = getIssueText(issue, genome);
        return std::string(GenomeIssueDescription::getIcon(issue)) + " " + text.title + " (" + getLocation(issue) + ")\n" + text.problem + " "
            + text.explanation + "\nCorrection in the genome of offspring: " + text.consequence;
    }
}

std::string GenomeIssueDescription::getTooltip(std::vector<GenomeIssue> const& issues, GenomeDesc const& genome)
{
    auto tooltips = issues | std::views::transform([&](auto const& issue) { return getIssueTooltip(issue, genome); });
    return boost::algorithm::join(std::vector(tooltips.begin(), tooltips.end()), "\n\n");
}
