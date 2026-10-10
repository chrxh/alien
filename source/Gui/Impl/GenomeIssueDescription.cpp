#include "GenomeIssueDescription.h"

#include <algorithm>
#include <ranges>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Interface/Definitions.h>
#include <Base/Interface/StringHelper.h>

#include "StyleService.h"

char const* GenomeIssueDescription::getIcon(GenomeIssue const& issue)
{
    return issue.isFollowUp() ? ICON_FA_LONG_ARROW_ALT_RIGHT : ICON_FA_EXCLAMATION_TRIANGLE;
}

ImColor GenomeIssueDescription::getColor(GenomeIssue const& issue)
{
    return issue.isFollowUp() ? Const::TextDecentColor : Const::WarningColor;
}

std::optional<GenomeIssue> GenomeIssueDescription::findMostRelevantIssue(std::vector<GenomeIssue> const& issues)
{
    if (issues.empty()) {
        return std::nullopt;
    }
    return *std::ranges::max_element(issues, {}, [](GenomeIssue const& issue) { return !issue.isFollowUp(); });
}

namespace
{
    struct IssueText
    {
        std::string title;
        std::string problem;
        std::string explanation;
        std::string consequence;
    };

    std::string toGeneList(std::vector<int> const& geneIndices)
    {
        auto numbers = geneIndices | std::views::transform([](int geneIndex) { return std::to_string(geneIndex); });
        return (geneIndices.size() == 1 ? "gene " : "genes ") + StringHelper::formatEnumeration(std::vector(numbers.begin(), numbers.end()));
    }

    std::string toNodeList(std::vector<int> const& nodeIndices)
    {
        return (nodeIndices.size() == 1 ? "node " : "nodes ") + StringHelper::formatRanges(nodeIndices);
    }

    std::vector<int> getVoidNodeIndices(GeneDesc const& gene)
    {
        std::vector<int> result;
        for (auto const& [nodeIndex, node] : gene._nodes | boost::adaptors::indexed(0)) {
            if (node.getCellType() == CellType_Void) {
                result.emplace_back(toInt(nodeIndex));
            }
        }
        return result;
    }

    std::vector<int> getNodeIndicesBecomingVoid(GenomeIssue const& issue, GeneDesc const& gene)
    {
        auto result = issue.cutOffNodeIndices | std::views::filter([&](int nodeIndex) { return gene._nodes.at(nodeIndex).getCellType() != CellType_Void; });
        return std::vector(result.begin(), result.end());
    }

    std::string getVoidBoundaryNodeProblem(GenomeIssue const& issue)
    {
        return issue.nodeIndex == 0 ? "The first node is void." : "The last node is void.";
    }

    std::string getMetaMutationNote(GenomeDesc const& genome)
    {
        return genome._applyMetaMutations ? " Meta-mutations may change the structural mutation rates, which decide whether the node gets a random "
                                            "cell type or the gene is removed."
                                          : "";
    }

    IssueText getNodesCutOffText(GenomeIssue const& issue, GenomeDesc const& genome)
    {
        auto const& gene = genome._genes.at(issue.geneIndex);
        auto nodeIndicesBecomingVoid = getNodeIndicesBecomingVoid(issue, gene);
        auto isSingleNode = nodeIndicesBecomingVoid.size() == 1;

        IssueText result{
            .title = "Nodes cut off from last node",
            .explanation = "A construction stays attached to its constructor at the last node of a gene. Void cells die right after their construction, "
                           "so nodes that are connected to the last node only via void nodes are lost. If the first node is among them, the whole "
                           "gene is lost."};
        if (issue.nodeIndex.has_value()) {
            result.problem = getVoidBoundaryNodeProblem(issue);
            result.explanation += getMetaMutationNote(genome);
        } else if (nodeIndicesBecomingVoid.empty()) {
            result.problem =
                "Void " + toNodeList(issue.cutOffNodeIndices) + (issue.cutOffNodeIndices.size() == 1 ? " is" : " are") + " cut off from the last node.";
        } else {
            auto voidNodeIndices = getVoidNodeIndices(gene);
            result.problem = "Void " + toNodeList(voidNodeIndices) + (voidNodeIndices.size() == 1 ? " cuts " : " cut ") + toNodeList(nodeIndicesBecomingVoid)
                + " off from the last node.";
        }

        if (issue.removesGene) {
            result.consequence = "The gene is removed.";
        } else if (nodeIndicesBecomingVoid.empty()) {
            result.consequence = "Their constructors are turned off.";
        } else {
            result.consequence =
                (isSingleNode ? "Node " : "Nodes ") + StringHelper::formatRanges(nodeIndicesBecomingVoid) + (isSingleNode ? " becomes void." : " become void.");
        }
        return result;
    }

    IssueText getIssueText(GenomeIssue const& issue, GenomeDesc const& genome)
    {
        switch (issue.type) {
        case GenomeIssueType::VoidBoundaryNode:
            return IssueText{
                .title = "Void node at gene boundary",
                .problem = getVoidBoundaryNodeProblem(issue),
                .explanation = "The first and the last node of a gene must not be void." + getMetaMutationNote(genome),
                .consequence = "The node gets a random cell type."};
        case GenomeIssueType::NodesCutOffFromLastNode:
            return getNodesCutOffText(issue, genome);
        case GenomeIssueType::CycleAvoidingRootGene: {
            auto genes = issue.cycleGeneIndices | std::views::transform([](int geneIndex) { return "gene " + std::to_string(geneIndex); });
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
                .problem = issue.isFollowUp() ? "Becomes unreachable from the root gene." : "Not reachable from the root gene.",
                .explanation = issue.isFollowUp() ? "Another correction removes the last chain of constructors that leads from the root gene to this gene."
                                                  : "No chain of constructors leads from the root gene to this gene, so it is never constructed.",
                .consequence = "The gene is removed."};
        case GenomeIssueType::TooManyGenesWithSeparation: {
            auto const& constructor = genome._genes.at(issue.geneIndex)._nodes.at(issue.nodeIndex.value())._constructor;
            auto targetGene = constructor.has_value() ? std::to_string(constructor->_geneIndex) : std::string("?");
            return IssueText{
                .title = "Too many genes with separation",
                .problem = "Constructs gene " + targetGene + " with separation in addition to " + toGeneList(issue.geneIndicesKeepingSeparation) + ".",
                .explanation = "At most " + std::to_string(Const::MaxGenesWithSeparation)
                    + " different genes can be constructed with separation. They are counted in construction order starting at the root gene.",
                .consequence = "Separation is turned off, so gene " + targetGene + " becomes part of the same creature."};
        }
        case GenomeIssueType::ConstructsRemovedGene:
            return IssueText{
                .title = "Constructs a removed gene",
                .problem = "Constructs gene " + std::to_string(issue.referencedGeneIndex.value_or(0)) + ", which is removed.",
                .explanation = "The constructed gene is removed because of another warning.",
                .consequence = "The constructor is turned off."};
        case GenomeIssueType::InjectsRemovedGene:
            return IssueText{
                .title = "Injects a removed gene",
                .problem = "Injects gene " + std::to_string(issue.referencedGeneIndex.value_or(0)) + ", which is removed.",
                .explanation = "The injected gene is removed because of another warning.",
                .consequence = "The injector uses the first remaining gene instead."};
        default:
            return IssueText();
        }
    }

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
