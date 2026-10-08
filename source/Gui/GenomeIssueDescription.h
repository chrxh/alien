#pragma once

#include <optional>
#include <string>
#include <vector>

#include <imgui.h>

#include <Data/GenomeDesc.h>
#include <Data/GenomeIssue.h>

class GenomeIssueDescription
{
public:
    static char const* getIcon(GenomeIssue const& issue);
    static ImColor getColor(GenomeIssue const& issue);
    static std::optional<GenomeIssue> findMostRelevantIssue(std::vector<GenomeIssue> const& issues);
    static std::string getTooltip(std::vector<GenomeIssue> const& issues, GenomeDesc const& genome);
};
