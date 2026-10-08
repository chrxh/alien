#pragma once

#include <optional>
#include <string>
#include <vector>

#include <imgui.h>

#include <EngineInterface/GenomeValidationService.h>

// Presentation of the issues found by GenomeValidationService in the genome editor
class GenomeIssueDescription
{
public:
    static char const* getIcon(GenomeIssue const& issue);
    static ImColor getColor(GenomeIssue const& issue);

    // The issue whose icon and color represent an element affected by several issues
    static std::optional<GenomeIssue> findMarkerIssue(std::vector<GenomeIssue> const& issues);

    static std::string getTooltip(std::vector<GenomeIssue> const& issues, GenomeDesc const& genome);
};
