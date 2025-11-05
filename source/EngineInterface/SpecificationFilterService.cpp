#include "SpecificationFilterService.h"

#include <algorithm>

ParametersSpec SpecificationFilterService::filter(ParametersSpec const& spec, ParametersFilter const& filter) const
{
    if (!filter.containedText.has_value() || filter.containedText->empty()) {
        return spec;
    }

    ParametersSpec result;

    for (auto const& groupSpec : spec._groups) {
        // Check if the group name matches the filter
        if (matchesFilter(groupSpec._name, filter)) {
            // If group name matches, include all parameters in the group
            result._groups.push_back(groupSpec);
        } else {
            // Otherwise, filter individual parameters
            ParameterGroupSpec filteredGroup;
            filteredGroup._name = groupSpec._name;
            filteredGroup._description = groupSpec._description;
            filteredGroup._expertToggle = groupSpec._expertToggle;

            for (auto const& parameterSpec : groupSpec._parameters) {
                auto filteredParam = filterParameterSpec(parameterSpec, filter);
                if (filteredParam._visible) {
                    filteredGroup._parameters.push_back(filteredParam);
                }
            }

            // Only add the group if it has any parameters left
            if (!filteredGroup._parameters.empty()) {
                result._groups.push_back(filteredGroup);
            }
        }
    }

    return result;
}

bool SpecificationFilterService::matchesFilter(std::string const& name, ParametersFilter const& filter) const
{
    if (!filter.containedText.has_value() || filter.containedText->empty()) {
        return true;
    }

    return name.find(*filter.containedText) != std::string::npos;
}

ParameterSpec SpecificationFilterService::filterParameterSpec(ParameterSpec const& spec, ParametersFilter const& filter) const
{
    ParameterSpec result = spec;

    // Check if this parameter contains an AlternativeSpec
    if (std::holds_alternative<AlternativeSpec>(spec._reference)) {
        auto const& alternativeSpec = std::get<AlternativeSpec>(spec._reference);
        AlternativeSpec filteredAlternativeSpec;
        filteredAlternativeSpec._member = alternativeSpec._member;

        // Filter each alternative's parameters
        for (auto const& [alternativeName, alternativeParams] : alternativeSpec._alternatives) {
            auto filteredParams = filterAlternativeSpecs(alternativeParams, filter);
            // Only add alternatives that have visible parameters
            if (!filteredParams.empty()) {
                filteredAlternativeSpec._alternatives.push_back({alternativeName, filteredParams});
            }
        }

        result._reference = filteredAlternativeSpec;
    }

    // If the parameter name matches, keep it visible
    if (matchesFilter(spec._name, filter)) {
        result._visible = true;
    } else {
        // Check if any nested parameters (in alternatives) match
        bool hasMatchingNested = false;
        if (std::holds_alternative<AlternativeSpec>(result._reference)) {
            auto const& filteredAlternativeSpec = std::get<AlternativeSpec>(result._reference);
            // If we have any alternatives with parameters, then some nested parameter matched
            hasMatchingNested = !filteredAlternativeSpec._alternatives.empty();
        }
        result._visible = hasMatchingNested;
    }

    return result;
}

std::vector<ParameterSpec> SpecificationFilterService::filterAlternativeSpecs(std::vector<ParameterSpec> const& specs, ParametersFilter const& filter) const
{
    std::vector<ParameterSpec> result;

    for (auto const& spec : specs) {
        auto filteredSpec = filterParameterSpec(spec, filter);
        if (filteredSpec._visible) {
            result.push_back(filteredSpec);
        }
    }

    return result;
}
