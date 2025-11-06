#include "SpecificationFilterService.h"

#include <algorithm>

#include <Base/StringHelper.h>

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

    return StringHelper::containsCaseInsensitive(name, *filter.containedText);
}

bool SpecificationFilterService::anyParameterMatchesRecursively(ParameterSpec const& spec, ParametersFilter const& filter) const
{
    // Check if this parameter's name matches
    if (matchesFilter(spec._name, filter)) {
        return true;
    }

    // If it contains an AlternativeSpec, check nested parameters
    if (std::holds_alternative<AlternativeSpec>(spec._reference)) {
        auto const& alternativeSpec = std::get<AlternativeSpec>(spec._reference);
        for (auto const& [_, alternativeParams] : alternativeSpec._alternatives) {
            for (auto const& param : alternativeParams) {
                if (anyParameterMatchesRecursively(param, filter)) {
                    return true;
                }
            }
        }
    }

    return false;
}

ParameterSpec SpecificationFilterService::filterParameterSpec(ParameterSpec const& spec, ParametersFilter const& filter) const
{
    ParameterSpec result = spec;

    // If the parameter name matches, keep it visible with filtered alternatives
    if (matchesFilter(spec._name, filter)) {
        result._visible = true;
        
        // If this is an AlternativeSpec, filter its nested parameters
        if (std::holds_alternative<AlternativeSpec>(spec._reference)) {
            auto const& alternativeSpec = std::get<AlternativeSpec>(spec._reference);
            AlternativeSpec filteredAltSpec = alternativeSpec;
            filteredAltSpec._alternatives.clear();
            
            // Filter each alternative
            for (auto const& [name, alternativeParams] : alternativeSpec._alternatives) {
                std::vector<ParameterSpec> filteredParams = filterAlternativeSpecs(alternativeParams, filter);
                // Always include the alternative, even if empty
                filteredAltSpec._alternatives.push_back({name, filteredParams});
            }
            
            result._reference = filteredAltSpec;
        }
        
        return result;
    }

    // Check if this parameter contains an AlternativeSpec
    if (std::holds_alternative<AlternativeSpec>(spec._reference)) {
        auto const& alternativeSpec = std::get<AlternativeSpec>(spec._reference);
        AlternativeSpec filteredAltSpec = alternativeSpec;
        filteredAltSpec._alternatives.clear();
        
        bool anyParameterMatches = false;
        
        // Filter each alternative
        for (auto const& [name, alternativeParams] : alternativeSpec._alternatives) {
            std::vector<ParameterSpec> filteredParams = filterAlternativeSpecs(alternativeParams, filter);
            
            // Check if any parameters matched in this alternative
            if (!filteredParams.empty()) {
                anyParameterMatches = true;
            }
            
            // Always include the alternative, even if empty
            filteredAltSpec._alternatives.push_back({name, filteredParams});
        }

        // If any parameter matches, include the AlternativeSpec with filtered alternatives
        if (anyParameterMatches) {
            result._visible = true;
            result._reference = filteredAltSpec;
        } else {
            result._visible = false;
        }
    } else {
        result._visible = false;
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
