#include "MutationRatesWidget.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include <Base/StringHelper.h>

#include <Data/GenomeDesc.h>

#include "AlienGui.h"
#include "EntityAttributeHelp.h"
#include "StyleService.h"

namespace
{
    struct ActiveMutationType
    {
        std::string name;
        std::string probabilities;
    };

    void addActiveMutationType(std::vector<ActiveMutationType>& result, std::string const& name, std::initializer_list<float> nodeProbabilities)
    {
        std::string probabilities;
        for (auto const& nodeProbability : nodeProbabilities) {
            if (nodeProbability > 0.0f) {
                if (!probabilities.empty()) {
                    probabilities += ", ";
                }
                probabilities += StringHelper::format(nodeProbability, 5);
            }
        }
        if (!probabilities.empty()) {
            result.emplace_back(name, probabilities);
        }
    }

    std::vector<ActiveMutationType> getActiveMutationTypes(MutationRatesDesc const& mutationRates)
    {
        std::vector<ActiveMutationType> activeMutations;
        addActiveMutationType(
            activeMutations,
            "Connection mutations",
            {mutationRates._connectionMutations[0]._nodeProbability, mutationRates._connectionMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations, "Neuron mutations", {mutationRates._neuronMutations[0]._nodeProbability, mutationRates._neuronMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Cell type property mut.",
            {mutationRates._cellTypePropertiesMutations[0]._nodeProbability, mutationRates._cellTypePropertiesMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Geometry mutations",
            {mutationRates._geometryMutations[0]._geneProbability, mutationRates._geometryMutations[1]._geneProbability});
        addActiveMutationType(activeMutations, "Cell type mode mut.", {mutationRates._cellTypeModeMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Cell type mutations", {mutationRates._cellTypeMutation._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Customization mutations",
            {mutationRates._customizationMutation._genomeProbability, mutationRates._customizationMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Void mutations", {mutationRates._voidMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Extend gene mutations", {mutationRates._extendGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, "Add node mutations", {mutationRates._addNodeMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Trim gene mutations", {mutationRates._trimGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, "Delete node mutations", {mutationRates._deleteNodeMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Add gene mutations", {mutationRates._addGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, "Duplicate gene mutations", {mutationRates._duplicateGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, "Delete gene mutations", {mutationRates._deleteGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, "Swap gene mutations", {mutationRates._swapGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, "Copy node section mutations", {mutationRates._copyNodeSectionMutation._geneProbability});
        addActiveMutationType(activeMutations, "Move node section mutations", {mutationRates._moveNodeSectionMutation._geneProbability});
        addActiveMutationType(activeMutations, "Roll out gene mutations", {mutationRates._rollOutGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, "Compact gene mutations", {mutationRates._compactGeneMutation._geneProbability});
        addActiveMutationType(
            activeMutations, "Constructor", {mutationRates._constructorMutations[0]._nodeProbability, mutationRates._constructorMutations[1]._nodeProbability});
        return activeMutations;
    }
}

void MutationRatesWidget::process(MutationRatesDesc& mutationRates, float rightColumnWidth, bool disabled)
{
    auto activeMutationTypes = getActiveMutationTypes(mutationRates);
    auto tooltip = EntityAttributeHelp::get(EntityAttribute::GenomeMutationRatesSummary);
    if (!activeMutationTypes.empty()) {
        tooltip += "\n\nActive mutation types:";
        for (auto const& [name, probabilities] : activeMutationTypes) {
            tooltip += "\n" + name + ": " + probabilities;
        }
    }

    ImGui::BeginDisabled(disabled);
    if (AlienGui::Button(AlienGui::ButtonParameters()
                             .buttonText("Edit (" + std::to_string(activeMutationTypes.size()) + " active)")
                             .name("Mutation rates")
                             .textWidth(rightColumnWidth)
                             .tooltip(tooltip))) {
        _dialog.open(mutationRates, [&mutationRates](MutationRatesDesc const& adoptedRates) { mutationRates = adoptedRates; });
    }
    ImGui::EndDisabled();

    // Processed outside of the disabled scope, since BeginDisabled() also applies to popups begun inside of it
    _dialog.process();
}
