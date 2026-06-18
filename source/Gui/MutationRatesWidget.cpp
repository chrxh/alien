#include "MutationRatesWidget.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include <Base/StringHelper.h>
#include <EngineInterface/GenomeDesc.h>

#include "AlienGui.h"
#include "MutationRatesDialog.h"
#include "StyleRepository.h"

namespace
{
    void
    addActiveMutationType(std::vector<std::pair<std::string, std::string>>& result, std::string const& name, std::initializer_list<float> nodeProbabilities)
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

    std::vector<std::pair<std::string, std::string>> getActiveMutationTypes(MutationRatesDesc const& mutationRates)
    {
        std::vector<std::pair<std::string, std::string>> activeMutations;
        addActiveMutationType(
            activeMutations, "Connection mutations", {mutationRates._connectionMutations[0]._nodeProbability, mutationRates._connectionMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations, "Neuron mutations", {mutationRates._neuronMutations[0]._nodeProbability, mutationRates._neuronMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Cell type property mut.",
            {mutationRates._cellTypePropertiesMutations[0]._nodeProbability, mutationRates._cellTypePropertiesMutations[1]._nodeProbability});
        addActiveMutationType(activeMutations, "Cell type mode mut.", {mutationRates._cellTypeModeMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Cell type mutations", {mutationRates._cellTypeMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Void mutations", {mutationRates._voidMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Append node mutations", {mutationRates._appendNodeMutation._geneProbability});
        addActiveMutationType(activeMutations, "Add node mutations", {mutationRates._addNodeMutation._geneProbability});
        addActiveMutationType(activeMutations, "Trim node mutations", {mutationRates._trimNodeMutation._geneProbability});
        addActiveMutationType(activeMutations, "Delete node mutations", {mutationRates._deleteNodeMutation._geneProbability});
        addActiveMutationType(
            activeMutations, "Constructor", {mutationRates._constructorMutations[0]._nodeProbability, mutationRates._constructorMutations[1]._nodeProbability});
        return activeMutations;
    }
}

void MutationRatesWidget::process(MutationRatesDesc& mutationRates, float rightColumnWidth, bool nested)
{
    ImGui::BeginGroup();

    if (AlienGui::Button(AlienGui::ButtonParameters().buttonText("Edit").name("Click to edit").textWidth(rightColumnWidth))) {
        auto onAdopt = [&mutationRates](MutationRatesDesc const& adoptedRates) { mutationRates = adoptedRates; };
        if (nested) {
            MutationRatesDialog::get().openNested(mutationRates, onAdopt);
        } else {
            MutationRatesDialog::get().open(mutationRates, onAdopt);
        }
    }

    for (auto const& [name, probabilities] : getActiveMutationTypes(mutationRates)) {
        auto value = probabilities;
        AlienGui::InputText(AlienGui::InputTextParameters().name(name).readOnly(true).textWidth(rightColumnWidth), value);
    }
    ImGui::EndGroup();
}
