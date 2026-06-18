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
            activeMutations, "Connection", {mutationRates._connectionMutations[0]._nodeProbability, mutationRates._connectionMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations, "Neuron", {mutationRates._neuronMutations[0]._nodeProbability, mutationRates._neuronMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Cell type property",
            {mutationRates._cellTypePropertiesMutations[0]._nodeProbability, mutationRates._cellTypePropertiesMutations[1]._nodeProbability});
        addActiveMutationType(activeMutations, "Cell type mode", {mutationRates._cellTypeModeMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Cell type", {mutationRates._cellTypeMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Void", {mutationRates._voidMutation._nodeProbability});
        addActiveMutationType(activeMutations, "Append node", {mutationRates._appendNodeMutation._geneProbability});
        addActiveMutationType(activeMutations, "Add node", {mutationRates._addNodeMutation._geneProbability});
        addActiveMutationType(activeMutations, "Trim node", {mutationRates._trimNodeMutation._geneProbability});
        addActiveMutationType(activeMutations, "Delete node", {mutationRates._deleteNodeMutation._geneProbability});
        addActiveMutationType(
            activeMutations, "Constructor", {mutationRates._constructorMutations[0]._nodeProbability, mutationRates._constructorMutations[1]._nodeProbability});
        return activeMutations;
    }
}

void MutationRatesWidget::process(MutationRatesDesc& mutationRates, float rightColumnWidth, bool nested)
{
    // Edit button spanning the field column, with a label to its right like the read-only fields below
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
}
