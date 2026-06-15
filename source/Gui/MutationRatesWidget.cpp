#include "MutationRatesWidget.h"

#include <initializer_list>
#include <string>
#include <vector>

#include <imgui.h>

#include <Base/StringHelper.h>
#include <EngineInterface/GenomeDesc.h>

#include "AlienGui.h"
#include "MutationRatesDialog.h"
#include "StyleRepository.h"

namespace
{
    void addActiveMutationType(std::vector<std::string>& result, std::string const& name, std::initializer_list<float> nodeProbabilities)
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
            result.push_back(name + ": " + probabilities);
        }
    }

    std::vector<std::string> getActiveMutationTypes(MutationRatesDesc const& mutationRates)
    {
        std::vector<std::string> activeMutations;
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
        addActiveMutationType(
            activeMutations, "Constructor", {mutationRates._constructorMutations[0]._nodeProbability, mutationRates._constructorMutations[1]._nodeProbability});
        return activeMutations;
    }
}

void MutationRatesWidget::process(MutationRatesDesc& mutationRates, bool nested)
{
    auto buttonWidth = scale(60.0f);
    auto availableWidth = ImGui::GetContentRegionAvail().x;
    auto listBoxWidth = availableWidth - buttonWidth - ImGui::GetStyle().ItemSpacing.x;
    AlienGui::ListBox(AlienGui::ListBoxParameters().items(getActiveMutationTypes(mutationRates)).width(listBoxWidth));
    ImGui::SameLine();
    if (AlienGui::Button("Edit")) {
        auto onAdopt = [&mutationRates](MutationRatesDesc const& adoptedRates) { mutationRates = adoptedRates; };
        if (nested) {
            MutationRatesDialog::get().openNested(mutationRates, onAdopt);
        } else {
            MutationRatesDialog::get().open(mutationRates, onAdopt);
        }
    }
}
