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
            activeMutations, _("Connection mutations"), {mutationRates._connectionMutations[0]._nodeProbability, mutationRates._connectionMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations, _("Neuron mutations"), {mutationRates._neuronMutations[0]._nodeProbability, mutationRates._neuronMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            _("Cell type property mut."),
            {mutationRates._cellTypePropertiesMutations[0]._nodeProbability, mutationRates._cellTypePropertiesMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations, _("Geometry mutations"), {mutationRates._geometryMutations[0]._geneProbability, mutationRates._geometryMutations[1]._geneProbability});
        addActiveMutationType(activeMutations, _("Cell type mode mut."), {mutationRates._cellTypeModeMutation._nodeProbability});
        addActiveMutationType(activeMutations, _("Cell type mutations"), {mutationRates._cellTypeMutation._nodeProbability});
        addActiveMutationType(activeMutations, _("Void mutations"), {mutationRates._voidMutation._nodeProbability});
        addActiveMutationType(activeMutations, _("Extend gene mutations"), {mutationRates._extendGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, _("Add node mutations"), {mutationRates._addNodeMutation._nodeProbability});
        addActiveMutationType(activeMutations, _("Trim gene mutations"), {mutationRates._trimGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, _("Delete node mutations"), {mutationRates._deleteNodeMutation._nodeProbability});
        addActiveMutationType(activeMutations, _("Duplicate gene mutations"), {mutationRates._duplicateGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, _("Delete gene mutations"), {mutationRates._deleteGeneMutation._geneProbability});
        addActiveMutationType(activeMutations, _("Copy node section mutations"), {mutationRates._copyNodeSectionMutation._geneProbability});
        addActiveMutationType(activeMutations, _("Move node section mutations"), {mutationRates._moveNodeSectionMutation._geneProbability});
        addActiveMutationType(
            activeMutations, _("Constructor"), {mutationRates._constructorMutations[0]._nodeProbability, mutationRates._constructorMutations[1]._nodeProbability});
        return activeMutations;
    }
}

void MutationRatesWidget::process(MutationRatesDesc& mutationRates, float rightColumnWidth, bool nested)
{
    ImGui::BeginGroup();

    if (AlienGui::Button(AlienGui::ButtonParameters().buttonText(_("Edit")).name(_("Click to edit")).textWidth(rightColumnWidth))) {
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
