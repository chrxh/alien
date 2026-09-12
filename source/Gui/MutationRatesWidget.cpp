#include "MutationRatesWidget.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include <Base/StringHelper.h>
#include <EngineInterface/GenomeDesc.h>

#include "AlienGui.h"
#include "EntityAttributeHelp.h"
#include "StyleService.h"

namespace
{
    struct ActiveMutationType
    {
        std::string name;
        std::string probabilities;
        EntityAttribute attribute;
    };

    void addActiveMutationType(
        std::vector<ActiveMutationType>& result,
        std::string const& name,
        EntityAttribute attribute,
        std::initializer_list<float> nodeProbabilities)
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
            result.emplace_back(name, probabilities, attribute);
        }
    }

    std::vector<ActiveMutationType> getActiveMutationTypes(MutationRatesDesc const& mutationRates)
    {
        std::vector<ActiveMutationType> activeMutations;
        addActiveMutationType(
            activeMutations,
            "Connection mutations",
            EntityAttribute::MutationConnectionProbability,
            {mutationRates._connectionMutations[0]._nodeProbability, mutationRates._connectionMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Neuron mutations",
            EntityAttribute::MutationNeuronProbability,
            {mutationRates._neuronMutations[0]._nodeProbability, mutationRates._neuronMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Cell type property mut.",
            EntityAttribute::MutationCellTypePropertiesProbability,
            {mutationRates._cellTypePropertiesMutations[0]._nodeProbability, mutationRates._cellTypePropertiesMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Geometry mutations",
            EntityAttribute::MutationGeometryProbability,
            {mutationRates._geometryMutations[0]._geneProbability, mutationRates._geometryMutations[1]._geneProbability});
        addActiveMutationType(
            activeMutations, "Cell type mode mut.", EntityAttribute::MutationCellTypeModeProbability, {mutationRates._cellTypeModeMutation._nodeProbability});
        addActiveMutationType(
            activeMutations, "Cell type mutations", EntityAttribute::MutationCellTypeProbability, {mutationRates._cellTypeMutation._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Customization mutations",
            EntityAttribute::MutationCustomizationProbability,
            {mutationRates._customizationMutation._genomeProbability});
        addActiveMutationType(activeMutations, "Void mutations", EntityAttribute::MutationVoidProbability, {mutationRates._voidMutation._nodeProbability});
        addActiveMutationType(
            activeMutations, "Extend gene mutations", EntityAttribute::MutationExtendGeneProbability, {mutationRates._extendGeneMutation._geneProbability});
        addActiveMutationType(
            activeMutations, "Add node mutations", EntityAttribute::MutationAddNodeProbability, {mutationRates._addNodeMutation._nodeProbability});
        addActiveMutationType(
            activeMutations, "Trim gene mutations", EntityAttribute::MutationTrimGeneProbability, {mutationRates._trimGeneMutation._geneProbability});
        addActiveMutationType(
            activeMutations, "Delete node mutations", EntityAttribute::MutationDeleteNodeProbability, {mutationRates._deleteNodeMutation._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Duplicate gene mutations",
            EntityAttribute::MutationDuplicateGeneProbability,
            {mutationRates._duplicateGeneMutation._geneProbability});
        addActiveMutationType(
            activeMutations, "Delete gene mutations", EntityAttribute::MutationDeleteGeneProbability, {mutationRates._deleteGeneMutation._geneProbability});
        addActiveMutationType(
            activeMutations,
            "Copy node section mutations",
            EntityAttribute::MutationCopyNodeSectionProbability,
            {mutationRates._copyNodeSectionMutation._geneProbability});
        addActiveMutationType(
            activeMutations,
            "Move node section mutations",
            EntityAttribute::MutationMoveNodeSectionProbability,
            {mutationRates._moveNodeSectionMutation._geneProbability});
        addActiveMutationType(
            activeMutations,
            "Constructor",
            EntityAttribute::MutationConstructorProbability,
            {mutationRates._constructorMutations[0]._nodeProbability, mutationRates._constructorMutations[1]._nodeProbability});
        return activeMutations;
    }
}

void MutationRatesWidget::process(MutationRatesDesc& mutationRates, float rightColumnWidth, bool disabled)
{
    ImGui::BeginDisabled(disabled);
    ImGui::BeginGroup();

    if (AlienGui::Button(AlienGui::ButtonParameters()
                             .buttonText("Edit")
                             .name("Click to edit")
                             .textWidth(rightColumnWidth)
                             .tooltip(EntityAttributeHelp::get(EntityAttribute::GenomeMutationRatesEdit)))) {
        _dialog.open(mutationRates, [&mutationRates](MutationRatesDesc const& adoptedRates) { mutationRates = adoptedRates; });
    }

    for (auto const& [name, probabilities, attribute] : getActiveMutationTypes(mutationRates)) {
        auto value = probabilities;
        AlienGui::InputText(
            AlienGui::InputTextParameters().name(name).readOnly(true).textWidth(rightColumnWidth).tooltip(EntityAttributeHelp::get(attribute)), value);
    }
    ImGui::EndGroup();
    ImGui::EndDisabled();

    // The dialog is opened from here and therefore also processed here, but outside of the disabled scope, since
    // BeginDisabled() also applies to popups that are begun inside of it
    _dialog.process();
}
