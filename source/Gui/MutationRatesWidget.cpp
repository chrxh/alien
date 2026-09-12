#include "MutationRatesWidget.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include <Base/StringHelper.h>
#include <EngineInterface/GenomeDesc.h>

#include "AlienGui.h"
#include "CellAttributeHelp.h"
#include "StyleService.h"

namespace
{
    struct ActiveMutationType
    {
        std::string name;
        std::string probabilities;
        CellAttribute attribute;
    };

    void addActiveMutationType(
        std::vector<ActiveMutationType>& result,
        std::string const& name,
        CellAttribute attribute,
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
            CellAttribute::MutationConnectionProbability,
            {mutationRates._connectionMutations[0]._nodeProbability, mutationRates._connectionMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Neuron mutations",
            CellAttribute::MutationNeuronProbability,
            {mutationRates._neuronMutations[0]._nodeProbability, mutationRates._neuronMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Cell type property mut.",
            CellAttribute::MutationCellTypePropertiesProbability,
            {mutationRates._cellTypePropertiesMutations[0]._nodeProbability, mutationRates._cellTypePropertiesMutations[1]._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Geometry mutations",
            CellAttribute::MutationGeometryProbability,
            {mutationRates._geometryMutations[0]._geneProbability, mutationRates._geometryMutations[1]._geneProbability});
        addActiveMutationType(
            activeMutations, "Cell type mode mut.", CellAttribute::MutationCellTypeModeProbability, {mutationRates._cellTypeModeMutation._nodeProbability});
        addActiveMutationType(
            activeMutations, "Cell type mutations", CellAttribute::MutationCellTypeProbability, {mutationRates._cellTypeMutation._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Customization mutations",
            CellAttribute::MutationCustomizationProbability,
            {mutationRates._customizationMutation._genomeProbability});
        addActiveMutationType(activeMutations, "Void mutations", CellAttribute::MutationVoidProbability, {mutationRates._voidMutation._nodeProbability});
        addActiveMutationType(
            activeMutations, "Extend gene mutations", CellAttribute::MutationExtendGeneProbability, {mutationRates._extendGeneMutation._geneProbability});
        addActiveMutationType(
            activeMutations, "Add node mutations", CellAttribute::MutationAddNodeProbability, {mutationRates._addNodeMutation._nodeProbability});
        addActiveMutationType(
            activeMutations, "Trim gene mutations", CellAttribute::MutationTrimGeneProbability, {mutationRates._trimGeneMutation._geneProbability});
        addActiveMutationType(
            activeMutations, "Delete node mutations", CellAttribute::MutationDeleteNodeProbability, {mutationRates._deleteNodeMutation._nodeProbability});
        addActiveMutationType(
            activeMutations,
            "Duplicate gene mutations",
            CellAttribute::MutationDuplicateGeneProbability,
            {mutationRates._duplicateGeneMutation._geneProbability});
        addActiveMutationType(
            activeMutations, "Delete gene mutations", CellAttribute::MutationDeleteGeneProbability, {mutationRates._deleteGeneMutation._geneProbability});
        addActiveMutationType(
            activeMutations,
            "Copy node section mutations",
            CellAttribute::MutationCopyNodeSectionProbability,
            {mutationRates._copyNodeSectionMutation._geneProbability});
        addActiveMutationType(
            activeMutations,
            "Move node section mutations",
            CellAttribute::MutationMoveNodeSectionProbability,
            {mutationRates._moveNodeSectionMutation._geneProbability});
        addActiveMutationType(
            activeMutations,
            "Constructor",
            CellAttribute::MutationConstructorProbability,
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
                             .tooltip(CellAttributeHelp::get(CellAttribute::GenomeMutationRatesEdit)))) {
        _dialog.open(mutationRates, [&mutationRates](MutationRatesDesc const& adoptedRates) { mutationRates = adoptedRates; });
    }

    for (auto const& [name, probabilities, attribute] : getActiveMutationTypes(mutationRates)) {
        auto value = probabilities;
        AlienGui::InputText(
            AlienGui::InputTextParameters().name(name).readOnly(true).textWidth(rightColumnWidth).tooltip(CellAttributeHelp::get(attribute)), value);
    }
    ImGui::EndGroup();
    ImGui::EndDisabled();

    // The dialog is opened from here and therefore also processed here, but outside of the disabled scope, since
    // BeginDisabled() also applies to popups that are begun inside of it
    _dialog.process();
}
