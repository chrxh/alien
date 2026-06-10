#include "MutationRateDialog.h"

#include <imgui.h>

#include <Base/GlobalSettings.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 160.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;
    auto constexpr HeaderMinColumnWidth = 300.0f;

    void processConnectionMutationRate(std::string const& name, std::string const& id, ConnectionMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Event probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._eventProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Sigma").id(id).min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &mutation._sigma);
        }
        AlienGui::EndTreeNode();
    }

    void processNeuronMutationRate(std::string const& name, std::string const& id, NeuronMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Event probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._eventProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Weight sigma").id(id).min(0.0f).max(2.0f).logarithmic(true).format("%.2f").textWidth(rightColumnWidth),
                &mutation._weightSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Bias sigma").id(id).min(0.0f).max(2.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &mutation._biasSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("ActFn probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._activationFunctionProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processCellTypePropertiesMutationRate(std::string const& name, std::string const& id, CellTypePropertiesMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Event probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._eventProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Sigma").id(id).min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &mutation._sigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Probability").id(id).min(0.0f).max(1.0f).logarithmic(true).format("%.5f").textWidth(rightColumnWidth),
                &mutation._probability);
        }
        AlienGui::EndTreeNode();
    }

    void processCellTypeModeMutationRate(std::string const& name, std::string const& id, CellTypeModeMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Event probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._eventProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processCellTypeMutationRate(std::string const& name, std::string const& id, CellTypeMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Event probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._eventProbability);
        }
        AlienGui::EndTreeNode();
    }

    template <typename Func>
    void processConcreteMutationRates(Func&& processMutationRates)
    {
        AlienGui::DynamicTableLayout table(HeaderMinColumnWidth);
        if (table.begin()) {
            processMutationRates(table);
            table.end();
        }
    }
}

MutationRateDialog::MutationRateDialog()
    : AlienDialog("Mutation rates", {800.0f, 400.0f})
{}

void MutationRateDialog::initIntern() {}

void MutationRateDialog::shutdownIntern() {}

void MutationRateDialog::loadSettings(MutationRatesDesc& mutationRates, std::string const& settingsPrefix) const
{
    auto& settings = GlobalSettings::get();

    mutationRates._connectionMutations[0]._eventProbability =
        settings.getValue(settingsPrefix + "connection mutation 1.probability", mutationRates._connectionMutations[0]._eventProbability);
    mutationRates._connectionMutations[0]._sigma = settings.getValue(settingsPrefix + "connection mutation 1.sigma", mutationRates._connectionMutations[0]._sigma);
    mutationRates._connectionMutations[1]._eventProbability =
        settings.getValue(settingsPrefix + "connection mutation 2.probability", mutationRates._connectionMutations[1]._eventProbability);
    mutationRates._connectionMutations[1]._sigma = settings.getValue(settingsPrefix + "connection mutation 2.sigma", mutationRates._connectionMutations[1]._sigma);

    mutationRates._neuronMutations[0]._eventProbability =
        settings.getValue(settingsPrefix + "neuron mutation 1.probability", mutationRates._neuronMutations[0]._eventProbability);
    mutationRates._neuronMutations[0]._weightSigma =
        settings.getValue(settingsPrefix + "neuron mutation 1.weight sigma", mutationRates._neuronMutations[0]._weightSigma);
    mutationRates._neuronMutations[0]._biasSigma = settings.getValue(settingsPrefix + "neuron mutation 1.bias sigma", mutationRates._neuronMutations[0]._biasSigma);
    mutationRates._neuronMutations[0]._activationFunctionProbability =
        settings.getValue(settingsPrefix + "neuron mutation 1.activation function probability", mutationRates._neuronMutations[0]._activationFunctionProbability);

    mutationRates._neuronMutations[1]._eventProbability =
        settings.getValue(settingsPrefix + "neuron mutation 2.probability", mutationRates._neuronMutations[1]._eventProbability);
    mutationRates._neuronMutations[1]._weightSigma =
        settings.getValue(settingsPrefix + "neuron mutation 2.weight sigma", mutationRates._neuronMutations[1]._weightSigma);
    mutationRates._neuronMutations[1]._biasSigma = settings.getValue(settingsPrefix + "neuron mutation 2.bias sigma", mutationRates._neuronMutations[1]._biasSigma);
    mutationRates._neuronMutations[1]._activationFunctionProbability =
        settings.getValue(settingsPrefix + "neuron mutation 2.activation function probability", mutationRates._neuronMutations[1]._activationFunctionProbability);
    mutationRates._cellTypePropertiesMutations[0]._eventProbability =
        settings.getValue(settingsPrefix + "cell type property mutation.probability", mutationRates._cellTypePropertiesMutations[0]._eventProbability);
    mutationRates._cellTypePropertiesMutations[0]._sigma =
        settings.getValue(settingsPrefix + "cell type property mutation.sigma", mutationRates._cellTypePropertiesMutations[0]._sigma);
    mutationRates._cellTypePropertiesMutations[0]._probability =
        settings.getValue(settingsPrefix + "cell type property mutation.value probability", mutationRates._cellTypePropertiesMutations[0]._probability);
    mutationRates._cellTypePropertiesMutations[1]._eventProbability =
        settings.getValue(settingsPrefix + "cell type property mutation 2.probability", mutationRates._cellTypePropertiesMutations[1]._eventProbability);
    mutationRates._cellTypePropertiesMutations[1]._sigma =
        settings.getValue(settingsPrefix + "cell type property mutation 2.sigma", mutationRates._cellTypePropertiesMutations[1]._sigma);
    mutationRates._cellTypePropertiesMutations[1]._probability =
        settings.getValue(settingsPrefix + "cell type property mutation 2.value probability", mutationRates._cellTypePropertiesMutations[1]._probability);
    mutationRates._cellTypeModeMutation._eventProbability =
        settings.getValue(settingsPrefix + "cell type mode mutation.probability", mutationRates._cellTypeModeMutation._eventProbability);
    mutationRates._cellTypeMutation._eventProbability =
        settings.getValue(settingsPrefix + "cell type mutation.probability", mutationRates._cellTypeMutation._eventProbability);
}

void MutationRateDialog::saveSettings(MutationRatesDesc const& mutationRates, std::string const& settingsPrefix) const
{
    auto& settings = GlobalSettings::get();

    settings.setValue(settingsPrefix + "connection mutation 1.probability", mutationRates._connectionMutations[0]._eventProbability);
    settings.setValue(settingsPrefix + "connection mutation 1.sigma", mutationRates._connectionMutations[0]._sigma);
    settings.setValue(settingsPrefix + "connection mutation 2.probability", mutationRates._connectionMutations[1]._eventProbability);
    settings.setValue(settingsPrefix + "connection mutation 2.sigma", mutationRates._connectionMutations[1]._sigma);

    settings.setValue(settingsPrefix + "neuron mutation 1.probability", mutationRates._neuronMutations[0]._eventProbability);
    settings.setValue(settingsPrefix + "neuron mutation 1.weight sigma", mutationRates._neuronMutations[0]._weightSigma);
    settings.setValue(settingsPrefix + "neuron mutation 1.bias sigma", mutationRates._neuronMutations[0]._biasSigma);
    settings.setValue(settingsPrefix + "neuron mutation 1.activation function probability", mutationRates._neuronMutations[0]._activationFunctionProbability);

    settings.setValue(settingsPrefix + "neuron mutation 2.probability", mutationRates._neuronMutations[1]._eventProbability);
    settings.setValue(settingsPrefix + "neuron mutation 2.weight sigma", mutationRates._neuronMutations[1]._weightSigma);
    settings.setValue(settingsPrefix + "neuron mutation 2.bias sigma", mutationRates._neuronMutations[1]._biasSigma);
    settings.setValue(settingsPrefix + "neuron mutation 2.activation function probability", mutationRates._neuronMutations[1]._activationFunctionProbability);
    settings.setValue(settingsPrefix + "cell type property mutation.probability", mutationRates._cellTypePropertiesMutations[0]._eventProbability);
    settings.setValue(settingsPrefix + "cell type property mutation.sigma", mutationRates._cellTypePropertiesMutations[0]._sigma);
    settings.setValue(settingsPrefix + "cell type property mutation.value probability", mutationRates._cellTypePropertiesMutations[0]._probability);
    settings.setValue(settingsPrefix + "cell type property mutation 2.probability", mutationRates._cellTypePropertiesMutations[1]._eventProbability);
    settings.setValue(settingsPrefix + "cell type property mutation 2.sigma", mutationRates._cellTypePropertiesMutations[1]._sigma);
    settings.setValue(settingsPrefix + "cell type property mutation 2.value probability", mutationRates._cellTypePropertiesMutations[1]._probability);
    settings.setValue(settingsPrefix + "cell type mode mutation.probability", mutationRates._cellTypeModeMutation._eventProbability);
    settings.setValue(settingsPrefix + "cell type mutation.probability", mutationRates._cellTypeMutation._eventProbability);
}

void MutationRateDialog::processIntern()
{
    // Use a child window with scrolling for the content, reserving space for buttons
    auto buttonAreaHeight = scale(50.0f);
    if (ImGui::BeginChild("MutationRateContent", ImVec2(0, -buttonAreaHeight), false)) {
        auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Connection weight mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processConnectionMutationRate("Mutation rate 1", "CMR1", _mutation._connectionMutations[0], rightColumnWidth);
                table.next();
                processConnectionMutationRate("Mutation rate 2", "CMR2", _mutation._connectionMutations[1], rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Neuron mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processNeuronMutationRate("Mutation rate 1", "NMR1", _mutation._neuronMutations[0], rightColumnWidth);
                table.next();
                processNeuronMutationRate("Mutation rate 2", "NMR2", _mutation._neuronMutations[1], rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Cell type property mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processCellTypePropertiesMutationRate("Mutation rate 1", "CTPM1", _mutation._cellTypePropertiesMutations[0], rightColumnWidth);
                table.next();
                processCellTypePropertiesMutationRate("Mutation rate 2", "CTPM2", _mutation._cellTypePropertiesMutations[1], rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Cell type mode mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processCellTypeModeMutationRate("Mutation rate", "CTMM", _mutation._cellTypeModeMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Cell type mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processCellTypeMutationRate("Mutation rate", "CTM", _mutation._cellTypeMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();
    }
    ImGui::EndChild();

    AlienGui::Separator();

    if (AlienGui::Button("Adopt")) {
        ImGui::CloseCurrentPopup();
        onAdopt();
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
        close();
    }
}

void MutationRateDialog::open(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback)
{
    _mutation = mutationRates;
    _onAdoptCallback = onAdoptCallback;
    AlienDialog::open();
}

void MutationRateDialog::openNested(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback)
{
    _mutation = mutationRates;
    _onAdoptCallback = onAdoptCallback;
    AlienDialog::openNested();
}

void MutationRateDialog::openIntern() {}

void MutationRateDialog::onAdopt()
{
    if (_onAdoptCallback) {
        _onAdoptCallback(_mutation);
    }
}
