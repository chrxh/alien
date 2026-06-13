#include "MutationRatesDialog.h"

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
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
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
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
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
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Sigma").id(id).min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &mutation._sigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Discrete change probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._discreteChangeProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processCellTypeModeMutationRate(std::string const& name, std::string const& id, CellTypeModeMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processCellTypeMutationRate(std::string const& name, std::string const& id, CellTypeMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processVoidMutationRate(std::string const& name, std::string const& id, VoidMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
        }
        AlienGui::EndTreeNode();
    }

    void processConstructorMutationRate(std::string const& name, std::string const& id, ConstructorMutationDesc& mutation, float rightColumnWidth)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name(name).rank(AlienGui::TreeNodeRank::Default))) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Node probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._nodeProbability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Sigma").id(id).min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &mutation._sigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Discrete change probability")
                    .id(id)
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &mutation._discreteChangeProbability);
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

MutationRatesDialog::MutationRatesDialog()
    : AlienDialog("Mutation rates", {800.0f, 400.0f})
{}

void MutationRatesDialog::initIntern() {}

void MutationRatesDialog::shutdownIntern() {}

void MutationRatesDialog::loadSettings(MutationRatesDesc& mutationRates, std::string const& settingsPrefix) const
{
    auto& settings = GlobalSettings::get();

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        mutationRates._connectionMutations[i]._nodeProbability =
            settings.getValue(settingsPrefix + "connection mutation " + indexSuffix + ".node probability", mutationRates._connectionMutations[i]._nodeProbability);
        mutationRates._connectionMutations[i]._sigma =
            settings.getValue(settingsPrefix + "connection mutation " + indexSuffix + ".sigma", mutationRates._connectionMutations[i]._sigma);

        mutationRates._neuronMutations[i]._nodeProbability =
            settings.getValue(settingsPrefix + "neuron mutation " + indexSuffix + ".node probability", mutationRates._neuronMutations[i]._nodeProbability);
        mutationRates._neuronMutations[i]._weightSigma =
            settings.getValue(settingsPrefix + "neuron mutation " + indexSuffix + ".weight sigma", mutationRates._neuronMutations[i]._weightSigma);
        mutationRates._neuronMutations[i]._biasSigma =
            settings.getValue(settingsPrefix + "neuron mutation " + indexSuffix + ".bias sigma", mutationRates._neuronMutations[i]._biasSigma);
        mutationRates._neuronMutations[i]._activationFunctionProbability = settings.getValue(
            settingsPrefix + "neuron mutation " + indexSuffix + ".activation function probability",
            mutationRates._neuronMutations[i]._activationFunctionProbability);
    }

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "" : " 2";

        mutationRates._cellTypePropertiesMutations[i]._nodeProbability = settings.getValue(
            settingsPrefix + "cell type property mutation" + indexSuffix + ".node probability", mutationRates._cellTypePropertiesMutations[i]._nodeProbability);
        mutationRates._cellTypePropertiesMutations[i]._sigma =
            settings.getValue(settingsPrefix + "cell type property mutation" + indexSuffix + ".sigma", mutationRates._cellTypePropertiesMutations[i]._sigma);
        mutationRates._cellTypePropertiesMutations[i]._discreteChangeProbability = settings.getValue(
            settingsPrefix + "cell type property mutation" + indexSuffix + ".value probability", mutationRates._cellTypePropertiesMutations[i]._discreteChangeProbability);
    }

    mutationRates._cellTypeModeMutation._nodeProbability =
        settings.getValue(settingsPrefix + "cell type mode mutation.node probability", mutationRates._cellTypeModeMutation._nodeProbability);
    mutationRates._cellTypeMutation._nodeProbability =
        settings.getValue(settingsPrefix + "cell type mutation.node probability", mutationRates._cellTypeMutation._nodeProbability);
    mutationRates._voidMutation._nodeProbability =
        settings.getValue(settingsPrefix + "void mutation.node probability", mutationRates._voidMutation._nodeProbability);

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        mutationRates._constructorMutations[i]._nodeProbability = settings.getValue(
            settingsPrefix + "constructor mutation " + indexSuffix + ".node probability", mutationRates._constructorMutations[i]._nodeProbability);
        mutationRates._constructorMutations[i]._sigma =
            settings.getValue(settingsPrefix + "constructor mutation " + indexSuffix + ".sigma", mutationRates._constructorMutations[i]._sigma);
        mutationRates._constructorMutations[i]._discreteChangeProbability = settings.getValue(
            settingsPrefix + "constructor mutation " + indexSuffix + ".value probability", mutationRates._constructorMutations[i]._discreteChangeProbability);
    }
}

void MutationRatesDialog::saveSettings(MutationRatesDesc const& mutationRates, std::string const& settingsPrefix) const
{
    auto& settings = GlobalSettings::get();

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        settings.setValue(settingsPrefix + "connection mutation " + indexSuffix + ".node probability", mutationRates._connectionMutations[i]._nodeProbability);
        settings.setValue(settingsPrefix + "connection mutation " + indexSuffix + ".sigma", mutationRates._connectionMutations[i]._sigma);

        settings.setValue(settingsPrefix + "neuron mutation " + indexSuffix + ".node probability", mutationRates._neuronMutations[i]._nodeProbability);
        settings.setValue(settingsPrefix + "neuron mutation " + indexSuffix + ".weight sigma", mutationRates._neuronMutations[i]._weightSigma);
        settings.setValue(settingsPrefix + "neuron mutation " + indexSuffix + ".bias sigma", mutationRates._neuronMutations[i]._biasSigma);
        settings.setValue(
            settingsPrefix + "neuron mutation " + indexSuffix + ".activation function probability",
            mutationRates._neuronMutations[i]._activationFunctionProbability);
    }

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        settings.setValue(
            settingsPrefix + "cell type property mutation " + indexSuffix + ".node probability", mutationRates._cellTypePropertiesMutations[i]._nodeProbability);
        settings.setValue(settingsPrefix + "cell type property mutation " + indexSuffix + ".sigma", mutationRates._cellTypePropertiesMutations[i]._sigma);
        settings.setValue(
            settingsPrefix + "cell type property mutation " + indexSuffix + ".value probability", mutationRates._cellTypePropertiesMutations[i]._discreteChangeProbability);
    }

    settings.setValue(settingsPrefix + "cell type mode mutation.node probability", mutationRates._cellTypeModeMutation._nodeProbability);
    settings.setValue(settingsPrefix + "cell type mutation.node probability", mutationRates._cellTypeMutation._nodeProbability);
    settings.setValue(settingsPrefix + "void mutation.node probability", mutationRates._voidMutation._nodeProbability);

    for (auto i = 0; i < 2; ++i) {
        auto const indexSuffix = i == 0 ? "1" : "2";

        settings.setValue(settingsPrefix + "constructor mutation " + indexSuffix + ".node probability", mutationRates._constructorMutations[i]._nodeProbability);
        settings.setValue(settingsPrefix + "constructor mutation " + indexSuffix + ".sigma", mutationRates._constructorMutations[i]._sigma);
        settings.setValue(settingsPrefix + "constructor mutation " + indexSuffix + ".value probability", mutationRates._constructorMutations[i]._discreteChangeProbability);
    }
}

void MutationRatesDialog::processIntern()
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

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Void mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processVoidMutationRate("Mutation rate", "VM", _mutation._voidMutation, rightColumnWidth);
                table.next();
            });
        }
        AlienGui::EndTreeNode();

        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Constructor mutations").rank(AlienGui::TreeNodeRank::High))) {
            processConcreteMutationRates([&](AlienGui::DynamicTableLayout& table) {
                processConstructorMutationRate("Mutation rate 1", "COM1", _mutation._constructorMutations[0], rightColumnWidth);
                table.next();
                processConstructorMutationRate("Mutation rate 2", "COM2", _mutation._constructorMutations[1], rightColumnWidth);
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

void MutationRatesDialog::open(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback)
{
    _mutation = mutationRates;
    _onAdoptCallback = onAdoptCallback;
    AlienDialog::open();
}

void MutationRatesDialog::openNested(MutationRatesDesc const& mutationRates, std::function<void(MutationRatesDesc const&)> const& onAdoptCallback)
{
    _mutation = mutationRates;
    _onAdoptCallback = onAdoptCallback;
    AlienDialog::openNested();
}

void MutationRatesDialog::openIntern() {}

void MutationRatesDialog::onAdopt()
{
    if (_onAdoptCallback) {
        _onAdoptCallback(_mutation);
    }
}
