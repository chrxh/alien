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
}

MutationRateDialog::MutationRateDialog()
    : AlienDialog("Mutation rates", {800.0f, 400.0f})
{}

void MutationRateDialog::initIntern() {}

void MutationRateDialog::shutdownIntern() {}

void MutationRateDialog::loadSettings(MutationRatesDesc& mutationRates, std::string const& settingsPrefix) const
{
    auto& settings = GlobalSettings::get();

    mutationRates._lineageMutationProbability = settings.getValue(settingsPrefix + "lineage mutation probability", mutationRates._lineageMutationProbability);

    mutationRates._connectionMutation1._probability =
        settings.getValue(settingsPrefix + "connection mutation 1.probability", mutationRates._connectionMutation1._probability);
    mutationRates._connectionMutation1._sigma = settings.getValue(settingsPrefix + "connection mutation 1.sigma", mutationRates._connectionMutation1._sigma);
    mutationRates._connectionMutation2._probability =
        settings.getValue(settingsPrefix + "connection mutation 2.probability", mutationRates._connectionMutation2._probability);
    mutationRates._connectionMutation2._sigma = settings.getValue(settingsPrefix + "connection mutation 2.sigma", mutationRates._connectionMutation2._sigma);

    mutationRates._neuronMutation1._probability =
        settings.getValue(settingsPrefix + "neuron mutation 1.probability", mutationRates._neuronMutation1._probability);
    mutationRates._neuronMutation1._weightSigma =
        settings.getValue(settingsPrefix + "neuron mutation 1.weight sigma", mutationRates._neuronMutation1._weightSigma);
    mutationRates._neuronMutation1._biasSigma = settings.getValue(settingsPrefix + "neuron mutation 1.bias sigma", mutationRates._neuronMutation1._biasSigma);
    mutationRates._neuronMutation1._activationFunctionProbability =
        settings.getValue(settingsPrefix + "neuron mutation 1.activation function probability", mutationRates._neuronMutation1._activationFunctionProbability);

    mutationRates._neuronMutation2._probability =
        settings.getValue(settingsPrefix + "neuron mutation 2.probability", mutationRates._neuronMutation2._probability);
    mutationRates._neuronMutation2._weightSigma =
        settings.getValue(settingsPrefix + "neuron mutation 2.weight sigma", mutationRates._neuronMutation2._weightSigma);
    mutationRates._neuronMutation2._biasSigma = settings.getValue(settingsPrefix + "neuron mutation 2.bias sigma", mutationRates._neuronMutation2._biasSigma);
    mutationRates._neuronMutation2._activationFunctionProbability =
        settings.getValue(settingsPrefix + "neuron mutation 2.activation function probability", mutationRates._neuronMutation2._activationFunctionProbability);
}

void MutationRateDialog::saveSettings(MutationRatesDesc const& mutationRates, std::string const& settingsPrefix) const
{
    auto& settings = GlobalSettings::get();

    settings.setValue(settingsPrefix + "lineage mutation probability", mutationRates._lineageMutationProbability);

    settings.setValue(settingsPrefix + "connection mutation 1.probability", mutationRates._connectionMutation1._probability);
    settings.setValue(settingsPrefix + "connection mutation 1.sigma", mutationRates._connectionMutation1._sigma);
    settings.setValue(settingsPrefix + "connection mutation 2.probability", mutationRates._connectionMutation2._probability);
    settings.setValue(settingsPrefix + "connection mutation 2.sigma", mutationRates._connectionMutation2._sigma);

    settings.setValue(settingsPrefix + "neuron mutation 1.probability", mutationRates._neuronMutation1._probability);
    settings.setValue(settingsPrefix + "neuron mutation 1.weight sigma", mutationRates._neuronMutation1._weightSigma);
    settings.setValue(settingsPrefix + "neuron mutation 1.bias sigma", mutationRates._neuronMutation1._biasSigma);
    settings.setValue(settingsPrefix + "neuron mutation 1.activation function probability", mutationRates._neuronMutation1._activationFunctionProbability);

    settings.setValue(settingsPrefix + "neuron mutation 2.probability", mutationRates._neuronMutation2._probability);
    settings.setValue(settingsPrefix + "neuron mutation 2.weight sigma", mutationRates._neuronMutation2._weightSigma);
    settings.setValue(settingsPrefix + "neuron mutation 2.bias sigma", mutationRates._neuronMutation2._biasSigma);
    settings.setValue(settingsPrefix + "neuron mutation 2.activation function probability", mutationRates._neuronMutation2._activationFunctionProbability);
}

void MutationRateDialog::processIntern()
{
    // Use a child window with scrolling for the content, reserving space for buttons
    auto buttonAreaHeight = scale(50.0f);
    if (ImGui::BeginChild("MutationRateContent", ImVec2(0, -buttonAreaHeight), false)) {
        AlienGui::DynamicTableLayout table(HeaderMinColumnWidth);
        if (table.begin()) {
            auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));

            AlienGui::Group(AlienGui::GroupParameters().text("Connection weight mutation rate 1"));
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Probability")
                    .id("CMR1")
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &_mutation._connectionMutation1._probability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Sigma").id("CMR1").min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &_mutation._connectionMutation1._sigma);

            AlienGui::Group(AlienGui::GroupParameters().text("Connection weight mutation rate 2"));
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Probability")
                    .id("CMR2")
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &_mutation._connectionMutation2._probability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Sigma").id("CMR2").min(0.0f).max(1.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &_mutation._connectionMutation2._sigma);
            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Neuron weight mutation rate 1"));
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Probability")
                    .id("NMR1")
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &_mutation._neuronMutation1._probability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Weight sigma")
                    .id("NMR1")
                    .min(0.0f)
                    .max(2.0f)
                    .logarithmic(true)
                    .format("%.2f")
                    .textWidth(rightColumnWidth),
                &_mutation._neuronMutation1._weightSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Bias sigma").id("NMR1").min(0.0f).max(2.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &_mutation._neuronMutation1._biasSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("ActFn probability")
                    .id("NMR1")
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &_mutation._neuronMutation1._activationFunctionProbability);
            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Neuron weight mutation rate 2"));
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Probability")
                    .id("NMR2")
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &_mutation._neuronMutation2._probability);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Weight sigma")
                    .id("NMR2")
                    .min(0.0f)
                    .max(2.0f)
                    .logarithmic(true)
                    .format("%.2f")
                    .textWidth(rightColumnWidth),
                &_mutation._neuronMutation2._weightSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Bias sigma").id("NMR2").min(0.0f).max(2.0f).logarithmic(true).format("%.3f").textWidth(rightColumnWidth),
                &_mutation._neuronMutation2._biasSigma);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("ActFn probability")
                    .id("NMR2")
                    .min(0.0f)
                    .max(1.0f)
                    .logarithmic(true)
                    .format("%.5f")
                    .textWidth(rightColumnWidth),
                &_mutation._neuronMutation2._activationFunctionProbability);
            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Lineage mutation rate"));
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Probability").min(0.0f).max(1.0f).logarithmic(true).format("%.5f").textWidth(rightColumnWidth),
                &_mutation._lineageMutationProbability);
            table.next();

            table.end();
        }
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
