#include "SimulatedPreviewWidget.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "PreviewDescriptionWidget.h"
#include "WindowController.h"

SimulatedPreviewWidget _SimulatedPreviewWidget::create(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData)
{
    return SimulatedPreviewWidget(new _SimulatedPreviewWidget(simulationFacade, editData));
}

void _SimulatedPreviewWidget::process()
{
    if (!_lastGenome.has_value() || _lastGenome.value() != _editData->genome) {
        initPreview();
    }
    calcPreview();
    showPreview();

    _lastGenome = _editData->genome;
}

_SimulatedPreviewWidget::_SimulatedPreviewWidget(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData)
    : _simulationFacade(simulationFacade), _editData(editData)
{
    _previewWidget = _PreviewDescriptionWidget::create();
}

void _SimulatedPreviewWidget::initPreview()
{
    auto castratedGenome = _editData->genome;
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(castratedGenome);
    _simulationFacade->newPreview(castratedGenome);
}

void _SimulatedPreviewWidget::calcPreview()
{
    auto fps = WindowController::get().getFps();
    auto duration = std::chrono::milliseconds(1000 / fps / 2);
    _simulationFacade->calcTimestepsForPreview(duration);
}

namespace
{
    int calcTimestepsPerSecond(uint64_t lastTimestep, uint64_t timestep, std::chrono::milliseconds const& duration)
    {
        if (duration.count() == 0) {
            return 0;
        }
        uint64_t deltaTimesteps = (timestep > lastTimestep) ? (timestep - lastTimestep) : 0;
        double seconds = static_cast<double>(duration.count()) / 1000.0;
        if (seconds == 0.0) {
            return 0;
        }
        return static_cast<int>(static_cast<double>(deltaTimesteps) / seconds);
    }
}

void _SimulatedPreviewWidget::showPreview()
{
    auto now = std::chrono::steady_clock::now();
    auto preview = _simulationFacade->getPreviewData();

    int tps = 0;
    if (_lastPreviewTimestep.has_value() && _lastTimepoint.has_value()) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_lastTimepoint);
        if (duration.count() > 300) {
            tps = calcTimestepsPerSecond(_lastPreviewTimestep.value(), preview.timestep, duration);
            _lastPreviewTimestep = preview.timestep;
            _lastTimepoint = now;
            _lastTps = tps;
        } else {
            tps = _lastTps.value();
        }
    } else {
        _lastPreviewTimestep = preview.timestep;
        _lastTimepoint = now;
        _lastTps = tps;
    }
    _previewWidget->process(tps, preview.description);

}
