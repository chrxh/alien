#include "McpController.h"

#include <EngineInterface/SimulationFacade.h>

#include <Network/McpService.h>

#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "ImageFileService.h"
#include "MainLoopController.h"
#include "NewSimulationService.h"
#include "OverlayController.h"
#include "Viewport.h"

namespace
{
    class GuiMcpHost : public _McpHost
    {
    public:
        RealVector2D getVisibleAreaCenter() const override { return Viewport::get().getCenterInWorldPos(); }

        RealVector2D getVisibleAreaSize() const override
        {
            auto viewSize = Viewport::get().getViewSize();
            auto zoom = Viewport::get().getZoomFactor();
            return {toFloat(viewSize.x) / zoom, toFloat(viewSize.y) / zoom};
        }

        void createSimulation(std::string const& projectName, IntVector2D const& worldSize) override
        {
            NewSimulationService::get().createSimulation(NewSimulationService::Parameters()
                                                             .projectName(projectName)
                                                             .worldSize(worldSize)
                                                             .externalEnergy(_SimulationFacade::get()->getSimulationParameters().externalEnergy.value));
        }

        void onSelectionChanged() override { EditorModel::get().update(); }

        std::optional<RgbImage> loadImage(std::filesystem::path const& path) const override { return ImageFileService::get().loadRgbImage(path); }

        void showMessage(std::string const& message) override { printOverlayMessage(message); }

        void showError(std::string const& title, std::string const& message) override { GenericMessageDialog::get().information(title, message); }
    };
}

void McpController::init()
{
    McpService::get().init(std::make_shared<GuiMcpHost>());
}

void McpController::process()
{
    if (MainLoopController::get().isOperatingMode()) {
        McpService::get().processPendingCommands();
    }
}

void McpController::shutdown()
{
    McpService::get().shutdown();
}
