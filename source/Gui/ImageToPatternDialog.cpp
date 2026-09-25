#include "ImageToPatternDialog.h"

#include <imgui.h>

#include <Base/Definitions.h>
#include <Base/GlobalSettings.h>
#include <Base/ImageFileService.h>

#include <EngineInterface/CreatorService.h>
#include <EngineInterface/SimulationFacade.h>

#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "Viewport.h"

#include <ImFileDialog.h>


void ImageToPatternDialog::init()
{

    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::get().getValue("dialogs.open image.starting path", path.string());
}

void ImageToPatternDialog::shutdown()
{
    GlobalSettings::get().setValue("dialogs.open image.starting path", _startingPath);
}

void ImageToPatternDialog::show()
{
    GenericFileDialog::get().showOpenFileDialog("Open image", "Image (*.png){.png},.*", _startingPath, [&](std::filesystem::path const& path) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        auto image = ImageFileService::get().loadRgbImage(firstFilename);
        if (!image) {
            GenericMessageDialog::get().information("Error", "The image could not be read.");
            return;
        }
        _SimulationFacade::get()->addAndSelectSimulationData(CreatorService::get().createPatternFromImage(*image, Viewport::get().getCenterInWorldPos()));
        // TODO: update pattern editor
    });
}
