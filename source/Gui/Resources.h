#pragma once

namespace Const
{
/*
    auto const SimulationFragmentShader = "shader.fs";
    auto const SimulationVertexShader = "shader.vs";

    auto const AutosaveFile = "evolution.sim";

    auto const FontFilename = "DroidSans.ttf";

    auto const NavigationOnFilename = "navigation on.png";
    auto const NavigationOffFilename = "navigation off.png";
    auto const ActionOnFilename = "action on.png";
    auto const ActionOffFilename = "action off.png";

    auto const RunFilename = "run.png";
    auto const PauseFilename = "pause.png";
    auto const StepBackwardFilename = "step backward.png";
    auto const StepForwardFilename = "step forward.png";
    auto const SnapshotFilename = "snapshot.png";
    auto const RestoreFilname = "restore.png";
*/
    std::string const BasePath = "d:\\temp\\alien\\source\\Gui\\Resources\\";

    auto const SimulationFragmentShader = BasePath + "shader.fs";
    auto const SimulationVertexShader = BasePath + "shader.vs";

    auto const AutosaveFile = BasePath + "autosave.sim";

    auto const StandardFontFilename = "d:\\temp\\alien\\external\\imgui\\misc\\fonts\\DroidSans.ttf";
    auto const MonospaceFontFilename = "d:\\temp\\alien\\external\\imgui\\misc\\fonts\\Cousine-Regular.ttf";

    auto const NavigationOnFilename = BasePath + "navigation on.png";
    auto const NavigationOffFilename = BasePath + "navigation off.png";
    auto const ActionOnFilename = BasePath + "action on.png";
    auto const ActionOffFilename = BasePath + "action off.png";

    auto const RunFilename = BasePath + "run.png";
    auto const PauseFilename = BasePath + "pause.png";
    auto const StepBackwardFilename = BasePath + "step backward.png";
    auto const StepForwardFilename = BasePath + "step forward.png";
    auto const SnapshotFilename = BasePath + "snapshot.png";
    auto const RestoreFilname = BasePath + "restore.png";

    auto const ZoomInFilename = BasePath + "zoom in.png";
    auto const ZoomOutFilename = BasePath + "zoom out.png";
    auto const ResizeFilename = BasePath + "resize.png";

    auto const LogoFilename = BasePath + "logo.png";
}
