#pragma once

#include <filesystem>

namespace Const
{
    std::string const ProgramVersion = "5.0.0-alpha.0";
    std::string const DiscordURL = "https://discord.gg/7bjyZdXXQ2";
    std::string const AlienURL = "alien-project.org";

    std::filesystem::path const ResourcePath = "resources";
    std::filesystem::path const ShaderPath = ResourcePath / "shaders";

    std::filesystem::path const LogFilename = "log.txt";
    std::filesystem::path const AutosaveFileWithoutPath = "autosave.sim";
    std::filesystem::path const AutosaveFile = ResourcePath / AutosaveFileWithoutPath;
    std::filesystem::path const SettingsFilename = ResourcePath / "settings.json";
    std::filesystem::path const SavepointTableFilename = "savepoints.json";

    std::filesystem::path const ObjectVertexShader = ShaderPath / "object.vs";
    std::filesystem::path const BackgroundObjectFragmentShader = ShaderPath / "background_object.fs";
    std::filesystem::path const ForegroundObjectFragmentShader = ShaderPath / "foreground_object.fs";
    std::filesystem::path const BlurHorizontalFragmentShader = ShaderPath / "blur_horizontal.fs";
    std::filesystem::path const BlurHorizontalVertexShader = ShaderPath / "blur_horizontal.vs";
    std::filesystem::path const BlurVerticalFragmentShader = ShaderPath / "blur_vertical.fs";
    std::filesystem::path const BlurVerticalVertexShader = ShaderPath / "blur_vertical.vs";
    std::filesystem::path const MetaballsFragmentShader = ShaderPath / "metaballs.fs";
    std::filesystem::path const MetaballsVertexShader = ShaderPath / "metaballs.vs";
    std::filesystem::path const SubsurfaceFragmentShader = ShaderPath / "subsurface.fs";
    std::filesystem::path const SubsurfaceVertexShader = ShaderPath / "subsurface.vs";
    std::filesystem::path const SubsurfaceScatterFragmentShader = ShaderPath / "subsurface_scatter.fs";
    std::filesystem::path const SubsurfaceScatterVertexShader = ShaderPath / "subsurface_scatter.vs";
    std::filesystem::path const FresnelFragmentShader = ShaderPath / "fresnel.fs";
    std::filesystem::path const FresnelVertexShader = ShaderPath / "fresnel.vs";
    std::filesystem::path const MergeFragmentShader = ShaderPath / "merge.fs";
    std::filesystem::path const MergeVertexShader = ShaderPath / "merge.vs";

    std::filesystem::path const EditorOnFilename = ResourcePath / "editor on.png";
    std::filesystem::path const EditorOffFilename = ResourcePath / "editor off.png";

    std::filesystem::path const LogoFilename = ResourcePath / "logo.png";
}
