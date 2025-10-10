#pragma once

#include <filesystem>

namespace Const
{
    std::string const ProgramVersion = "5.0.0-alpha.0";
    std::string const DiscordURL = "https://discord.gg/7bjyZdXXQ2";
    std::string const AlienURL = "alien-project.org";

    std::filesystem::path const ResourcePath = "resources";
    std::filesystem::path const AutosavePath = ResourcePath / "autosave";
    std::filesystem::path const ImagesPath = ResourcePath / "images";
    std::filesystem::path const ShaderPath = ResourcePath / "shaders";

    std::filesystem::path const LogFilename = "log.txt";
    std::filesystem::path const AutosaveFileWithoutPath = "autosave.sim";
    std::filesystem::path const AutosaveFile = AutosavePath / AutosaveFileWithoutPath;
    std::filesystem::path const SettingsFilename = AutosavePath / "settings.json";
    std::filesystem::path const SavepointTableFilename = "savepoints.json";

    std::filesystem::path const ObjectBackgroundVertexShader = ShaderPath / "ObjectBackground.vs";
    std::filesystem::path const ObjectBackgroundFragmentShader = ShaderPath / "ObjectBackground.fs";

    std::filesystem::path const ObjectForegroundVertexShader = ShaderPath / "ObjectForeground.vs";
    std::filesystem::path const ObjectForegroundFragmentShader = ShaderPath / "ObjectForeground.fs";

    std::filesystem::path const BlurHorizontalVertexShader = ShaderPath / "BlurHorizontal.vs";
    std::filesystem::path const BlurHorizontalFragmentShader = ShaderPath / "BlurHorizontal.fs";

    std::filesystem::path const BlurVerticalVertexShader = ShaderPath / "BlurVertical.vs";
    std::filesystem::path const BlurVerticalFragmentShader = ShaderPath / "BlurVertical.fs";

    std::filesystem::path const MetaballsVertexShader = ShaderPath / "Metaballs.vs";
    std::filesystem::path const MetaballsFragmentShader = ShaderPath / "Metaballs.fs";

    std::filesystem::path const FresnelVertexShader = ShaderPath / "Fresnel.vs";
    std::filesystem::path const FresnelFragmentShader = ShaderPath / "Fresnel.fs";

    std::filesystem::path const SubsurfaceScatterVertexShader = ShaderPath / "SubsurfaceScatter.vs";
    std::filesystem::path const SubsurfaceScatterFragmentShader = ShaderPath / "SubsurfaceScatter.fs";

    std::filesystem::path const MergeVertexShader = ShaderPath / "Merge.vs";
    std::filesystem::path const MergeFragmentShader = ShaderPath / "Merge.fs";

    std::filesystem::path const EditorOnFilename = ImagesPath / "editor on.png";
    std::filesystem::path const EditorOffFilename = ImagesPath / "editor off.png";

    std::filesystem::path const LogoFilename = ImagesPath / "logo.png";
}
