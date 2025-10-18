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

    std::filesystem::path const PointSmallShader = ShaderPath / "PointSmall";
    std::filesystem::path const PointLargeShader = ShaderPath / "PointLarge";
    std::filesystem::path const EnergyParticleShader = ShaderPath / "EnergyParticle";
    std::filesystem::path const LineShader = ShaderPath / "Line";
    std::filesystem::path const TriangleShader = ShaderPath / "Triangle";
    std::filesystem::path const BlurHorizontalShader = ShaderPath / "BlurHorizontal";
    std::filesystem::path const BlurVerticalShader = ShaderPath / "BlurVertical";
    std::filesystem::path const MetaballsShader = ShaderPath / "Metaballs";
    std::filesystem::path const FresnelShader = ShaderPath / "Fresnel";
    std::filesystem::path const SubsurfaceScatterShader = ShaderPath / "SubsurfaceScatter";
    std::filesystem::path const ThresholdShader = ShaderPath / "Threshold";
    std::filesystem::path const MergeShader = ShaderPath / "Merge";
    std::filesystem::path const MergeBlurShader = ShaderPath / "MergeBlur";
    std::filesystem::path const ToneMappingShader = ShaderPath / "ToneMapping";

    std::filesystem::path const EditorOnFilename = ImagesPath / "editor on.png";
    std::filesystem::path const EditorOffFilename = ImagesPath / "editor off.png";

    std::filesystem::path const LogoFilename = ImagesPath / "logo.png";
}
