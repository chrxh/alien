#pragma once

#include <filesystem>

#include "Gui/ShaderSources.h"

namespace Const
{
    std::string const ProgramVersion = "5.0.0-alpha.1";
    std::string const DiscordURL = "https://discord.gg/7bjyZdXXQ2";
    std::string const AlienURL = "alien-project.org";

    std::filesystem::path const ResourcePath = "resources";
    std::filesystem::path const AutosavePath = ResourcePath / "autosave";
    std::filesystem::path const ImagesPath = ResourcePath / "images";

    std::filesystem::path const LogFilename = "log.txt";
    std::filesystem::path const AutosaveFileWithoutPath = "autosave.sim";
    std::filesystem::path const AutosaveFile = AutosavePath / AutosaveFileWithoutPath;
    std::filesystem::path const SettingsFilename = AutosavePath / "settings.json";
    std::filesystem::path const SavepointTableFilename = "savepoints.json";

    inline ShaderSources::ShaderSource const& BackgroundShader = ShaderSources::Background;
    inline ShaderSources::ShaderSource const& CellShader = ShaderSources::Cell;
    inline ShaderSources::ShaderSource const& EnergyParticleShader = ShaderSources::EnergyParticle;
    inline ShaderSources::ShaderSource const& LineShader = ShaderSources::Line;
    inline ShaderSources::ShaderSource const& TriangleShader = ShaderSources::Triangle;
    inline ShaderSources::ShaderSource const& AttackEventShader = ShaderSources::AttackEvent;
    inline ShaderSources::ShaderSource const& LocationShader = ShaderSources::Location;
    inline ShaderSources::ShaderSource const& SelectedObjectShader = ShaderSources::SelectedObject;
    inline ShaderSources::ShaderSource const& SelectedConnectionShader = ShaderSources::SelectedConnection;
    inline ShaderSources::ShaderSource const& CellTypeOverlayShader = ShaderSources::CellTypeOverlay;
    inline ShaderSources::ShaderSource const& BlurHorizontalShader = ShaderSources::BlurHorizontal;
    inline ShaderSources::ShaderSource const& BlurVerticalShader = ShaderSources::BlurVertical;
    inline ShaderSources::ShaderSource const& MetaballsShader = ShaderSources::Metaballs;
    inline ShaderSources::ShaderSource const& FresnelShader = ShaderSources::Fresnel;
    inline ShaderSources::ShaderSource const& SubsurfaceScatterShader = ShaderSources::SubsurfaceScatter;
    inline ShaderSources::ShaderSource const& ThresholdShader = ShaderSources::Threshold;
    inline ShaderSources::ShaderSource const& MergeAdditiveShader = ShaderSources::MergeAdditive;
    inline ShaderSources::ShaderSource const& MergeLayersShader = ShaderSources::MergeLayers;
    inline ShaderSources::ShaderSource const& MergeMaxShader = ShaderSources::MergeMax;
    inline ShaderSources::ShaderSource const& DownSamplerShader = ShaderSources::DownSampler;
    inline ShaderSources::ShaderSource const& UpSamplerShader = ShaderSources::UpSampler;
    inline ShaderSources::ShaderSource const& ToneMappingShader = ShaderSources::ToneMapping;
    inline ShaderSources::ShaderSource const& ZoomBrightnessCorrectionShader = ShaderSources::ZoomBrightnessCorrection;

    std::filesystem::path const EditorOnFilename = ImagesPath / "editor on.png";
    std::filesystem::path const EditorOffFilename = ImagesPath / "editor off.png";

    std::filesystem::path const LogoFilename = ImagesPath / "logo.png";
}
