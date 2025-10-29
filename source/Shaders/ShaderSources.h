#pragma once

#include <string_view>

#include "Shaders/AllShaders.h"

namespace ShaderSources
{
    struct ShaderSource
    {
        std::string_view vertex;
        std::string_view fragment;
        std::string_view geometry;
    };

    inline ShaderSource const AttackEvent{Shaders::AttackEventVS, Shaders::AttackEventFS, Shaders::AttackEventGS};
    inline ShaderSource const Background{Shaders::BackgroundVS, Shaders::BackgroundFS};
    inline ShaderSource const BlurHorizontal{Shaders::BlurHorizontalVS, Shaders::BlurHorizontalFS};
    inline ShaderSource const BlurVertical{Shaders::BlurVerticalVS, Shaders::BlurVerticalFS};
    inline ShaderSource const Cell{Shaders::CellVS, Shaders::CellFS};
    inline ShaderSource const CellTypeOverlay{Shaders::CellTypeOverlayVS, Shaders::CellTypeOverlayFS, Shaders::CellTypeOverlayGS};
    inline ShaderSource const DetonationEvent{Shaders::DetonationEventVS, Shaders::DetonationEventFS, Shaders::DetonationEventGS};
    inline ShaderSource const DeNoise{Shaders::DeNoiseVS, Shaders::DeNoiseFS};
    inline ShaderSource const DownSampler{Shaders::DownSamplerVS, Shaders::DownSamplerFS};
    inline ShaderSource const EnergyParticle{Shaders::EnergyParticleVS, Shaders::EnergyParticleFS};
    inline ShaderSource const Fresnel{Shaders::FresnelVS, Shaders::FresnelFS};
    inline ShaderSource const Line{Shaders::LineVS, Shaders::LineFS, Shaders::LineGS};
    inline ShaderSource const Location{Shaders::LocationVS, Shaders::LocationFS, Shaders::LocationGS};
    inline ShaderSource const MergeAdditive{Shaders::MergeAdditiveVS, Shaders::MergeAdditiveFS};
    inline ShaderSource const MergeLayers{Shaders::MergeLayersVS, Shaders::MergeLayersFS};
    inline ShaderSource const MergeMax{Shaders::MergeMaxVS, Shaders::MergeMaxFS};
    inline ShaderSource const Metaballs{Shaders::MetaballsVS, Shaders::MetaballsFS};
    inline ShaderSource const ModuloCopy{Shaders::ModuloCopyVS, Shaders::ModuloCopyFS};
    inline ShaderSource const SelectedConnection{Shaders::SelectedConnectionVS, Shaders::SelectedConnectionFS, Shaders::SelectedConnectionGS};
    inline ShaderSource const SelectedObject{Shaders::SelectedObjectVS, Shaders::SelectedObjectFS, Shaders::SelectedObjectGS};
    inline ShaderSource const SubsurfaceScatter{Shaders::SubsurfaceScatterVS, Shaders::SubsurfaceScatterFS};
    inline ShaderSource const Threshold{Shaders::ThresholdVS, Shaders::ThresholdFS};
    inline ShaderSource const ToneMapping{Shaders::ToneMappingVS, Shaders::ToneMappingFS};
    inline ShaderSource const Triangle{Shaders::TriangleVS, Shaders::TriangleFS, Shaders::TriangleGS};
    inline ShaderSource const UpSampler{Shaders::UpSamplerVS, Shaders::UpSamplerFS};
    inline ShaderSource const ZoomBrightnessCorrection{Shaders::ZoomBrightnessCorrectionVS, Shaders::ZoomBrightnessCorrectionFS};
}
