#pragma once

#include <boost/shared_ptr.hpp>
#include <boost/optional.hpp>

class _MainWindow;
using MainWindow = boost::shared_ptr<_MainWindow>;

class _SimulationView;
using SimulationView= boost::shared_ptr<_SimulationView>;

class _Shader;
using Shader = boost::shared_ptr<_Shader>;

class _SimulationScrollbar;
using SimulationScrollbar = boost::shared_ptr<_SimulationScrollbar>;

class _Viewport;
using Viewport = boost::shared_ptr<_Viewport>;

class _StyleRepository;
using StyleRepository = boost::shared_ptr<_StyleRepository>;

class _TemporalControlWindow;
using TemporalControlWindow = boost::shared_ptr<_TemporalControlWindow>;

struct GLFWwindow;
struct ImFont;

struct TextureData
{
    unsigned int textureId;
    int width;
    int height;
};
