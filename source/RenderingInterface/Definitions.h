#pragma once

#include <memory>

struct GLFWwindow;

class _RenderingFacade;
using RenderingFacade = std::shared_ptr<_RenderingFacade>;
