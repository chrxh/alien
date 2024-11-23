#pragma once

#include "Base/Definitions.h"

class _PersisterFacade;
using PersisterFacade = std::shared_ptr<_PersisterFacade>;

class _TaskProcessor;
using TaskProcessor = std::shared_ptr<_TaskProcessor>;

class SavepointTable;
class SavepointTableService;