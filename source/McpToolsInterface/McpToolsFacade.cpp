#include "McpToolsFacade.h"

McpToolsFacade _McpToolsFacade::_instance;

McpToolsFacade _McpToolsFacade::get()
{
    return _instance;
}
