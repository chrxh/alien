#include "BrowserDataTO.h"

bool _BrowserDataTO::isLeaf()
{
    return std::holds_alternative<BrowserLeaf>(node);
}

BrowserLeaf& _BrowserDataTO::getLeaf()
{
    return std::get<BrowserLeaf>(node);
}

BrowserFolder& _BrowserDataTO::getFolder()
{
    return std::get<BrowserFolder>(node);
}
